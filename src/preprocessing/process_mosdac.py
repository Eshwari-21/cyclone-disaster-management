from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOSDAC_DIR = PROJECT_ROOT / "data" / "raw" / "mosdac"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "mosdac"
    / "mosdac_gpi_points.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Study-area bounding box
# ---------------------------------------------------------
MIN_LON = 81.903123
MAX_LON = 84.765077
MIN_LAT = 16.672423
MAX_LAT = 19.166914


# ---------------------------------------------------------
# Find MOSDAC GPI files
# ---------------------------------------------------------
files = sorted(
    MOSDAC_DIR.glob("3RIMG_*_L2G_GPI_*.h5")
)

if not files:
    raise FileNotFoundError(
        f"No MOSDAC GPI HDF5 files found in {MOSDAC_DIR}"
    )

print(f"Found {len(files)} MOSDAC GPI files.")


# ---------------------------------------------------------
# Read each MOSDAC GPI file
# ---------------------------------------------------------
records = []

for i, file_path in enumerate(files, start=1):

    print(f"[{i}/{len(files)}] Processing {file_path.name}")

    with h5py.File(file_path, "r") as f:

        # -------------------------------------------------
        # Read GPI rainfall data and coordinates
        # -------------------------------------------------
        gpi = f["GPI"][:]
        lat = f["latitude"][:]
        lon = f["longitude"][:]

        # -------------------------------------------------
        # Remove single time dimension
        # (1, lat, lon) -> (lat, lon)
        # -------------------------------------------------
        gpi = np.squeeze(gpi)

        # -------------------------------------------------
        # Get MOSDAC fill value
        # -------------------------------------------------
        fill_value = f["GPI"].attrs.get(
            "_FillValue",
            -999.0
        )

        # Convert to floating point so NaN can be used
        gpi = gpi.astype("float32")

        # Only the fill value is treated as missing.
        # IMPORTANT: 0 is a valid rainfall value.
        gpi[gpi == fill_value] = np.nan

        # -------------------------------------------------
        # Extract timestamp from filename
        #
        # Example:
        # 3RIMG_30AUG2024_0015_L2G_GPI_V01R00.h5
        #
        # -> 30AUG2024 + 0015
        # -------------------------------------------------
        match = re.search(
            r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_",
            file_path.name
        )

        if not match:
            raise ValueError(
                f"Could not extract timestamp from {file_path.name}"
            )

        timestamp = pd.to_datetime(
            match.group(1) + match.group(2),
            format="%d%b%Y%H%M"
        )

        # -------------------------------------------------
        # Create coordinate grid
        # -------------------------------------------------
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # -------------------------------------------------
        # Flatten arrays
        # -------------------------------------------------
        gpi_flat = gpi.ravel()
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()

        # -------------------------------------------------
        # Keep only valid GPI values
        # -------------------------------------------------
        valid = np.isfinite(gpi_flat)

        df = pd.DataFrame(
            {
                "timestamp_utc": timestamp,
                "latitude": lat_flat[valid],
                "longitude": lon_flat[valid],
                "gpi_mm": gpi_flat[valid],
            }
        )

        # -------------------------------------------------
        # Keep only points inside study-area bounding box
        # -------------------------------------------------
        df = df[
            (df["longitude"] >= MIN_LON)
            & (df["longitude"] <= MAX_LON)
            & (df["latitude"] >= MIN_LAT)
            & (df["latitude"] <= MAX_LAT)
        ].copy()

        records.append(df)

        # -------------------------------------------------
        # Print validation information
        # -------------------------------------------------
        if len(df) > 0:
            print(
                f"    Study-area points: {len(df):,} | "
                f"GPI range: "
                f"{df['gpi_mm'].min():.3f} - "
                f"{df['gpi_mm'].max():.3f} mm"
            )
        else:
            print("    Study-area points: 0")


# ---------------------------------------------------------
# Combine all files
# ---------------------------------------------------------
if not records:
    raise ValueError(
        "No valid GPI observations were produced."
    )

result = pd.concat(
    records,
    ignore_index=True
)


# ---------------------------------------------------------
# Sort observations
# ---------------------------------------------------------
result = result.sort_values(
    [
        "timestamp_utc",
        "latitude",
        "longitude"
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# Save intermediate dataset
# ---------------------------------------------------------
result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------
print()
print("MOSDAC GPI preprocessing complete.")
print(f"Rows: {len(result):,}")
print(f"Files processed: {len(files)}")
print(f"Output: {OUTPUT_FILE}")

print(
    f"GPI range: "
    f"{result['gpi_mm'].min():.3f} - "
    f"{result['gpi_mm'].max():.3f} mm"
)

print(
    f"Unique timestamps: "
    f"{result['timestamp_utc'].nunique():,}"
)