from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats


# ============================================================
# GEE ENVIRONMENTAL RASTER → 1-KM GRID
# Cyclone Disaster Management Project
# Year: 2024
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coastal_ap_grid_1km.gpkg"
)

GEE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gee"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gee_environmental_grid_2024.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. Load 1-km grid
# ------------------------------------------------------------

print("=" * 60)
print("LOADING 1-KM GRID")
print("=" * 60)

grid = gpd.read_file(GRID_FILE)

if grid.crs is None:
    raise ValueError("Grid has no CRS.")

print("Grid cells:", len(grid))
print("Grid CRS:", grid.crs)

if "grid_id" not in grid.columns:
    raise ValueError(
        "grid_id column not found in grid."
    )

if grid["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found."
    )


# ------------------------------------------------------------
# 3. Raster files
# ------------------------------------------------------------

rasters = {
    "elevation_m":
        GEE_DIR / "coastal_ap_elevation_2024.tif",

    "slope_deg":
        GEE_DIR / "coastal_ap_slope_2024_final.tif",

    "annual_rainfall_mm":
        GEE_DIR / "coastal_ap_rainfall_2024.tif",

    "mean_ndvi":
        GEE_DIR / "coastal_ap_ndvi_2024.tif",
}


# ------------------------------------------------------------
# 4. Check raster files
# ------------------------------------------------------------

print()
print("=" * 60)
print("CHECKING RASTER FILES")
print("=" * 60)

for column, raster_path in rasters.items():

    print(f"\n{column}")
    print("File:", raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster file not found: {raster_path}"
        )

    print(
        "Size:",
        raster_path.stat().st_size,
        "bytes"
    )


# ------------------------------------------------------------
# 5. Extract raster means for every 1-km grid cell
# ------------------------------------------------------------

print()
print("=" * 60)
print("EXTRACTING GEE VALUES TO 1-KM GRID")
print("=" * 60)

for column, raster_path in rasters.items():

    print()
    print(f"Processing: {raster_path.name}")

    # --------------------------------------------------------
    # Use rasterstats to calculate the mean of valid raster
    # pixels inside each project grid cell.
    #
    # NaN pixels are ignored by rasterstats.
    #
    # all_touched=False means we use raster pixels whose
    # centers fall inside the grid cell rather than every
    # pixel that merely touches the boundary.
    # --------------------------------------------------------

    stats = zonal_stats(
        grid.geometry,
        str(raster_path),
        stats=["mean"],
        nodata=np.nan,
        all_touched=False
    )

    values = []

    for result in stats:

        value = result["mean"]

        if value is None:
            values.append(np.nan)
        else:
            values.append(float(value))

    grid[column] = values

    # --------------------------------------------------------
    # Per-raster validation
    # --------------------------------------------------------

    missing_count = grid[column].isna().sum()
    valid_count = grid[column].notna().sum()

    print(
        "Valid grid cells:",
        valid_count
    )

    print(
        "Missing grid cells:",
        missing_count
    )

    if valid_count > 0:

        print(
            "Minimum:",
            grid[column].min()
        )

        print(
            "Maximum:",
            grid[column].max()
        )

        print(
            "Mean:",
            grid[column].mean()
        )


# ------------------------------------------------------------
# 6. Create final environmental table
# ------------------------------------------------------------

output = grid[
    [
        "grid_id",
        "elevation_m",
        "slope_deg",
        "annual_rainfall_mm",
        "mean_ndvi"
    ]
].copy()


# ------------------------------------------------------------
# 7. Validate grid IDs
# ------------------------------------------------------------

print()
print("=" * 60)
print("VALIDATING GRID")
print("=" * 60)

print(
    "Rows:",
    len(output)
)

print(
    "Unique grid IDs:",
    output["grid_id"].nunique()
)

print(
    "Duplicate grid IDs:",
    output["grid_id"].duplicated().sum()
)

if len(output) != 17737:
    print(
        "WARNING: Expected 17,737 grid rows."
    )


# ------------------------------------------------------------
# 8. Missing-value validation
# ------------------------------------------------------------

print()
print("=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print(output.isna().sum())


# ------------------------------------------------------------
# 9. Environmental feature summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("ENVIRONMENTAL FEATURE SUMMARY")
print("=" * 60)

print(output.describe())


# ------------------------------------------------------------
# 10. Slope validation
# ------------------------------------------------------------

print()
print("=" * 60)
print("SLOPE VALIDATION")
print("=" * 60)

slope_valid = output["slope_deg"].dropna()

if len(slope_valid) == 0:
    raise ValueError(
        "Slope contains no valid values."
    )

print(
    "Slope minimum:",
    slope_valid.min()
)

print(
    "Slope maximum:",
    slope_valid.max()
)

print(
    "Slope mean:",
    slope_valid.mean()
)

print(
    "Unique slope values:",
    slope_valid.nunique()
)

if slope_valid.nunique() <= 1:
    raise ValueError(
        "Slope contains only one unique value. "
        "Check the slope raster."
    )


# ------------------------------------------------------------
# 11. Basic range validation
# ------------------------------------------------------------

print()
print("=" * 60)
print("RANGE VALIDATION")
print("=" * 60)

if (output["slope_deg"].dropna() < 0).any():
    raise ValueError(
        "Negative slope values detected."
    )

if (output["mean_ndvi"].dropna() < -1).any():
    raise ValueError(
        "NDVI values below -1 detected."
    )

if (output["mean_ndvi"].dropna() > 1).any():
    raise ValueError(
        "NDVI values above 1 detected."
    )

if (output["annual_rainfall_mm"].dropna() < 0).any():
    raise ValueError(
        "Negative rainfall values detected."
    )


# ------------------------------------------------------------
# 12. Save output
# ------------------------------------------------------------

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. Final confirmation
# ------------------------------------------------------------

print()
print("=" * 60)
print("GEE → 1-KM GRID MAPPING COMPLETED")
print("=" * 60)

print(
    "Rows:",
    len(output)
)

print(
    "Columns:",
    list(output.columns)
)

print(
    "Output:",
    OUTPUT_FILE
)

print()
print("Final missing values:")
print(output.isna().sum())

print()
print("Final summary:")
print(output.describe())