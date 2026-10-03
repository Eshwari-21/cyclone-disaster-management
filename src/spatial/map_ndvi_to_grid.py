from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coastal_ap_grid_1km.gpkg"
)

NDVI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gee"
    / "coastal_ap_ndvi_2024.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ndvi_grid_2024.csv"
)


# ---------------------------------------------------------
# Load grid
# ---------------------------------------------------------

print("Loading project grid...")

grid = gpd.read_file(GRID_PATH)

print(f"Grid cells: {len(grid)}")
print(f"Grid CRS: {grid.crs}")


# ---------------------------------------------------------
# Load NDVI raster
# ---------------------------------------------------------

print("\nLoading NDVI raster...")

with rasterio.open(NDVI_PATH) as src:

    ndvi = src.read(1).astype(float)

    transform = src.transform
    raster_crs = src.crs
    nodata = src.nodata

    print(f"Raster shape: {ndvi.shape}")
    print(f"Raster CRS: {raster_crs}")
    print(f"Raster resolution: {src.res}")
    print(f"Raster bounds: {src.bounds}")
    print(f"Raster nodata: {nodata}")


# ---------------------------------------------------------
# Calculate mean NDVI for each 1-km grid cell
# ---------------------------------------------------------

print("\nCalculating NDVI for each grid cell...")

stats = zonal_stats(
    grid,
    ndvi,
    affine=transform,
    stats=["mean"],
    nodata=np.nan,
    all_touched=True,
)

mean_ndvi = np.array(
    [item["mean"] for item in stats],
    dtype=float
)


# ---------------------------------------------------------
# Create output
# ---------------------------------------------------------

result = pd.DataFrame(
    {
        "grid_id": grid["grid_id"],
        "mean_ndvi": mean_ndvi,
    }
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\nValidation")
print("-" * 50)

print(f"Total grid cells: {len(result)}")

print(
    f"Valid NDVI cells: "
    f"{result['mean_ndvi'].notna().sum()}"
)

print(
    f"Missing NDVI cells: "
    f"{result['mean_ndvi'].isna().sum()}"
)

print(
    f"Unique grid IDs: "
    f"{result['grid_id'].nunique()}"
)

print(
    f"Duplicate grid IDs: "
    f"{result['grid_id'].duplicated().sum()}"
)

print(
    f"Minimum NDVI: "
    f"{result['mean_ndvi'].min()}"
)

print(
    f"Maximum NDVI: "
    f"{result['mean_ndvi'].max()}"
)

print(
    f"Mean NDVI: "
    f"{result['mean_ndvi'].mean()}"
)


# ---------------------------------------------------------
# Range validation
# ---------------------------------------------------------

valid_ndvi = result["mean_ndvi"].dropna()

if ((valid_ndvi < -1) | (valid_ndvi > 1)).any():
    raise ValueError(
        "NDVI contains values outside the valid range [-1, 1]."
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)