from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID_PATH = PROJECT_ROOT / "data" / "processed" / "coastal_ap_grid_1km.gpkg"

RAINFALL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gee"
    / "coastal_ap_rainfall_2024.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_grid_2024.csv"
)


# ---------------------------------------------------------
# Load 1-km project grid
# ---------------------------------------------------------

print("Loading project grid...")

grid = gpd.read_file(GRID_PATH)

print(f"Grid cells: {len(grid)}")
print(f"Grid CRS: {grid.crs}")


# ---------------------------------------------------------
# Read rainfall raster
# ---------------------------------------------------------

print("\nLoading rainfall raster...")

with rasterio.open(RAINFALL_PATH) as src:

    rainfall = src.read(1).astype(float)

    transform = src.transform
    raster_crs = src.crs
    nodata = src.nodata

    print(f"Raster shape: {rainfall.shape}")
    print(f"Raster CRS: {raster_crs}")
    print(f"Raster resolution: {src.res}")
    print(f"Raster bounds: {src.bounds}")
    print(f"Raster nodata: {nodata}")


# ---------------------------------------------------------
# Identify valid rainfall pixels
# ---------------------------------------------------------

valid_mask = np.isfinite(rainfall)

if nodata is not None:
    valid_mask &= rainfall != nodata

rows, cols = np.where(valid_mask)

print(f"\nValid rainfall pixels: {len(rows)}")

if len(rows) == 0:
    raise ValueError("No valid rainfall pixels found.")


# ---------------------------------------------------------
# Convert rainfall pixels into polygons
# ---------------------------------------------------------

print("\nCreating rainfall pixel footprints...")

pixel_shapes = shapes(
    rainfall,
    mask=valid_mask,
    transform=transform
)

rainfall_polygons = []

for geom, value in pixel_shapes:

    rainfall_polygons.append(
        {
            "rainfall_mm": float(value),
            "geometry": shape(geom),
        }
    )


rainfall_gdf = gpd.GeoDataFrame(
    rainfall_polygons,
    crs=raster_crs
)

print(f"Rainfall pixel polygons: {len(rainfall_gdf)}")


# ---------------------------------------------------------
# Reproject to UTM Zone 44N
# ---------------------------------------------------------
# The project grid was originally created in metres using
# EPSG:32644. We use the same CRS for the spatial operation.
# ---------------------------------------------------------

TARGET_CRS = "EPSG:32644"

print(f"\nReprojecting to {TARGET_CRS}...")

grid_utm = grid.to_crs(TARGET_CRS)
rainfall_utm = rainfall_gdf.to_crs(TARGET_CRS)


# ---------------------------------------------------------
# Spatial intersection
# ---------------------------------------------------------

print("\nMapping rainfall pixels to 1-km grid...")

grid_result = gpd.overlay(
    grid_utm[["grid_id", "geometry"]],
    rainfall_utm[["rainfall_mm", "geometry"]],
    how="intersection",
    keep_geom_type=False,
)

print(f"Intersection records: {len(grid_result)}")


# ---------------------------------------------------------
# Calculate intersection area
# ---------------------------------------------------------

grid_result["intersection_area_m2"] = grid_result.geometry.area


# ---------------------------------------------------------
# Keep the rainfall pixel contributing the largest area
# ---------------------------------------------------------
#
# A 1-km grid cell can theoretically intersect more than one
# rainfall pixel. We assign the rainfall value from the pixel
# covering the largest portion of that grid cell.
# ---------------------------------------------------------

grid_result = grid_result.sort_values(
    ["grid_id", "intersection_area_m2"],
    ascending=[True, False]
)

grid_result = grid_result.drop_duplicates(
    subset=["grid_id"],
    keep="first"
)


# ---------------------------------------------------------
# Restore all project grid cells
# ---------------------------------------------------------

result = grid[["grid_id"]].copy()

result = result.merge(
    grid_result[["grid_id", "rainfall_mm"]],
    on="grid_id",
    how="left"
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\nValidation")
print("-" * 50)

print(f"Total grid cells: {len(result)}")

print(
    f"Grid cells with rainfall: "
    f"{result['rainfall_mm'].notna().sum()}"
)

print(
    f"Grid cells without rainfall: "
    f"{result['rainfall_mm'].isna().sum()}"
)

print(
    f"Minimum rainfall: "
    f"{result['rainfall_mm'].min()}"
)

print(
    f"Maximum rainfall: "
    f"{result['rainfall_mm'].max()}"
)

print(
    f"Mean rainfall: "
    f"{result['rainfall_mm'].mean()}"
)

print(
    f"Unique grid IDs: "
    f"{result['grid_id'].nunique()}"
)

print(
    f"Duplicate grid IDs: "
    f"{result['grid_id'].duplicated().sum()}"
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