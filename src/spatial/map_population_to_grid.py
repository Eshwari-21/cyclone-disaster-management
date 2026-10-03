import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

POPULATION_FILE = Path(
    "data/raw/population/ind_ppp_2020_1km_Aggregated_UNadj.tif"
)

GRID_FILE = Path(
    "data/processed/coastal_ap_grid_1km.gpkg"
)

OUTPUT_FILE = Path(
    "data/processed/population_grid_2020.csv"
)


# ---------------------------------------------------------
# Load population raster
# ---------------------------------------------------------

print("Loading population raster...")

with rasterio.open(POPULATION_FILE) as src:

    population_crs = src.crs
    population_nodata = src.nodata

    print(f"Population CRS: {population_crs}")
    print(f"Population resolution: {src.res}")
    print(f"Population NoData: {population_nodata}")


# ---------------------------------------------------------
# Load grid
# ---------------------------------------------------------

print()
print("Loading grid...")

grid = gpd.read_file(GRID_FILE)

print(f"Grid cells: {len(grid)}")
print(f"Grid CRS: {grid.crs}")


# ---------------------------------------------------------
# CRS validation
# ---------------------------------------------------------

if grid.crs != population_crs:

    print()
    print("Reprojecting grid to population raster CRS...")

    grid_for_stats = grid.to_crs(population_crs)

else:

    grid_for_stats = grid.copy()


# ---------------------------------------------------------
# Calculate population statistics
# ---------------------------------------------------------

print()
print("Calculating population for each grid cell...")
print("This may take some time...")


stats = zonal_stats(
    grid_for_stats.geometry,
    str(POPULATION_FILE),
    stats=[
        "sum",
        "mean",
        "min",
        "max",
        "count",
    ],
    nodata=population_nodata,
    all_touched=True,
)


# ---------------------------------------------------------
# Add statistics to grid
# ---------------------------------------------------------

stats_df = pd.DataFrame(stats)

grid_result = pd.DataFrame({
    "grid_id": grid["grid_id"],
})

grid_result["population_sum"] = (
    stats_df["sum"]
)

grid_result["population_mean"] = (
    stats_df["mean"]
)

grid_result["population_min"] = (
    stats_df["min"]
)

grid_result["population_max"] = (
    stats_df["max"]
)

grid_result["population_pixel_count"] = (
    stats_df["count"]
)


# ---------------------------------------------------------
# Clean population values
# ---------------------------------------------------------

grid_result["population_sum"] = (
    grid_result["population_sum"]
    .fillna(0)
)

grid_result["population_mean"] = (
    grid_result["population_mean"]
    .fillna(0)
)

grid_result["population_min"] = (
    grid_result["population_min"]
    .fillna(0)
)

grid_result["population_max"] = (
    grid_result["population_max"]
    .fillna(0)
)

grid_result["population_pixel_count"] = (
    grid_result["population_pixel_count"]
    .fillna(0)
    .astype(int)
)


# ---------------------------------------------------------
# Round population values
# ---------------------------------------------------------

grid_result["population_sum"] = (
    grid_result["population_sum"].round(2)
)

grid_result["population_mean"] = (
    grid_result["population_mean"].round(2)
)

grid_result["population_min"] = (
    grid_result["population_min"].round(2)
)

grid_result["population_max"] = (
    grid_result["population_max"].round(2)
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    f"Grid cells: "
    f"{len(grid_result)}"
)

print(
    f"Unique grid IDs: "
    f"{grid_result['grid_id'].nunique()}"
)

print(
    f"Missing grid IDs: "
    f"{grid_result['grid_id'].isna().sum()}"
)

print(
    f"Missing population values: "
    f"{grid_result['population_sum'].isna().sum()}"
)

print()
print("Population statistics:")

print(
    grid_result["population_sum"].describe()
)

print()
print(
    f"Total estimated population: "
    f"{grid_result['population_sum'].sum():,.2f}"
)

print()
print("Population pixel count:")

print(
    grid_result["population_pixel_count"].describe()
)


# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

grid_result.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved output:")
print(OUTPUT_FILE)

print("=" * 60)
print("POPULATION GRID MAPPING COMPLETE")
print("=" * 60)