import geopandas as gpd
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/coastal_ap_grid_1km.gpkg"
)


print("=" * 70)
print("1 KM GRID VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

grid = gpd.read_file(INPUT_PATH)

print("\nShape:")
print(grid.shape)

print("\nColumns:")
print(grid.columns.tolist())

print("\nCRS:")
print(grid.crs)


# ------------------------------------------------------------
# GRID ID
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GRID ID CHECK")
print("=" * 70)

print(
    "Unique grid IDs:",
    grid["grid_id"].nunique()
)

print(
    "Total rows:",
    len(grid)
)

print(
    "Duplicate grid IDs:",
    grid["grid_id"].duplicated().sum()
)


# ------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GEOMETRY CHECK")
print("=" * 70)

print(
    "Empty geometries:",
    grid.geometry.is_empty.sum()
)

print(
    "Missing geometries:",
    grid.geometry.isna().sum()
)

print(
    "Invalid geometries:",
    (~grid.geometry.is_valid).sum()
)

print(
    "\nGeometry types:"
)

print(
    grid.geometry.geom_type.value_counts().to_string()
)


# ------------------------------------------------------------
# CELL AREA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CELL AREA CHECK")
print("=" * 70)

print(
    "Minimum area (km²):",
    grid["area_km2"].min()
)

print(
    "Maximum area (km²):",
    grid["area_km2"].max()
)

print(
    "Mean area (km²):",
    grid["area_km2"].mean()
)


# ------------------------------------------------------------
# TOTAL STUDY AREA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOTAL GRID AREA")
print("=" * 70)

print(
    "Total area (km²):",
    grid["area_km2"].sum()
)


# ------------------------------------------------------------
# BOUNDS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GRID BOUNDING BOX")
print("=" * 70)

minx, miny, maxx, maxy = grid.total_bounds

print("Minimum longitude:", minx)
print("Minimum latitude: ", miny)
print("Maximum longitude:", maxx)
print("Maximum latitude: ", maxy)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GRID VALIDATION COMPLETE")
print("=" * 70)