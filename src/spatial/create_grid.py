import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import box


# ============================================================
# PATHS
# ============================================================

BOUNDARY_PATH = Path(
    "data/raw/boundaries/coastal_ap_districts.geojson.gpkg"
)

OUTPUT_PATH = Path(
    "data/processed/coastal_ap_grid_1km.gpkg"
)


# ============================================================
# SETTINGS
# ============================================================

# UTM Zone 44N is appropriate for the study area.
PROJECTED_CRS = "EPSG:32644"

# Grid cell size in metres.
GRID_SIZE = 1000


# ============================================================
# LOAD DISTRICT BOUNDARY
# ============================================================

print("=" * 70)
print("CREATING 1 KM GRID")
print("=" * 70)

print("\nLoading boundary:")
print(BOUNDARY_PATH)

districts = gpd.read_file(
    BOUNDARY_PATH
)

print(
    "Districts loaded:",
    len(districts)
)

print(
    "Original CRS:",
    districts.crs
)


# ============================================================
# PROJECT TO METRIC CRS
# ============================================================

districts_projected = districts.to_crs(
    PROJECTED_CRS
)

print(
    "Projected CRS:",
    districts_projected.crs
)


# ============================================================
# STUDY AREA
# ============================================================

study_area = districts_projected.geometry.union_all()

print("\nStudy area created.")


# ============================================================
# CREATE GRID EXTENT
# ============================================================

minx, miny, maxx, maxy = study_area.bounds

print("\nProjected study-area bounds:")
print("minx:", minx)
print("miny:", miny)
print("maxx:", maxx)
print("maxy:", maxy)


# ============================================================
# GENERATE GRID CELLS
# ============================================================

print("\nGenerating 1 km × 1 km cells...")

grid_cells = []

grid_id = 1

x = minx

while x < maxx:

    y = miny

    while y < maxy:

        cell = box(
            x,
            y,
            x + GRID_SIZE,
            y + GRID_SIZE
        )

        # Keep cells that intersect the study area.
        if cell.intersects(study_area):

            grid_cells.append(
                {
                    "grid_id": f"grid_{grid_id:05d}",
                    "geometry": cell,
                }
            )

            grid_id += 1

        y += GRID_SIZE

    x += GRID_SIZE


# ============================================================
# CREATE GEODATAFRAME
# ============================================================

grid = gpd.GeoDataFrame(
    grid_cells,
    crs=PROJECTED_CRS
)

print(
    "\nGenerated cells:",
    len(grid)
)


# ============================================================
# CLIP GRID TO STUDY AREA
# ============================================================

print("\nClipping grid to study area...")

grid = gpd.clip(
    grid,
    districts_projected
)

# Re-create sequential grid IDs after clipping.
grid = grid.reset_index(drop=True)

grid["grid_id"] = [
    f"grid_{i:05d}"
    for i in range(1, len(grid) + 1)
]


# ============================================================
# CALCULATE CELL AREA
# ============================================================

grid["area_m2"] = grid.geometry.area

grid["area_km2"] = (
    grid["area_m2"] / 1_000_000
)


# ============================================================
# KEEP ONLY REQUIRED COLUMNS
# ============================================================

grid = grid[
    [
        "grid_id",
        "area_m2",
        "area_km2",
        "geometry",
    ]
]


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("GRID VALIDATION")
print("=" * 70)

print(
    "Grid cells:",
    len(grid)
)

print(
    "CRS:",
    grid.crs
)

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
    "Minimum cell area:",
    grid["area_m2"].min(),
    "m²"
)

print(
    "Maximum cell area:",
    grid["area_m2"].max(),
    "m²"
)

print(
    "Total grid area:",
    grid["area_km2"].sum(),
    "km²"
)


# ============================================================
# REPROJECT TO WGS84 FOR STORAGE
# ============================================================

grid_wgs84 = grid.to_crs(
    "EPSG:4326"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

grid_wgs84.to_file(
    OUTPUT_PATH,
    layer="coastal_ap_grid_1km",
    driver="GPKG"
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("GRID CREATION COMPLETE")
print("=" * 70)