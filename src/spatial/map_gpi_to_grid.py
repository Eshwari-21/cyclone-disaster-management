from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GPI_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "mosdac"
    / "mosdac_gpi_points.csv"
)

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coastal_ap_grid_1km.gpkg"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mosdac_gpi_grid_2024.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# GPI native grid resolution
# ---------------------------------------------------------
GPI_LAT_INTERVAL = 1.0
GPI_LON_INTERVAL = 1.0


# ---------------------------------------------------------
# Read GPI observations
# ---------------------------------------------------------
gpi = pd.read_csv(GPI_FILE)

required_gpi_columns = {
    "timestamp_utc",
    "latitude",
    "longitude",
    "gpi_mm",
}

missing_columns = required_gpi_columns - set(gpi.columns)

if missing_columns:
    raise ValueError(
        f"Missing GPI columns: {sorted(missing_columns)}"
    )

print(f"GPI observations: {len(gpi):,}")


# ---------------------------------------------------------
# Read project 1-km grid
# ---------------------------------------------------------
grid = gpd.read_file(GRID_FILE)

if grid.crs is None:
    raise ValueError("Grid has no CRS.")

grid = grid.to_crs("EPSG:4326")

required_grid_columns = {"grid_id", "geometry"}

missing_grid_columns = required_grid_columns - set(grid.columns)

if missing_grid_columns:
    raise ValueError(
        f"Missing grid columns: {sorted(missing_grid_columns)}"
    )

print(f"Project grid cells: {len(grid):,}")


# ---------------------------------------------------------
# Create centroid points for project grid
#
# We use centroids rather than assigning the GPI value to
# every cell that merely touches the GPI boundary.
# --------------------------------------------------------
# ---------------------------------------------------------
# Create grid centroids in projected CRS
# ---------------------------------------------------------
grid_centroids = grid[["grid_id", "geometry"]].copy()

# Reproject to UTM Zone 44N so centroid calculation
# is performed in a projected coordinate system.
grid_centroids = grid_centroids.to_crs("EPSG:32644")

grid_centroids["geometry"] = (
    grid_centroids.geometry.centroid
)

# Convert centroid coordinates back to geographic CRS
# because the GPI cells are defined in latitude/longitude.
grid_centroids = grid_centroids.to_crs("EPSG:4326")

# ---------------------------------------------------------
# Create GPI cell polygons
#
# GPI coordinates are treated as cell centers.
#
# Example:
# latitude  = 17.5
# longitude = 82.5
#
# becomes:
# latitude  17.0 -> 18.0
# longitude 82.0 -> 83.0
# ---------------------------------------------------------
gpi_cells = []

for row in gpi.itertuples(index=False):

    lat_center = float(row.latitude)
    lon_center = float(row.longitude)

    min_lat = lat_center - GPI_LAT_INTERVAL / 2
    max_lat = lat_center + GPI_LAT_INTERVAL / 2

    min_lon = lon_center - GPI_LON_INTERVAL / 2
    max_lon = lon_center + GPI_LON_INTERVAL / 2

    gpi_cells.append(
        {
            "timestamp_utc": row.timestamp_utc,
            "gpi_latitude": lat_center,
            "gpi_longitude": lon_center,
            "gpi_mm": row.gpi_mm,
            "geometry": box(
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            ),
        }
    )


gpi_cells = gpd.GeoDataFrame(
    gpi_cells,
    geometry="geometry",
    crs="EPSG:4326",
)


print(f"GPI cells created: {len(gpi_cells):,}")


# ---------------------------------------------------------
# Spatially associate 1-km grid centroids with GPI cells
# ---------------------------------------------------------
mapped = gpd.sjoin(
    grid_centroids,
    gpi_cells[
        [
            "timestamp_utc",
            "gpi_latitude",
            "gpi_longitude",
            "gpi_mm",
            "geometry",
        ]
    ],
    how="inner",
    predicate="within",
)


# ---------------------------------------------------------
# Keep only required output columns
# ---------------------------------------------------------
result = mapped[
    [
        "grid_id",
        "timestamp_utc",
        "gpi_latitude",
        "gpi_longitude",
        "gpi_mm",
    ]
].copy()


# ---------------------------------------------------------
# Remove accidental duplicates
# ---------------------------------------------------------
result = result.drop_duplicates(
    subset=[
        "grid_id",
        "timestamp_utc",
    ]
)


# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------
result = result.sort_values(
    [
        "timestamp_utc",
        "grid_id",
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
print()
print("Mapping validation")
print("------------------")

print(f"Mapped rows: {len(result):,}")
print(
    f"Unique grid cells mapped: "
    f"{result['grid_id'].nunique():,}"
)
print(
    f"Unique timestamps: "
    f"{result['timestamp_utc'].nunique():,}"
)

duplicate_count = result.duplicated(
    ["grid_id", "timestamp_utc"]
).sum()

print(
    f"Duplicate grid/timestamp pairs: "
    f"{duplicate_count}"
)

print(
    f"Missing GPI values: "
    f"{result['gpi_mm'].isna().sum()}"
)

if len(result) > 0:
    print(
        f"GPI range: "
        f"{result['gpi_mm'].min():.3f} - "
        f"{result['gpi_mm'].max():.3f} mm"
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
result.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("GPI spatial mapping complete.")
print(f"Output: {OUTPUT_FILE}")