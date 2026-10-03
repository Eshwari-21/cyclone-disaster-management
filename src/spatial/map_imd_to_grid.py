import geopandas as gpd
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

IMD_PATH = Path(
    "data/processed/imd_best_track_2024_clean.csv"
)

GRID_PATH = Path(
    "data/processed/coastal_ap_grid_1km.gpkg"
)

OUTPUT_PATH = Path(
    "data/processed/imd_best_track_2024_grid.csv"
)


# ============================================================
# LOAD IMD DATA
# ============================================================

print("=" * 70)
print("MAPPING IMD TRACKS TO 1 KM GRID")
print("=" * 70)

imd = pd.read_csv(
    IMD_PATH
)

imd["timestamp_utc"] = pd.to_datetime(
    imd["timestamp_utc"]
)

print("\nIMD observations:")
print(len(imd))


# ============================================================
# CONVERT IMD POINTS TO GEODATAFRAME
# ============================================================

imd_gdf = gpd.GeoDataFrame(
    imd,
    geometry=gpd.points_from_xy(
        imd["longitude"],
        imd["latitude"]
    ),
    crs="EPSG:4326"
)


# ============================================================
# LOAD GRID
# ============================================================

grid = gpd.read_file(
    GRID_PATH
)

print(
    "Grid cells:",
    len(grid)
)

print(
    "Grid CRS:",
    grid.crs
)


# ============================================================
# SPATIAL JOIN
# ============================================================

print("\nPerforming spatial join...")

joined = gpd.sjoin(
    imd_gdf,
    grid[
        [
            "grid_id",
            "geometry",
        ]
    ],
    how="left",
    predicate="within"
)


# ============================================================
# CHECK UNMATCHED OBSERVATIONS
# ============================================================

unmatched = joined["grid_id"].isna()

print("\n" + "=" * 70)
print("GRID ASSIGNMENT CHECK")
print("=" * 70)

print(
    "Total IMD observations:",
    len(joined)
)

print(
    "Matched to grid:",
    (~unmatched).sum()
)

print(
    "Unmatched:",
    unmatched.sum()
)


# ============================================================
# SHOW UNMATCHED RECORDS
# ============================================================

if unmatched.sum() > 0:

    print("\nUnmatched observations:")

    print(
        joined.loc[
            unmatched,
            [
                "event_id",
                "event_name",
                "timestamp_utc",
                "latitude",
                "longitude",
            ]
        ].to_string(index=False)
    )


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

output_columns = [
    "event_id",
    "event_name",
    "timestamp_utc",
    "date",
    "time_utc",
    "latitude",
    "longitude",
    "ci_no",
    "ecp_hpa",
    "msw_kt",
    "delta_p_hpa",
    "category",
    "grid_id",
]

result = joined[
    output_columns
].copy()


# ============================================================
# VALIDATE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT VALIDATION")
print("=" * 70)

print(
    "Output rows:",
    len(result)
)

print(
    "Unique grid cells containing tracks:",
    result["grid_id"].nunique()
)

print(
    "Missing grid IDs:",
    result["grid_id"].isna().sum()
)

print(
    "Duplicate event + timestamp:",
    result.duplicated(
        subset=[
            "event_id",
            "timestamp_utc",
        ]
    ).sum()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("IMD → GRID MAPPING COMPLETE")
print("=" * 70)