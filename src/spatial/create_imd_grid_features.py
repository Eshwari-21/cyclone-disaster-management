import geopandas as gpd
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

IMD_FILE = Path(
    "data/processed/imd_best_track_2024_clean.csv"
)

GRID_FILE = Path(
    "data/processed/coastal_ap_grid_1km.gpkg"
)

OUTPUT_FILE = Path(
    "data/processed/imd_grid_event_features_2024.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading IMD data...")
imd = pd.read_csv(IMD_FILE)

print(f"IMD observations: {len(imd)}")


print("Loading grid...")
grid = gpd.read_file(GRID_FILE)

print(f"Grid cells: {len(grid)}")
print(f"Grid CRS: {grid.crs}")


# ---------------------------------------------------------
# Validate columns
# ---------------------------------------------------------

required_columns = [
    "event_id",
    "event_name",
    "timestamp_utc",
    "latitude",
    "longitude",
    "ci_no",
    "ecp_hpa",
    "msw_kt",
    "delta_p_hpa",
    "category",
]

missing_columns = [
    col for col in required_columns
    if col not in imd.columns
]

if missing_columns:
    raise ValueError(
        f"Missing IMD columns: {missing_columns}"
    )

if "grid_id" not in grid.columns:
    raise ValueError(
        "Grid does not contain grid_id"
    )


# ---------------------------------------------------------
# Prepare IMD points
# ---------------------------------------------------------

imd["timestamp_utc"] = pd.to_datetime(
    imd["timestamp_utc"]
)

imd_points = gpd.GeoDataFrame(
    imd.copy(),
    geometry=gpd.points_from_xy(
        imd["longitude"],
        imd["latitude"]
    ),
    crs="EPSG:4326"
)


# ---------------------------------------------------------
# Reproject to UTM Zone 44N
# ---------------------------------------------------------

print("Reprojecting to UTM Zone 44N...")

imd_utm = imd_points.to_crs("EPSG:32644")
grid_utm = grid.to_crs("EPSG:32644")


# ---------------------------------------------------------
# Create grid centroids
# ---------------------------------------------------------

print("Creating grid centroids...")

grid_centroids = grid_utm[
    ["grid_id", "geometry"]
].copy()

grid_centroids["geometry"] = (
    grid_centroids.geometry.centroid
)


# ---------------------------------------------------------
# Calculate nearest track point per event
# ---------------------------------------------------------

all_features = []

event_ids = sorted(
    imd_utm["event_id"].unique()
)

print()
print(f"Number of cyclone events: {len(event_ids)}")
print()


for event_id in event_ids:

    print(
        f"Processing event {event_id}..."
    )

    # -----------------------------------------------------
    # Select event track
    # -----------------------------------------------------

    event_track = imd_utm[
        imd_utm["event_id"] == event_id
    ].copy()

    if event_track.empty:
        print(
            f"Skipping event {event_id}: no observations"
        )
        continue


    # -----------------------------------------------------
    # Nearest spatial join
    # -----------------------------------------------------

    event_result = gpd.sjoin_nearest(
        grid_centroids,
        event_track[
            [
                "event_id",
                "event_name",
                "timestamp_utc",
                "latitude",
                "longitude",
                "ci_no",
                "ecp_hpa",
                "msw_kt",
                "delta_p_hpa",
                "category",
                "geometry",
            ]
        ],
        how="left",
        distance_col="distance_m",
    )


    # -----------------------------------------------------
    # Convert distance to kilometres
    # -----------------------------------------------------

    event_result["distance_km"] = (
        event_result["distance_m"] / 1000.0
    )


    # -----------------------------------------------------
    # Sort for deterministic tie-breaking
    # -----------------------------------------------------
    #
    # If multiple track observations are exactly the same
    # distance from a grid centroid, keep the earliest
    # observation.
    #

    event_result = event_result.sort_values(
        [
            "grid_id",
            "distance_m",
            "timestamp_utc",
        ]
    )


    # -----------------------------------------------------
    # Remove duplicate grid-event pairs
    # -----------------------------------------------------

    event_result = event_result.drop_duplicates(
        subset=[
            "grid_id",
            "event_id",
        ],
        keep="first",
    )


    # -----------------------------------------------------
    # Select output columns
    # -----------------------------------------------------

    event_features = event_result[
        [
            "grid_id",
            "event_id",
            "event_name",
            "timestamp_utc",
            "latitude",
            "longitude",
            "ci_no",
            "ecp_hpa",
            "msw_kt",
            "delta_p_hpa",
            "category",
            "distance_km",
        ]
    ].copy()


    all_features.append(
        event_features
    )


# ---------------------------------------------------------
# Combine all events
# ---------------------------------------------------------

print()
print("Combining event results...")

features = pd.concat(
    all_features,
    ignore_index=True
)


# ---------------------------------------------------------
# Sort final dataset
# ---------------------------------------------------------

features = features.sort_values(
    [
        "grid_id",
        "event_id",
        "distance_km",
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    f"Output rows: {len(features)}"
)

print(
    f"Unique grid cells: "
    f"{features['grid_id'].nunique()}"
)

print(
    f"Unique events: "
    f"{features['event_id'].nunique()}"
)

print(
    f"Missing grid IDs: "
    f"{features['grid_id'].isna().sum()}"
)

print(
    f"Missing event IDs: "
    f"{features['event_id'].isna().sum()}"
)


# ---------------------------------------------------------
# Rows per event
# ---------------------------------------------------------

print()
print("Rows per event:")

print(
    features[
        "event_id"
    ].value_counts().sort_index()
)


# ---------------------------------------------------------
# Distance statistics
# ---------------------------------------------------------

print()
print("Distance statistics:")

print(
    features["distance_km"].describe()
)


# ---------------------------------------------------------
# Expected row count
# ---------------------------------------------------------

print()
print("Checking grid-event combinations...")

expected_rows = (
    grid["grid_id"].nunique()
    * len(event_ids)
)

actual_rows = len(features)

print(
    f"Expected rows: {expected_rows}"
)

print(
    f"Actual rows:   {actual_rows}"
)


# ---------------------------------------------------------
# Duplicate check
# ---------------------------------------------------------

duplicates = features.duplicated(
    subset=[
        "grid_id",
        "event_id",
    ]
).sum()

print(
    f"Duplicate grid-event pairs: {duplicates}"
)


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print()
print("Saved output:")
print(OUTPUT_FILE)

print("=" * 60)
print("IMD GRID-EVENT FEATURE CREATION COMPLETE")
print("=" * 60)