from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imd_grid_event_features_2024.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imd_grid_features_2024.csv"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

print("Loading IMD grid-event data...")

df = pd.read_csv(INPUT_PATH)

print("Rows:", len(df))
print("Grid cells:", df["grid_id"].nunique())
print("Events:", df["event_id"].nunique())


# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------

if df["grid_id"].isna().any():
    raise ValueError("Missing grid IDs found.")

if df["event_id"].isna().any():
    raise ValueError("Missing event IDs found.")

if df.duplicated(["grid_id", "event_id"]).any():
    raise ValueError("Duplicate grid-event pairs found.")


# ---------------------------------------------------------
# Pivot event distances
# ---------------------------------------------------------

distance_table = df.pivot(
    index="grid_id",
    columns="event_id",
    values="distance_km"
)

distance_table.columns = [
    f"distance_event_{int(c)}_km"
    for c in distance_table.columns
]

distance_table = distance_table.reset_index()


# ---------------------------------------------------------
# Minimum distance to any historical event
# ---------------------------------------------------------

distance_columns = [
    c for c in distance_table.columns
    if c.startswith("distance_event_")
]

distance_table["min_event_distance_km"] = (
    distance_table[distance_columns].min(axis=1)
)


# ---------------------------------------------------------
# Event with maximum recorded wind speed
# ---------------------------------------------------------

max_msw = (
    df.groupby("event_id")["msw_kt"]
    .max()
    .rename("event_max_msw_kt")
    .reset_index()
)

df2 = df.merge(
    max_msw,
    on="event_id",
    how="left",
    validate="many_to_one",
)


# ---------------------------------------------------------
# For each grid cell, find the strongest historical event
# ---------------------------------------------------------

df2 = df2.sort_values(
    ["grid_id", "event_max_msw_kt", "distance_km"],
    ascending=[True, False, True]
)

strongest = (
    df2.drop_duplicates("grid_id")
    [
        [
            "grid_id",
            "event_id",
            "event_name",
            "event_max_msw_kt",
            "distance_km",
        ]
    ]
    .rename(
        columns={
            "event_id": "strongest_event_id",
            "event_name": "strongest_event_name",
            "event_max_msw_kt": "max_event_msw_kt",
            "distance_km": "distance_to_strongest_event_km",
        }
    )
)


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

result = distance_table.merge(
    strongest,
    on="grid_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Optional threshold counts
# ---------------------------------------------------------
# These are descriptive historical features, not
# scientifically established risk thresholds.
# ---------------------------------------------------------

result["events_within_50km"] = (
    result[distance_columns].le(50).sum(axis=1)
)

result["events_within_100km"] = (
    result[distance_columns].le(100).sum(axis=1)
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\nValidation")
print("-" * 50)

print("Rows:", len(result))

print(
    "Unique grid IDs:",
    result["grid_id"].nunique()
)

print(
    "Duplicate grid IDs:",
    result["grid_id"].duplicated().sum()
)

print("\nColumns:")
print(result.columns.tolist())

print("\nMissing values:")
print(result.isna().sum())

print("\nMinimum historical distance:")
print(result["min_event_distance_km"].min())

print("\nMaximum historical distance:")
print(result["min_event_distance_km"].max())

print("\nMaximum event MSW:")
print(result["max_event_msw_kt"].max())

print("\nEvents within 50 km:")
print(result["events_within_50km"].describe())

print("\nEvents within 100 km:")
print(result["events_within_100km"].describe())


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