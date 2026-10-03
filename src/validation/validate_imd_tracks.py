import pandas as pd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

INPUT_PATH = Path(
    "data/processed/imd_best_track_2024_clean.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"]
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("IMD BEST TRACK VALIDATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 1. TIMESTAMP RANGE
# ============================================================

print("\n" + "=" * 70)
print("1. OVERALL TIMESTAMP RANGE")
print("=" * 70)

print("Start:", df["timestamp_utc"].min())
print("End:  ", df["timestamp_utc"].max())


# ============================================================
# 2. EVENT TIMESTAMP RANGES
# ============================================================

print("\n" + "=" * 70)
print("2. TIMESTAMP RANGE PER EVENT")
print("=" * 70)

event_summary = (
    df.groupby(
        ["event_id", "event_name"]
    )
    .agg(
        start_time=("timestamp_utc", "min"),
        end_time=("timestamp_utc", "max"),
        observations=("timestamp_utc", "count"),
    )
    .reset_index()
)

print(
    event_summary.to_string(index=False)
)


# ============================================================
# 3. COORDINATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("3. COORDINATE RANGE")
print("=" * 70)

print(
    "Latitude:",
    df["latitude"].min(),
    "to",
    df["latitude"].max()
)

print(
    "Longitude:",
    df["longitude"].min(),
    "to",
    df["longitude"].max()
)


# ============================================================
# 4. INVALID COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("4. INVALID COORDINATES")
print("=" * 70)

invalid_lat = df[
    (df["latitude"] < -90)
    | (df["latitude"] > 90)
]

invalid_lon = df[
    (df["longitude"] < -180)
    | (df["longitude"] > 180)
]

print(
    "Invalid latitude rows:",
    len(invalid_lat)
)

print(
    "Invalid longitude rows:",
    len(invalid_lon)
)


# ============================================================
# 5. DUPLICATE TIMESTAMPS WITHIN EACH EVENT
# ============================================================

print("\n" + "=" * 70)
print("5. DUPLICATE EVENT + TIMESTAMP")
print("=" * 70)

duplicates = df.duplicated(
    subset=[
        "event_id",
        "timestamp_utc",
    ]
)

print(
    "Duplicates:",
    duplicates.sum()
)


# ============================================================
# 6. TIMESTAMP ORDER
# ============================================================

print("\n" + "=" * 70)
print("6. TIMESTAMP ORDER CHECK")
print("=" * 70)

df_sorted = df.sort_values(
    [
        "event_id",
        "timestamp_utc",
    ]
).copy()

df_sorted["previous_timestamp"] = (
    df_sorted
    .groupby("event_id")["timestamp_utc"]
    .shift(1)
)

df_sorted["time_difference_hours"] = (
    df_sorted["timestamp_utc"]
    - df_sorted["previous_timestamp"]
).dt.total_seconds() / 3600


# Negative differences indicate ordering problems.
negative_gaps = df_sorted[
    df_sorted["time_difference_hours"] < 0
]

print(
    "Negative time gaps:",
    len(negative_gaps)
)


# ============================================================
# 7. LARGE TIME GAPS
# ============================================================

print("\n" + "=" * 70)
print("7. LARGE TIME GAPS (> 24 HOURS)")
print("=" * 70)

large_gaps = df_sorted[
    df_sorted["time_difference_hours"] > 24
]

if len(large_gaps) == 0:

    print("No gaps greater than 24 hours.")

else:

    print(
        large_gaps[
            [
                "event_id",
                "event_name",
                "previous_timestamp",
                "timestamp_utc",
                "time_difference_hours",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 8. CATEGORY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("8. CATEGORY COUNTS")
print("=" * 70)

print(
    df["category"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 9. WIND RANGE
# ============================================================

print("\n" + "=" * 70)
print("9. MSW RANGE")
print("=" * 70)

print(
    "Minimum MSW:",
    df["msw_kt"].min(),
    "kt"
)

print(
    "Maximum MSW:",
    df["msw_kt"].max(),
    "kt"
)


# ============================================================
# 10. PRESSURE RANGE
# ============================================================

print("\n" + "=" * 70)
print("10. PRESSURE RANGE")
print("=" * 70)

print(
    "Minimum ECP:",
    df["ecp_hpa"].min(),
    "hPa"
)

print(
    "Maximum ECP:",
    df["ecp_hpa"].max(),
    "hPa"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)