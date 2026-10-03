import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/imd_best_track_2024_clean.csv")

df = pd.read_csv(INPUT_PATH)

print("========== IMD DATA VALIDATION ==========")

print("\n1. Shape")
print(df.shape)

print("\n2. Events")
print(df[["event_id", "event_name"]].drop_duplicates().to_string(index=False))

print("\n3. Missing event IDs")
print(df["event_id"].isna().sum())

print("\n4. Duplicate event + timestamp")
duplicate_rows = df.duplicated(
    subset=["event_id", "timestamp_utc"]
).sum()

print(duplicate_rows)

print("\n5. Rows per event")
print(
    df.groupby(["event_id", "event_name"])
      .size()
      .to_string()
)

print("\n6. Date range")
print("Start:", df["timestamp_utc"].min())
print("End:", df["timestamp_utc"].max())

print("\n7. Missing values")
print(df.isnull().sum())

print("\n8. Categories")
print(df["category"].value_counts())

print("\n9. Latitude range")
print(df["latitude"].min(), "to", df["latitude"].max())

print("\n10. Longitude range")
print(df["longitude"].min(), "to", df["longitude"].max())

print("\n==========================================")