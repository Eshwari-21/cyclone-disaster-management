from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gee_environmental_grid_2024.csv"
)

NDVI_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ndvi_grid_2024.csv"
)


print("Loading environmental dataset...")
env = pd.read_csv(ENV_PATH)

print("Loading corrected NDVI dataset...")
ndvi = pd.read_csv(NDVI_PATH)


print("\nInput validation")
print("-" * 50)

print("Environmental rows:", len(env))
print("NDVI rows:", len(ndvi))

if env["grid_id"].duplicated().any():
    raise ValueError("Environmental dataset contains duplicate grid IDs.")

if ndvi["grid_id"].duplicated().any():
    raise ValueError("NDVI dataset contains duplicate grid IDs.")


# Remove old NDVI column
if "mean_ndvi" not in env.columns:
    raise ValueError(
        "Expected column 'mean_ndvi' was not found."
    )

env = env.drop(columns=["mean_ndvi"])


# Merge corrected NDVI
result = env.merge(
    ndvi,
    on="grid_id",
    how="left",
    validate="one_to_one"
)


print("\nFinal validation")
print("-" * 50)

print("Rows:", len(result))
print("Columns:", result.columns.tolist())

print("Unique grid IDs:", result["grid_id"].nunique())

print(
    "Duplicate grid IDs:",
    result["grid_id"].duplicated().sum()
)

print(
    "Missing NDVI:",
    result["mean_ndvi"].isna().sum()
)

print(
    "NDVI minimum:",
    result["mean_ndvi"].min()
)

print(
    "NDVI maximum:",
    result["mean_ndvi"].max()
)

print(
    "NDVI mean:",
    result["mean_ndvi"].mean()
)


# Save updated environmental dataset
result.to_csv(
    ENV_PATH,
    index=False
)

print("\nUpdated:")
print(ENV_PATH)