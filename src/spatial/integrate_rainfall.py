from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gee_environmental_grid_2024.csv"
)

RAINFALL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_grid_2024.csv"
)

OUTPUT_PATH = ENV_PATH


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("Loading environmental dataset...")
env = pd.read_csv(ENV_PATH)

print("Loading corrected rainfall dataset...")
rainfall = pd.read_csv(RAINFALL_PATH)


# ---------------------------------------------------------
# Validate input datasets
# ---------------------------------------------------------

print("\nInput validation")
print("-" * 50)

print("Environmental rows:", len(env))
print("Rainfall rows:", len(rainfall))

if env["grid_id"].duplicated().any():
    raise ValueError("Environmental dataset contains duplicate grid IDs.")

if rainfall["grid_id"].duplicated().any():
    raise ValueError("Rainfall dataset contains duplicate grid IDs.")

if env["grid_id"].isna().any():
    raise ValueError("Environmental dataset contains missing grid IDs.")

if rainfall["grid_id"].isna().any():
    raise ValueError("Rainfall dataset contains missing grid IDs.")


# ---------------------------------------------------------
# Remove old rainfall column
# ---------------------------------------------------------

if "annual_rainfall_mm" not in env.columns:
    raise ValueError(
        "Expected column 'annual_rainfall_mm' "
        "was not found in environmental dataset."
    )

env = env.drop(columns=["annual_rainfall_mm"])


# ---------------------------------------------------------
# Merge corrected rainfall
# ---------------------------------------------------------

print("\nMerging corrected rainfall...")

result = env.merge(
    rainfall,
    on="grid_id",
    how="left",
    validate="one_to_one"
)

result = result.rename(
    columns={"rainfall_mm": "annual_rainfall_mm"}
)


# ---------------------------------------------------------
# Validate final dataset
# ---------------------------------------------------------

print("\nFinal validation")
print("-" * 50)

print("Rows:", len(result))
print("Columns:", result.columns.tolist())

print(
    "Unique grid IDs:",
    result["grid_id"].nunique()
)

print(
    "Duplicate grid IDs:",
    result["grid_id"].duplicated().sum()
)

print(
    "Missing rainfall:",
    result["annual_rainfall_mm"].isna().sum()
)

print(
    "Rainfall minimum:",
    result["annual_rainfall_mm"].min()
)

print(
    "Rainfall maximum:",
    result["annual_rainfall_mm"].max()
)

print(
    "Rainfall mean:",
    result["annual_rainfall_mm"].mean()
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nUpdated:")
print(OUTPUT_PATH)