from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_grid_base_2024.csv"
)

IMD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imd_grid_features_2024.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_grid_2024.csv"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

print("Loading master base grid...")
master = pd.read_csv(MASTER_PATH)

print("Loading IMD grid features...")
imd = pd.read_csv(IMD_PATH)


# ---------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------

print("\nInput validation")
print("-" * 50)

print("Master rows:", len(master))
print("IMD rows:", len(imd))

if master["grid_id"].duplicated().any():
    raise ValueError(
        "Master dataset contains duplicate grid IDs."
    )

if imd["grid_id"].duplicated().any():
    raise ValueError(
        "IMD dataset contains duplicate grid IDs."
    )

if set(master["grid_id"]) != set(imd["grid_id"]):
    raise ValueError(
        "Master and IMD grid IDs do not match."
    )


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

print("\nMerging master + IMD features...")

result = master.merge(
    imd,
    on="grid_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\nFinal validation")
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