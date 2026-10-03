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

POP_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "population_grid_2020.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_grid_base_2024.csv"
)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("Loading environmental dataset...")
env = pd.read_csv(ENV_PATH)

print("Loading population dataset...")
pop = pd.read_csv(POP_PATH)


# ---------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------

print("\nInput validation")
print("-" * 50)

print("Environmental rows:", len(env))
print("Population rows:", len(pop))

if env["grid_id"].duplicated().any():
    raise ValueError(
        "Environmental dataset contains duplicate grid IDs."
    )

if pop["grid_id"].duplicated().any():
    raise ValueError(
        "Population dataset contains duplicate grid IDs."
    )

if set(env["grid_id"]) != set(pop["grid_id"]):
    raise ValueError(
        "Environmental and population grid IDs do not match."
    )


# ---------------------------------------------------------
# Select population feature
# ---------------------------------------------------------
#
# population_sum represents the estimated population
# associated with each 1-km grid cell.
# ---------------------------------------------------------

pop = pop[
    [
        "grid_id",
        "population_sum",
    ]
]


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

print("\nMerging environmental + population data...")

master = env.merge(
    pop,
    on="grid_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("\nFinal validation")
print("-" * 50)

print("Rows:", len(master))

print(
    "Unique grid IDs:",
    master["grid_id"].nunique()
)

print(
    "Duplicate grid IDs:",
    master["grid_id"].duplicated().sum()
)

print("\nColumns:")
print(master.columns.tolist())

print("\nMissing values:")
print(master.isna().sum())

print("\nPopulation summary:")
print(master["population_sum"].describe())


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

master.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)