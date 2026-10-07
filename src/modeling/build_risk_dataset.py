from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMD_EVENT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "imd_grid_event_features_2024.csv"
)

MASTER_GRID_FILE = (
    PROJECT_ROOT / "data" / "processed" / "master_grid_2024.csv"
)
imd = pd.read_csv(IMD_EVENT_FILE)
master = pd.read_csv(MASTER_GRID_FILE)

print(f"IMD event rows: {len(imd):,}")
print(f"Master grid rows: {len(master):,}")
required_imd = [
    "grid_id",
    "event_id",
    "msw_kt",
    "distance_km",
]

required_master = [
    "grid_id",
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
    "population_sum",
]

missing_imd = [col for col in required_imd if col not in imd.columns]
missing_master = [col for col in required_master if col not in master.columns]

if missing_imd:
    raise ValueError(f"Missing IMD columns: {missing_imd}")

if missing_master:
    raise ValueError(f"Missing master-grid columns: {missing_master}")

print("Required columns validated.")
intensity_min = imd["msw_kt"].min()
intensity_max = imd["msw_kt"].max()

imd["intensity_norm"] = (
    (imd["msw_kt"] - intensity_min)
    / (intensity_max - intensity_min)
)

print("Intensity normalization complete.")
print(
    imd["intensity_norm"]
    .describe()
    .to_string()
)
distance_scale_km = 100.0

raw_proximity = 1 / (1 + imd["distance_km"] / distance_scale_km)

proximity_min = raw_proximity.min()
proximity_max = raw_proximity.max()

imd["proximity_norm"] = (
    (raw_proximity - proximity_min)
    / (proximity_max - proximity_min)
)

print("Proximity normalization complete.")
print(
    imd["proximity_norm"]
    .describe()
    .to_string()
)
imd["hazard"] = (
    imd["intensity_norm"]
    * imd["proximity_norm"]
)

print("Event-level hazard calculated.")
print(
    imd["hazard"]
    .describe()
    .to_string()
)
hazard_summary = (
    imd.groupby("grid_id")["hazard"]
    .agg(
        max_hazard="max",
        cumulative_hazard="sum",
    )
    .reset_index()
)

print("Grid-level hazard aggregation complete.")
print(f"Grid cells: {len(hazard_summary):,}")
print(hazard_summary.describe().to_string())
modeling_grid = master.merge(
    hazard_summary,
    on="grid_id",
    how="left",
    validate="one_to_one",
)

print("Modeling grid created.")
print(f"Rows: {len(modeling_grid):,}")
print(f"Columns: {len(modeling_grid.columns)}")
print(
    modeling_grid[
        [
            "grid_id",
            "elevation_m",
            "slope_deg",
            "annual_rainfall_mm",
            "mean_ndvi",
            "population_sum",
            "max_hazard",
            "cumulative_hazard",
        ]
    ].head().to_string(index=False)
)
if len(modeling_grid) != len(master):
    raise ValueError(
        f"Unexpected row count: {len(modeling_grid)} "
        f"(expected {len(master)})"
    )

if modeling_grid["grid_id"].duplicated().any():
    raise ValueError("Duplicate grid_id values found.")

hazard_missing = modeling_grid[
    ["max_hazard", "cumulative_hazard"]
].isna().sum()

if hazard_missing.any():
    raise ValueError(
        f"Missing hazard values found:\n{hazard_missing}"
    )

print("Modeling dataset validation passed.")
print("\nMissing values:")
print(
    modeling_grid[
        [
            "elevation_m",
            "slope_deg",
            "annual_rainfall_mm",
            "mean_ndvi",
            "population_sum",
            "max_hazard",
            "cumulative_hazard",
        ]
    ].isna().sum().to_string()
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "modeling_grid_2024.csv"
)
modeling_grid["log_population"] = (
    __import__("numpy").log1p(modeling_grid["population_sum"])
)

print("Log population feature created.")

# Preserve missingness information before imputation.
environmental_cols = [
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
]

for col in environmental_cols:
    modeling_grid[f"{col}_missing"] = modeling_grid[col].isna().astype(int)

# Median imputation for the small number of missing environmental values.
for col in environmental_cols:
    median_value = modeling_grid[col].median()
    modeling_grid[col] = modeling_grid[col].fillna(median_value)

print("Environmental missing values imputed using dataset medians.")
print("Missingness flags created.")

# Build baseline hazard-exposure index.
baseline_features = [
    "max_hazard",
    "cumulative_hazard",
    "log_population",
]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
z_scores = scaler.fit_transform(modeling_grid[baseline_features])

modeling_grid["baseline_hazard_exposure_index"] = z_scores.mean(axis=1)

# Relative priority classes based on study-area quartiles.
modeling_grid["baseline_priority_class"] = pd.qcut(
    modeling_grid["baseline_hazard_exposure_index"],
    q=4,
    labels=["Low", "Moderate", "High", "Very High"],
)

print("Baseline hazard-exposure index created.")
print("Baseline priority classes created.")
print(
    modeling_grid["baseline_priority_class"]
    .value_counts()
    .sort_index()
    .to_string()
)
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "modeling_grid_2024.csv"
modeling_grid.to_csv(OUTPUT_FILE, index=False)