from pathlib import Path

import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_grid_2024.csv"
)

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coastal_ap_grid_1km.gpkg"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
    "log_population",
    "max_hazard",
    "cumulative_hazard",
]

N_CLUSTERS = 2


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

modeling_df = pd.read_csv(MODELING_FILE)
grid_gdf = gpd.read_file(GRID_FILE)

print(f"Modeling rows: {len(modeling_df):,}")
print(f"Grid cells: {len(grid_gdf):,}")


# ---------------------------------------------------------
# Validate alignment
# ---------------------------------------------------------

if modeling_df["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found in modeling dataset."
    )

if grid_gdf["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found in spatial grid."
    )

if set(modeling_df["grid_id"]) != set(grid_gdf["grid_id"]):
    raise ValueError(
        "Modeling dataset and spatial grid are not aligned."
    )

print("Modeling dataset and spatial grid are aligned.")


# ---------------------------------------------------------
# Validate features
# ---------------------------------------------------------

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in modeling_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

if modeling_df[FEATURE_COLUMNS].isna().any().any():
    raise ValueError(
        "Missing values found in modeling features."
    )


# ---------------------------------------------------------
# Standardize features
# ---------------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(
    modeling_df[FEATURE_COLUMNS]
)

print(
    f"Standardized feature matrix shape: "
    f"{X.shape}"
)


# ---------------------------------------------------------
# PCA dimensionality reduction
# ---------------------------------------------------------

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X)

explained_variance = (
    pca.explained_variance_ratio_
)

print()
print("PCA results")
print("--------------------------------")
print(
    f"PC1 explained variance: "
    f"{explained_variance[0]:.4f}"
)

print(
    f"PC2 explained variance: "
    f"{explained_variance[1]:.4f}"
)

print(
    f"Total explained variance: "
    f"{explained_variance.sum():.4f}"
)


# ---------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(
    X_pca
)

silhouette = silhouette_score(
    X_pca,
    labels
)

print()
print("Baseline clustering")
print("--------------------------------")
print(
    f"Clusters: {N_CLUSTERS}"
)

print(
    f"Silhouette score: "
    f"{silhouette:.4f}"
)


# ---------------------------------------------------------
# Cluster counts
# ---------------------------------------------------------

cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

print()
print("Cluster counts")
print("--------------------------------")

for cluster_id, count in cluster_counts.items():

    print(
        f"Cluster {cluster_id}: "
        f"{count:,} cells"
    )


# ---------------------------------------------------------
# Spatial graph
# ---------------------------------------------------------

weights = Queen.from_dataframe(
    grid_gdf,
    use_index=True
)

weights.transform = "r"


# ---------------------------------------------------------
# Spatial neighbor agreement
# ---------------------------------------------------------

neighbor_agreement = []

for node_index, neighbors in weights.neighbors.items():

    if len(neighbors) == 0:
        continue

    same_cluster = sum(
        labels[node_index] == labels[neighbor]
        for neighbor in neighbors
    )

    agreement = (
        same_cluster / len(neighbors)
    )

    neighbor_agreement.append(
        agreement
    )

neighbor_agreement = pd.Series(
    neighbor_agreement
)

print()
print("Baseline spatial coherence")
print("--------------------------------")

print(
    f"Mean neighbor agreement: "
    f"{neighbor_agreement.mean():.4f}"
)

print(
    f"Median neighbor agreement: "
    f"{neighbor_agreement.median():.4f}"
)

print(
    f"Cells with >=75% same-cluster neighbors: "
    f"{(neighbor_agreement >= 0.75).sum():,}"
)

print(
    f"Cells with <25% same-cluster neighbors: "
    f"{(neighbor_agreement < 0.25).sum():,}"
)


# ---------------------------------------------------------
# Cluster profiles
# ---------------------------------------------------------

profile_df = modeling_df.copy()

profile_df["spatial_cluster"] = labels

profile_columns = [
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
    "population_sum",
    "max_hazard",
    "cumulative_hazard",
]

cluster_profiles = (
    profile_df
    .groupby("spatial_cluster")[profile_columns]
    .median()
)

print()
print("Baseline cluster profiles")
print("--------------------------------")

print(
    cluster_profiles.to_string()
)


# ---------------------------------------------------------
# Save baseline cluster output
# ---------------------------------------------------------

output_df = modeling_df[
    ["grid_id"]
].copy()

output_df["baseline_cluster"] = labels

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_baseline_clusters_2024.csv"
)

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    f"Saved baseline cluster assignments to: "
    f"{OUTPUT_FILE}"
)

print()
print("Deterministic spatial baseline completed.")