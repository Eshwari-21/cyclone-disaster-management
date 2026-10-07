from pathlib import Path

import geopandas as gpd
import pandas as pd
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
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

BASELINE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_baseline_clusters_2024.csv"
)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_embeddings_2024.csv"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N_CLUSTERS = 2


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

modeling_df = pd.read_csv(MODEL_FILE)

grid_gdf = gpd.read_file(GRID_FILE)

baseline_df = pd.read_csv(BASELINE_FILE)

embedding_df = pd.read_csv(EMBEDDING_FILE)

print(f"Modeling rows: {len(modeling_df):,}")
print(f"Grid cells: {len(grid_gdf):,}")
print(f"Baseline rows: {len(baseline_df):,}")
print(f"Embedding rows: {len(embedding_df):,}")


# ---------------------------------------------------------
# Validate grid IDs
# ---------------------------------------------------------

datasets = {
    "modeling": modeling_df,
    "baseline": baseline_df,
    "embedding": embedding_df,
}

for name, df in datasets.items():

    if "grid_id" not in df.columns:
        raise ValueError(
            f"{name} dataset does not contain grid_id."
        )

    if df["grid_id"].duplicated().any():
        raise ValueError(
            f"Duplicate grid_id values found in {name} dataset."
        )


reference_ids = set(
    modeling_df["grid_id"]
)

if set(grid_gdf["grid_id"]) != reference_ids:
    raise ValueError(
        "Spatial grid does not match modeling dataset."
    )

if set(baseline_df["grid_id"]) != reference_ids:
    raise ValueError(
        "Baseline clusters do not match modeling dataset."
    )

if set(embedding_df["grid_id"]) != reference_ids:
    raise ValueError(
        "Embeddings do not match modeling dataset."
    )

print()
print("Grid ID validation passed.")


# ---------------------------------------------------------
# Reproduce GNN seed-42 clustering
# ---------------------------------------------------------

embedding_columns = [
    column
    for column in embedding_df.columns
    if column.startswith("embedding_")
]

if len(embedding_columns) != 8:
    raise ValueError(
        f"Expected 8 embedding columns, "
        f"found {len(embedding_columns)}."
    )

print(
    f"GNN embedding dimensions: "
    f"{len(embedding_columns)}"
)


# ---------------------------------------------------------
# Important note:
# spatial_embeddings_2024.csv represents the latest
# spatial_model.py run.
#
# We use K-Means with the same configuration used by
# spatial_model.py to reproduce its cluster assignment.
# ---------------------------------------------------------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

gnn_labels = kmeans.fit_predict(
    embedding_df[embedding_columns]
)

gnn_silhouette = silhouette_score(
    embedding_df[embedding_columns],
    gnn_labels
)


# ---------------------------------------------------------
# Baseline clusters
# ---------------------------------------------------------

baseline_labels = (
    baseline_df["baseline_cluster"]
    .to_numpy()
)


# ---------------------------------------------------------
# Compare cluster assignments
# ---------------------------------------------------------

ari = adjusted_rand_score(
    baseline_labels,
    gnn_labels
)

print()
print("MODEL COMPARISON")
print("--------------------------------")

print(
    f"PCA baseline silhouette: "
    f"0.4581"
)

print(
    f"GNN embedding silhouette: "
    f"{gnn_silhouette:.4f}"
)

print(
    f"PCA vs GNN Adjusted Rand Index: "
    f"{ari:.4f}"
)


# ---------------------------------------------------------
# Cluster counts
# ---------------------------------------------------------

baseline_counts = (
    pd.Series(baseline_labels)
    .value_counts()
    .sort_index()
)

gnn_counts = (
    pd.Series(gnn_labels)
    .value_counts()
    .sort_index()
)

print()
print("CLUSTER SIZE COMPARISON")
print("--------------------------------")

for cluster_id in range(N_CLUSTERS):

    baseline_count = baseline_counts.get(
        cluster_id,
        0
    )

    gnn_count = gnn_counts.get(
        cluster_id,
        0
    )

    print(
        f"Cluster {cluster_id}: "
        f"PCA={baseline_count:,} | "
        f"GNN={gnn_count:,}"
    )


# ---------------------------------------------------------
# Cross-tabulation
# ---------------------------------------------------------

comparison_df = pd.DataFrame(
    {
        "baseline_cluster": baseline_labels,
        "gnn_cluster": gnn_labels,
    }
)

cross_tab = pd.crosstab(
    comparison_df["baseline_cluster"],
    comparison_df["gnn_cluster"]
)

print()
print("PCA vs GNN CLUSTER CROSS-TABULATION")
print("--------------------------------")

print(
    cross_tab.to_string()
)


# ---------------------------------------------------------
# Save comparison
# ---------------------------------------------------------

comparison_output = modeling_df[
    ["grid_id"]
].copy()

comparison_output["baseline_cluster"] = (
    baseline_labels
)

comparison_output["gnn_cluster"] = (
    gnn_labels
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_vs_gnn_clusters_2024.csv"
)

comparison_output.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    f"Saved comparison file to: "
    f"{OUTPUT_FILE}"
)

print()
print("Model comparison completed.")