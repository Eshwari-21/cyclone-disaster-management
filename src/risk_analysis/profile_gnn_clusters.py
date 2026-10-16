
"""Inspect hazard and population profiles of GNN clusters."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/modeling_grid_2024.csv")
CLUSTER_PATH = Path(
    "data/processed/baseline_vs_gnn_clusters_2024.csv"
)
OUTPUT_PATH = Path(
    "data/processed/gnn_cluster_profiles_2024.csv"
)


def profile_gnn_clusters():
    data = pd.read_csv(DATA_PATH)
    clusters = pd.read_csv(CLUSTER_PATH)

    required_data = {
        "grid_id",
        "max_hazard",
        "cumulative_hazard",
        "population_sum",
    }
    required_clusters = {"grid_id", "gnn_cluster"}

    if not required_data.issubset(data.columns):
        raise ValueError(
            f"Missing data columns: {required_data - set(data.columns)}"
        )

    if not required_clusters.issubset(clusters.columns):
        raise ValueError(
            "The cluster file must contain grid_id and gnn_cluster. "
            f"Found columns: {clusters.columns.tolist()}"
        )

    if data["grid_id"].duplicated().any():
        raise ValueError("Duplicate grid IDs in modeling data.")

    if clusters["grid_id"].duplicated().any():
        raise ValueError("Duplicate grid IDs in cluster data.")

    merged = data.merge(
        clusters[["grid_id", "gnn_cluster"]],
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if merged["gnn_cluster"].isna().any():
        raise ValueError(
            "Some grid cells do not have a GNN cluster assignment."
        )

    profile = (
        merged.groupby("gnn_cluster")
        .agg(
            grid_cells=("grid_id", "count"),
            mean_max_hazard=("max_hazard", "mean"),
            mean_cumulative_hazard=("cumulative_hazard", "mean"),
            mean_population=("population_sum", "mean"),
            median_population=("population_sum", "median"),
        )
        .round(3)
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUTPUT_PATH, index=False)

    print("\nGNN CLUSTER PROFILES")
    print(profile.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    profile_gnn_clusters()
