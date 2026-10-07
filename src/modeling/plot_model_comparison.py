from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coastal_ap_grid_1km.gpkg"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_vs_gnn_clusters_2024.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_vs_gnn_cluster_map_2024.png"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

grid_gdf = gpd.read_file(GRID_FILE)

comparison_df = pd.read_csv(
    COMPARISON_FILE
)

print(
    f"Grid cells: {len(grid_gdf):,}"
)

print(
    f"Comparison rows: {len(comparison_df):,}"
)


# ---------------------------------------------------------
# Validate grid IDs
# ---------------------------------------------------------

if grid_gdf["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found in spatial grid."
    )

if comparison_df["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found in comparison file."
    )

if set(grid_gdf["grid_id"]) != set(
    comparison_df["grid_id"]
):
    raise ValueError(
        "Grid IDs do not match between spatial grid "
        "and model comparison."
    )

print(
    "Grid ID validation passed."
)


# ---------------------------------------------------------
# Merge cluster results with spatial grid
# ---------------------------------------------------------

map_gdf = grid_gdf.merge(
    comparison_df[
        [
            "grid_id",
            "baseline_cluster",
            "gnn_cluster",
        ]
    ],
    on="grid_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Validate geometry
# ---------------------------------------------------------

if map_gdf.geometry.isna().any():
    raise ValueError(
        "Missing geometries found."
    )

if not map_gdf.geometry.is_valid.all():
    raise ValueError(
        "Invalid geometries found."
    )

if map_gdf[
    ["baseline_cluster", "gnn_cluster"]
].isna().any().any():

    raise ValueError(
        "Missing cluster assignments found."
    )

print(
    "Spatial and cluster validation passed."
)


# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 8)
)


# ---------------------------------------------------------
# PCA baseline map
# ---------------------------------------------------------

map_gdf.plot(
    column="baseline_cluster",
    categorical=True,
    legend=True,
    ax=axes[0],
    linewidth=0,
)

axes[0].set_title(
    "PCA Baseline Spatial Clusters"
)

axes[0].set_axis_off()


# ---------------------------------------------------------
# GNN map
# ---------------------------------------------------------

map_gdf.plot(
    column="gnn_cluster",
    categorical=True,
    legend=True,
    ax=axes[1],
    linewidth=0,
)

axes[1].set_title(
    "GNN Spatial Clusters"
)

axes[1].set_axis_off()


# ---------------------------------------------------------
# Figure title
# ---------------------------------------------------------

fig.suptitle(
    "Spatial Cluster Comparison — 1 km Grid",
    fontsize=16,
)

plt.tight_layout()


# ---------------------------------------------------------
# Save figure
# ---------------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print()
print(
    f"Saved spatial comparison map to:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Spatial cluster comparison map completed."
)