from pathlib import Path

import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------
# Project paths
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

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_embeddings_2024.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "decision_support_grid_2024.gpkg"
)


# ---------------------------------------------------------
# Load final modeling data
# ---------------------------------------------------------

modeling_df = pd.read_csv(
    MODELING_FILE
)

grid_gdf = gpd.read_file(
    GRID_FILE
)

embedding_df = pd.read_csv(
    EMBEDDING_FILE
)

print(
    f"Modeling rows: {len(modeling_df):,}"
)

print(
    f"Grid cells: {len(grid_gdf):,}"
)

print(
    f"Embedding rows: {len(embedding_df):,}"
)


# ---------------------------------------------------------
# Validate grid IDs
# ---------------------------------------------------------

for dataframe, name in [
    (modeling_df, "modeling dataset"),
    (grid_gdf, "spatial grid"),
    (embedding_df, "spatial embeddings"),
]:

    if "grid_id" not in dataframe.columns:

        raise ValueError(
            f"grid_id missing from {name}."
        )

    if dataframe["grid_id"].duplicated().any():

        raise ValueError(
            f"Duplicate grid_id values found "
            f"in {name}."
        )


if set(modeling_df["grid_id"]) != set(
    grid_gdf["grid_id"]
):

    raise ValueError(
        "Modeling dataset and spatial grid "
        "do not contain the same grid IDs."
    )


if set(modeling_df["grid_id"]) != set(
    embedding_df["grid_id"]
):

    raise ValueError(
        "Modeling dataset and spatial embeddings "
        "do not contain the same grid IDs."
    )


print(
    "Grid ID validation passed."
)


# ---------------------------------------------------------
# Validate final embedding dimensions
# ---------------------------------------------------------

embedding_columns = [
    column
    for column in embedding_df.columns
    if column.startswith("embedding_")
]

print(
    f"Embedding dimensions found: "
    f"{len(embedding_columns)}"
)

if len(embedding_columns) != 8:

    raise ValueError(
        "Expected 8 final GNN embedding dimensions."
    )


# ---------------------------------------------------------
# Recreate final cluster labels
# ---------------------------------------------------------

# The final spatial model selected k=2.
#
# The actual cluster labels are not stored directly
# in the embedding CSV, so they must be reconstructed
# using the same final 8-D embeddings and K-Means setup.

from sklearn.cluster import KMeans


embedding_matrix = embedding_df[
    embedding_columns
].to_numpy()


final_kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10,
)


final_labels = final_kmeans.fit_predict(
    embedding_matrix
)


cluster_df = embedding_df[
    ["grid_id"]
].copy()

cluster_df["spatial_cluster"] = (
    final_labels
)


print(
    "Final spatial cluster labels generated."
)

print(
    "Cluster counts:"
)

print(
    cluster_df["spatial_cluster"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Select decision-support attributes
# ---------------------------------------------------------

decision_columns = [
    "grid_id",
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
    "population_sum",
    "log_population",
    "max_hazard",
    "cumulative_hazard",
]

missing_decision_columns = [
    column
    for column in decision_columns
    if column not in modeling_df.columns
]

if missing_decision_columns:

    raise ValueError(
        "Missing decision-support columns: "
        f"{missing_decision_columns}"
    )


decision_df = modeling_df[
    decision_columns
].copy()


# ---------------------------------------------------------
# Merge final spatial clusters
# ---------------------------------------------------------

decision_df = decision_df.merge(
    cluster_df,
    on="grid_id",
    how="left",
    validate="one_to_one",
)


if decision_df["spatial_cluster"].isna().any():

    raise ValueError(
        "Some grid cells do not have "
        "a spatial cluster."
    )


# ---------------------------------------------------------
# Merge with geographic grid
# ---------------------------------------------------------

decision_gdf = grid_gdf.merge(
    decision_df,
    on="grid_id",
    how="left",
    validate="one_to_one",
)


if len(decision_gdf) != len(grid_gdf):

    raise ValueError(
        "Decision-support grid changed "
        "the number of spatial cells."
    )


if decision_gdf[
    "spatial_cluster"
].isna().any():

    raise ValueError(
        "Some spatial grid cells are missing "
        "their final cluster assignment."
    )


# ---------------------------------------------------------
# Validate geometry
# ---------------------------------------------------------

if decision_gdf.geometry.isna().any():

    raise ValueError(
        "Missing geometries found."
    )


if not decision_gdf.geometry.is_valid.all():

    raise ValueError(
        "Invalid geometries found."
    )


# ---------------------------------------------------------
# Save decision-support grid
# ---------------------------------------------------------

decision_gdf.to_file(
    OUTPUT_FILE,
    driver="GPKG",
)

print()
print(
    "Decision-support grid created."
)

print(
    f"Output file: {OUTPUT_FILE}"
)

print(
    f"Output cells: {len(decision_gdf):,}"
)

print(
    f"Output CRS: {decision_gdf.crs}"
)

print(
    "Decision-support grid validation passed."
)