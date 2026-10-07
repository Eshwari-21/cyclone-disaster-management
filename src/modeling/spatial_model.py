from pathlib import Path

import geopandas as gpd
import pandas as pd
import torch

from libpysal.weights import Queen
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
# Set random seed for reproducibility
# ---------------------------------------------------------

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

print(
    f"PyTorch random seed: {RANDOM_SEED}"
)


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

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
# Features used by the spatial model
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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

modeling_df = pd.read_csv(MODELING_FILE)
grid_gdf = gpd.read_file(GRID_FILE)

print(
    f"Modeling rows: {len(modeling_df):,}"
)

print(
    f"Grid cells: {len(grid_gdf):,}"
)


# ---------------------------------------------------------
# Validate grid IDs
# ---------------------------------------------------------

if "grid_id" not in modeling_df.columns:
    raise ValueError(
        "grid_id missing from modeling dataset."
    )

if "grid_id" not in grid_gdf.columns:
    raise ValueError(
        "grid_id missing from grid dataset."
    )

if modeling_df["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found in modeling dataset."
    )

if grid_gdf["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found in grid dataset."
    )

if set(modeling_df["grid_id"]) != set(
    grid_gdf["grid_id"]
):
    raise ValueError(
        "Modeling grid IDs do not match spatial grid IDs."
    )

print(
    "Modeling dataset and spatial grid are aligned."
)


# ---------------------------------------------------------
# Validate model features
# ---------------------------------------------------------

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in modeling_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )

feature_missing = (
    modeling_df[FEATURE_COLUMNS]
    .isna()
    .sum()
)

if feature_missing.any():
    raise ValueError(
        "Missing values found in model features:\n"
        f"{feature_missing[feature_missing > 0]}"
    )


# ---------------------------------------------------------
# Standardize features
# ---------------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(
    modeling_df[FEATURE_COLUMNS]
)

print(
    f"Feature matrix shape: {X.shape}"
)

print(
    "Feature standardization complete."
)


# ---------------------------------------------------------
# Convert features to PyTorch tensor
# ---------------------------------------------------------

X_tensor = torch.tensor(
    X,
    dtype=torch.float32,
)

print(
    f"PyTorch feature tensor shape: "
    f"{X_tensor.shape}"
)

print(
    f"PyTorch feature tensor dtype: "
    f"{X_tensor.dtype}"
)


# ---------------------------------------------------------
# Build spatial graph
# ---------------------------------------------------------

weights = Queen.from_dataframe(
    grid_gdf,
    use_index=True,
)

print(
    f"Graph nodes: {len(weights.neighbors):,}"
)

print(
    "Graph neighbor links:",
    sum(
        len(neighbors)
        for neighbors in weights.neighbors.values()
    ),
)

print(
    "Average neighbors:",
    sum(
        len(neighbors)
        for neighbors in weights.neighbors.values()
    )
    / len(weights.neighbors),
)


# ---------------------------------------------------------
# Validate graph connectivity
# ---------------------------------------------------------

if any(
    len(neighbors) == 0
    for neighbors in weights.neighbors.values()
):
    raise ValueError(
        "Spatial graph contains isolated grid cells."
    )

print(
    "Spatial graph construction complete."
)


# ---------------------------------------------------------
# Convert graph to edge list
# ---------------------------------------------------------

edge_pairs = []

for source_node, neighbor_nodes in (
    weights.neighbors.items()
):

    for target_node in neighbor_nodes:

        edge_pairs.append(
            (source_node, target_node)
        )

print(
    f"Directed edges: {len(edge_pairs):,}"
)

if len(edge_pairs) == 0:
    raise ValueError(
        "No graph edges were created."
    )

print(
    "Node-to-feature and node-to-edge "
    "ordering prepared."
)


# ---------------------------------------------------------
# Validate feature/graph alignment
# ---------------------------------------------------------

if len(X_tensor) != len(grid_gdf):
    raise ValueError(
        "Feature matrix and spatial grid "
        "have different numbers of nodes."
    )

node_indices = set(
    range(len(grid_gdf))
)

edge_nodes = {
    node
    for edge in edge_pairs
    for node in edge
}

if edge_nodes != node_indices:

    missing_nodes = (
        node_indices - edge_nodes
    )

    raise ValueError(
        "Some graph nodes are not represented "
        f"in edges: {len(missing_nodes)}"
    )

print(
    "Graph validation passed."
)

print(
    f"Feature nodes: {len(X_tensor):,}"
)

print(
    f"Connected nodes: {len(edge_nodes):,}"
)


# ---------------------------------------------------------
# Build PyTorch graph edge representation
# ---------------------------------------------------------

source_nodes = torch.tensor(
    [edge[0] for edge in edge_pairs],
    dtype=torch.long,
)

target_nodes = torch.tensor(
    [edge[1] for edge in edge_pairs],
    dtype=torch.long,
)

edge_index = torch.stack(
    [source_nodes, target_nodes],
    dim=0,
)

print(
    f"PyTorch edge index shape: {edge_index.shape}"
)

print(
    f"PyTorch edge index dtype: {edge_index.dtype}"
)

if edge_index.shape[1] != len(edge_pairs):
    raise ValueError(
        "Edge index does not match edge list."
    )

print(
    "PyTorch graph representation created."
)


# ---------------------------------------------------------
# Build normalized spatial adjacency
# ---------------------------------------------------------

num_nodes = len(grid_gdf)


# Add self-loops so each grid cell keeps
# its own features.

self_loops = torch.arange(
    num_nodes,
    dtype=torch.long,
)

self_loop_edges = torch.stack(
    [self_loops, self_loops],
    dim=0,
)


# Combine spatial neighbor edges with self-loops.

adjacency_edges = torch.cat(
    [edge_index, self_loop_edges],
    dim=1,
)


# Compute node degree.

degree = torch.bincount(
    adjacency_edges[0],
    minlength=num_nodes,
).float()


# Compute D^(-1/2).

degree_inv_sqrt = degree.pow(-0.5)


# Compute normalized edge weights:
# D^(-1/2) A D^(-1/2)

source_degree = degree_inv_sqrt[
    adjacency_edges[0]
]

target_degree = degree_inv_sqrt[
    adjacency_edges[1]
]

edge_weights = (
    source_degree
    * target_degree
)

print(
    f"Adjacency edges including self-loops: "
    f"{adjacency_edges.shape[1]:,}"
)

print(
    f"Normalized edge weights: "
    f"{edge_weights.shape}"
)

if torch.isnan(edge_weights).any():
    raise ValueError(
        "NaN values found in normalized edge weights."
    )

print(
    "Normalized spatial adjacency created."
)


# ---------------------------------------------------------
# Define graph message passing
# ---------------------------------------------------------

def graph_message_passing(
    node_features,
    adjacency_edges,
    edge_weights,
):
    """
    Aggregate normalized neighboring-node features.

    Each node receives information from its spatial
    neighbors and from itself.
    """

    source_nodes = adjacency_edges[0]
    target_nodes = adjacency_edges[1]

    messages = (
        node_features[source_nodes]
        * edge_weights.unsqueeze(1)
    )

    aggregated_features = torch.zeros_like(
        node_features
    )

    aggregated_features.index_add_(
        0,
        target_nodes,
        messages,
    )

    return aggregated_features


# ---------------------------------------------------------
# Run one spatial message-passing step
# ---------------------------------------------------------

spatial_features = graph_message_passing(
    X_tensor,
    adjacency_edges,
    edge_weights,
)

print(
    f"Spatial feature tensor shape: "
    f"{spatial_features.shape}"
)

print(
    f"Spatial feature tensor dtype: "
    f"{spatial_features.dtype}"
)

if spatial_features.shape != X_tensor.shape:
    raise ValueError(
        "Spatial message passing changed "
        "the feature matrix shape unexpectedly."
    )

if torch.isnan(spatial_features).any():
    raise ValueError(
        "NaN values found after spatial message passing."
    )

print(
    "Spatial message passing completed."
)


# ---------------------------------------------------------
# Define a simple graph convolution layer
# ---------------------------------------------------------

class GraphConvolution(torch.nn.Module):

    def __init__(
        self,
        input_features,
        output_features,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            input_features,
            output_features,
        )

    def forward(
        self,
        node_features,
        adjacency_edges,
        edge_weights,
    ):

        spatial_features = graph_message_passing(
            node_features,
            adjacency_edges,
            edge_weights,
        )

        return self.linear(
            spatial_features
        )


# ---------------------------------------------------------
# Create the graph convolution model
# ---------------------------------------------------------

input_features = X_tensor.shape[1]

hidden_features = 8

graph_layer = GraphConvolution(
    input_features=input_features,
    output_features=hidden_features,
)

print(
    f"Graph convolution input features: "
    f"{input_features}"
)

print(
    f"Graph convolution output features: "
    f"{hidden_features}"
)

print(
    "Graph convolution layer initialized."
)


# ---------------------------------------------------------
# Test the graph convolution layer
# ---------------------------------------------------------

graph_output = graph_layer(
    X_tensor,
    adjacency_edges,
    edge_weights,
)

print(
    f"Graph convolution output shape: "
    f"{graph_output.shape}"
)

print(
    f"Graph convolution output dtype: "
    f"{graph_output.dtype}"
)

if graph_output.shape != (
    len(grid_gdf),
    hidden_features,
):
    raise ValueError(
        "Unexpected graph convolution output shape."
    )

if torch.isnan(graph_output).any():
    raise ValueError(
        "NaN values found in graph convolution output."
    )

print(
    "Graph convolution forward pass completed."
)


# ---------------------------------------------------------
# Create the reconstruction decoder
# ---------------------------------------------------------

decoder = torch.nn.Linear(
    hidden_features,
    input_features,
)

print(
    f"Decoder input features: "
    f"{hidden_features}"
)

print(
    f"Decoder output features: "
    f"{input_features}"
)

print(
    "Reconstruction decoder initialized."
)


# ---------------------------------------------------------
# Define reconstruction loss
# ---------------------------------------------------------

reconstructed_features = decoder(
    graph_output
)

loss_function = torch.nn.MSELoss()

reconstruction_loss = loss_function(
    reconstructed_features,
    X_tensor,
)

print(
    f"Reconstructed feature shape: "
    f"{reconstructed_features.shape}"
)

print(
    f"Initial reconstruction loss: "
    f"{reconstruction_loss.item():.6f}"
)

if torch.isnan(reconstruction_loss):
    raise ValueError(
        "NaN value found in reconstruction loss."
    )

print(
    "Reconstruction loss calculation completed."
)


# ---------------------------------------------------------
# Create the optimizer
# ---------------------------------------------------------

optimizer = torch.optim.Adam(
    list(graph_layer.parameters())
    + list(decoder.parameters()),
    lr=0.001,
)

print(
    "Adam optimizer initialized."
)

print(
    "Learning rate: 0.001"
)


# ---------------------------------------------------------
# Train the spatial representation model
# ---------------------------------------------------------

epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()

    embeddings = graph_layer(
        X_tensor,
        adjacency_edges,
        edge_weights,
    )

    reconstructed = decoder(
        embeddings
    )

    loss = loss_function(
        reconstructed,
        X_tensor,
    )

    loss.backward()

    optimizer.step()

    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch {epoch + 1:03d}/{epochs} "
            f"- Loss: {loss.item():.6f}"
        )

print(
    "Spatial representation model training completed."
)


# ---------------------------------------------------------
# Extract learned spatial embeddings
# ---------------------------------------------------------

graph_layer.eval()

with torch.no_grad():

    embeddings = graph_layer(
        X_tensor,
        adjacency_edges,
        edge_weights,
    )

print(
    f"Learned embedding shape: "
    f"{embeddings.shape}"
)

print(
    f"Learned embedding dtype: "
    f"{embeddings.dtype}"
)

if embeddings.shape != (
    len(grid_gdf),
    hidden_features,
):
    raise ValueError(
        "Unexpected learned embedding shape."
    )

if torch.isnan(embeddings).any():
    raise ValueError(
        "NaN values found in learned embeddings."
    )

if torch.isinf(embeddings).any():
    raise ValueError(
        "Infinite values found in learned embeddings."
    )

print(
    "Learned spatial embeddings extracted."
)


# ---------------------------------------------------------
# Save learned spatial embeddings
# ---------------------------------------------------------

embedding_columns = [
    f"embedding_{i + 1}"
    for i in range(hidden_features)
]

embedding_df = pd.DataFrame(
    embeddings.detach().numpy(),
    columns=embedding_columns,
)

embedding_df.insert(
    0,
    "grid_id",
    modeling_df["grid_id"].values,
)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_embeddings_2024.csv"
)

embedding_df.to_csv(
    EMBEDDING_FILE,
    index=False,
)

print(
    f"Saved spatial embeddings to: "
    f"{EMBEDDING_FILE}"
)

print(
    f"Saved embedding rows: "
    f"{len(embedding_df):,}"
)

print(
    f"Saved embedding dimensions: "
    f"{len(embedding_columns)}"
)


# ---------------------------------------------------------
# Validate saved spatial embeddings
# ---------------------------------------------------------

saved_embeddings = pd.read_csv(
    EMBEDDING_FILE
)

if len(saved_embeddings) != len(modeling_df):
    raise ValueError(
        "Saved embeddings have a different "
        "number of rows than the modeling dataset."
    )

if saved_embeddings["grid_id"].duplicated().any():
    raise ValueError(
        "Duplicate grid_id values found "
        "in saved embeddings."
    )

if not saved_embeddings["grid_id"].equals(
    modeling_df["grid_id"]
):
    raise ValueError(
        "Saved embedding grid_id ordering does "
        "not match the modeling dataset."
    )

if saved_embeddings[embedding_columns].isna().any().any():
    raise ValueError(
        "Missing values found in saved embeddings."
    )

print(
    "Saved embedding validation passed."
)

print(
    f"Validated embedding rows: "
    f"{len(saved_embeddings):,}"
)

print(
    f"Validated embedding dimensions: "
    f"{len(embedding_columns)}"
)


# ---------------------------------------------------------
# Cluster learned spatial embeddings
# ---------------------------------------------------------

embedding_matrix = saved_embeddings[
    embedding_columns
].to_numpy()

cluster_results = []

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=10,
    )

    cluster_labels = kmeans.fit_predict(
        embedding_matrix
    )

    silhouette = silhouette_score(
        embedding_matrix,
        cluster_labels,
    )

    cluster_results.append(
        {
            "k": k,
            "silhouette_score": silhouette,
        }
    )

    print(
        f"Embedding K-Means | "
        f"k={k} | "
        f"silhouette={silhouette:.4f}"
    )

cluster_results_df = pd.DataFrame(
    cluster_results
)

best_result = cluster_results_df.loc[
    cluster_results_df[
        "silhouette_score"
    ].idxmax()
]

print()

print(
    "Best embedding cluster count: "
    f"k={int(best_result['k'])}"
)

print(
    "Best embedding silhouette score: "
    f"{best_result['silhouette_score']:.4f}"
)


# ---------------------------------------------------------
# Profile the best embedding clusters
# ---------------------------------------------------------

best_k = int(
    best_result["k"]
)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_SEED,
    n_init=10,
)

final_labels = final_kmeans.fit_predict(
    embedding_matrix
)

cluster_profile_df = modeling_df[
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
].copy()

cluster_profile_df["spatial_cluster"] = (
    final_labels
)

profile_columns = [
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
    "population_sum",
    "max_hazard",
    "cumulative_hazard",
]

cluster_summary = (
    cluster_profile_df
    .groupby("spatial_cluster")[profile_columns]
    .median()
)

cluster_counts = (
    cluster_profile_df["spatial_cluster"]
    .value_counts()
    .sort_index()
)

print()

print(
    "Spatial cluster profiles"
)

print(
    "------------------------"
)

for cluster_id in cluster_summary.index:

    print()

    print(
        f"Cluster {cluster_id}: "
        f"{cluster_counts[cluster_id]:,} cells"
    )

    for column in profile_columns:

        print(
            f"  {column}: "
            f"{cluster_summary.loc[cluster_id, column]:.4f}"
        )


# ---------------------------------------------------------
# Validate spatial cluster coherence
# ---------------------------------------------------------

cluster_by_node = {
    node: int(label)
    for node, label in enumerate(final_labels)
}

neighbor_agreement = []

for node, neighbors in weights.neighbors.items():

    same_cluster = sum(
        cluster_by_node[neighbor] == cluster_by_node[node]
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

print(
    "Spatial cluster coherence"
)

print(
    "--------------------------"
)

print(
    f"Mean neighbor agreement: "
    f"{neighbor_agreement.mean():.4f}"
)

print(
    f"Median neighbor agreement: "
    f"{neighbor_agreement.median():.4f}"
)

print(
    f"Minimum neighbor agreement: "
    f"{neighbor_agreement.min():.4f}"
)

print(
    f"Maximum neighbor agreement: "
    f"{neighbor_agreement.max():.4f}"
)

print(
    "Cells with >=75% same-cluster neighbors: "
    f"{(neighbor_agreement >= 0.75).sum():,}"
)

print(
    "Cells with <25% same-cluster neighbors: "
    f"{(neighbor_agreement < 0.25).sum():,}"
)

# ---------------------------------------------------------
# Compare GNN clustering with baseline K-Means
# ---------------------------------------------------------

print()
print("Baseline vs GNN comparison")
print("---------------------------")


# ---------------------------------------------------------
# Baseline K-Means on original standardized features
# ---------------------------------------------------------

baseline_kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_SEED,
    n_init=10,
)

baseline_labels = baseline_kmeans.fit_predict(
    X
)

baseline_silhouette = silhouette_score(
    X,
    baseline_labels,
)

print(
    f"Baseline K-Means silhouette: "
    f"{baseline_silhouette:.4f}"
)

print(
    f"GNN embedding silhouette: "
    f"{best_result['silhouette_score']:.4f}"
)


# ---------------------------------------------------------
# Baseline spatial coherence
# ---------------------------------------------------------

baseline_cluster_by_node = {
    node: int(label)
    for node, label in enumerate(baseline_labels)
}

baseline_neighbor_agreement = []

for node, neighbors in weights.neighbors.items():

    same_cluster = sum(
        baseline_cluster_by_node[neighbor]
        == baseline_cluster_by_node[node]
        for neighbor in neighbors
    )

    agreement = (
        same_cluster / len(neighbors)
    )

    baseline_neighbor_agreement.append(
        agreement
    )

baseline_neighbor_agreement = pd.Series(
    baseline_neighbor_agreement
)

print()

print(
    "Baseline spatial coherence"
)

print(
    "---------------------------"
)

print(
    f"Mean neighbor agreement: "
    f"{baseline_neighbor_agreement.mean():.4f}"
)

print(
    f"Median neighbor agreement: "
    f"{baseline_neighbor_agreement.median():.4f}"
)

print(
    "Cells with >=75% same-cluster neighbors: "
    f"{(baseline_neighbor_agreement >= 0.75).sum():,}"
)

print(
    "Cells with <25% same-cluster neighbors: "
    f"{(baseline_neighbor_agreement < 0.25).sum():,}"
)


# ---------------------------------------------------------
# Compare metrics directly
# ---------------------------------------------------------

silhouette_difference = (
    best_result["silhouette_score"]
    - baseline_silhouette
)

coherence_difference = (
    neighbor_agreement.mean()
    - baseline_neighbor_agreement.mean()
)

print()

print(
    "Improvement from GNN representation"
)

print(
    "-----------------------------------"
)

print(
    f"Silhouette difference: "
    f"{silhouette_difference:+.4f}"
)

print(
    f"Mean neighbor-agreement difference: "
    f"{coherence_difference:+.4f}"
)

# ---------------------------------------------------------
# Embedding-dimension sensitivity analysis
# ---------------------------------------------------------

print()
print("Embedding-dimension sensitivity")
print("--------------------------------")

sensitivity_results = []

for test_hidden_features in [8, 16, 32]:

    # Reset the seed so each dimension starts
    # from the same reproducible initialization.
    torch.manual_seed(RANDOM_SEED)

    # Create a fresh graph convolution model.
    test_graph_layer = GraphConvolution(
        input_features=input_features,
        output_features=test_hidden_features,
    )

    # Create a fresh decoder.
    test_decoder = torch.nn.Linear(
        test_hidden_features,
        input_features,
    )

    # Create a fresh optimizer.
    test_optimizer = torch.optim.Adam(
        list(test_graph_layer.parameters())
        + list(test_decoder.parameters()),
        lr=0.001,
    )

    # Train the test model.
    for epoch in range(epochs):

        test_optimizer.zero_grad()

        test_embeddings = test_graph_layer(
            X_tensor,
            adjacency_edges,
            edge_weights,
        )

        test_reconstructed = test_decoder(
            test_embeddings
        )

        test_loss = loss_function(
            test_reconstructed,
            X_tensor,
        )

        test_loss.backward()

        test_optimizer.step()

    # Extract test embeddings.
    test_graph_layer.eval()

    with torch.no_grad():

        test_embeddings = test_graph_layer(
            X_tensor,
            adjacency_edges,
            edge_weights,
        )

    test_embedding_matrix = (
        test_embeddings.detach().numpy()
    )

    # Cluster the test embeddings.
    test_kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_SEED,
        n_init=10,
    )

    test_labels = test_kmeans.fit_predict(
        test_embedding_matrix
    )

    # Calculate silhouette.
    test_silhouette = silhouette_score(
        test_embedding_matrix,
        test_labels,
    )

    # Calculate spatial neighbor agreement.
    test_cluster_by_node = {
        node: int(label)
        for node, label in enumerate(test_labels)
    }

    test_neighbor_agreement = []

    for node, neighbors in weights.neighbors.items():

        same_cluster = sum(
            test_cluster_by_node[neighbor]
            == test_cluster_by_node[node]
            for neighbor in neighbors
        )

        agreement = (
            same_cluster / len(neighbors)
        )

        test_neighbor_agreement.append(
            agreement
        )

    test_neighbor_agreement = pd.Series(
        test_neighbor_agreement
    )

    mean_neighbor_agreement = (
        test_neighbor_agreement.mean()
    )

    cells_high_coherence = (
        test_neighbor_agreement >= 0.75
    ).sum()

    cells_low_coherence = (
        test_neighbor_agreement < 0.25
    ).sum()

    sensitivity_results.append(
        {
            "embedding_dimensions":
                test_hidden_features,
            "silhouette_score":
                test_silhouette,
            "mean_neighbor_agreement":
                mean_neighbor_agreement,
            "cells_ge_75_percent":
                cells_high_coherence,
            "cells_lt_25_percent":
                cells_low_coherence,
            "final_reconstruction_loss":
                test_loss.item(),
        }
    )

    print(
        f"Dimensions={test_hidden_features:02d} | "
        f"Silhouette={test_silhouette:.4f} | "
        f"Neighbor agreement="
        f"{mean_neighbor_agreement:.4f} | "
        f">=75%="
        f"{cells_high_coherence:,} | "
        f"<25%="
        f"{cells_low_coherence:,} | "
        f"Loss={test_loss.item():.6f}"
    )


sensitivity_results_df = pd.DataFrame(
    sensitivity_results
)

best_dimension_result = (
    sensitivity_results_df.loc[
        sensitivity_results_df[
            "silhouette_score"
        ].idxmax()
    ]
)

print()

print(
    "Best embedding dimension by silhouette: "
    f"{int(best_dimension_result['embedding_dimensions'])}"
)

print(
    "Best sensitivity-test silhouette: "
    f"{best_dimension_result['silhouette_score']:.4f}"
)

print(
    "Corresponding mean neighbor agreement: "
    f"{best_dimension_result['mean_neighbor_agreement']:.4f}"
)
# ---------------------------------------------------------
# Reproducibility summary
# ---------------------------------------------------------

print()

print(
    "Spatial modeling setup summary"
)

print(
    "--------------------------------"
)

print(
    f"Grid cells: {len(grid_gdf):,}"
)

print(
    f"Features: {len(FEATURE_COLUMNS)}"
)

print(
    "Feature names:",
    ", ".join(FEATURE_COLUMNS),
)

print(
    f"Directed edges: {len(edge_pairs):,}"
)

print(
    f"Embedding dimensions: {hidden_features}"
)

print(
    f"Best cluster count: {best_k}"
)

print(
    f"Best silhouette score: "
    f"{best_result['silhouette_score']:.4f}"
)