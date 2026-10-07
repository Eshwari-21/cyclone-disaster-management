from pathlib import Path

import geopandas as gpd
import pandas as pd
import torch
from libpysal.weights import Queen
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score


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

SEEDS = [7, 21, 42, 84, 123]

EMBEDDING_DIM = 8
EPOCHS = 300
LEARNING_RATE = 0.001
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
# Prepare standardized features
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

scaler = StandardScaler()

X = scaler.fit_transform(
    modeling_df[FEATURE_COLUMNS]
)

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

print(f"Feature matrix shape: {X.shape}")


# ---------------------------------------------------------
# Build spatial graph
# ---------------------------------------------------------

weights = Queen.from_dataframe(
    grid_gdf,
    use_index=True
)

weights.transform = "r"

edge_pairs = []

for source_index, neighbor_indices in weights.neighbors.items():

    for target_index in neighbor_indices:

        edge_pairs.append(
            (source_index, target_index)
        )

edge_index = torch.tensor(
    edge_pairs,
    dtype=torch.long
).t().contiguous()

print(f"Graph nodes: {len(grid_gdf):,}")
print(
    f"Graph directed edges: "
    f"{edge_index.shape[1]:,}"
)


# ---------------------------------------------------------
# Add self-loops and normalized weights
# ---------------------------------------------------------

num_nodes = len(grid_gdf)

self_loops = torch.arange(
    num_nodes,
    dtype=torch.long
)

self_loop_edges = torch.stack(
    [self_loops, self_loops]
)

adjacency_edges = torch.cat(
    [edge_index, self_loop_edges],
    dim=1
)

source_nodes = adjacency_edges[0]
target_nodes = adjacency_edges[1]

degree = torch.bincount(
    source_nodes,
    minlength=num_nodes
).float()

degree_inv_sqrt = degree.pow(-0.5)

edge_weights = (
    degree_inv_sqrt[source_nodes]
    * degree_inv_sqrt[target_nodes]
)


# ---------------------------------------------------------
# Graph message passing
# ---------------------------------------------------------

def graph_message_passing(
    node_features,
    edges,
    weights,
):

    source = edges[0]
    target = edges[1]

    messages = (
        node_features[source]
        * weights.unsqueeze(1)
    )

    aggregated = torch.zeros_like(
        node_features
    )

    aggregated.index_add_(
        0,
        target,
        messages
    )

    return aggregated


# ---------------------------------------------------------
# Graph convolution layer
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
            output_features
        )

    def forward(
        self,
        node_features,
        edges,
        weights,
    ):

        spatial_features = graph_message_passing(
            node_features,
            edges,
            weights
        )

        return self.linear(
            spatial_features
        )


# ---------------------------------------------------------
# Run model for each random seed
# ---------------------------------------------------------

results = []
cluster_labels = {}

for seed in SEEDS:

    print()
    print("=" * 60)
    print(f"Running seed: {seed}")
    print("=" * 60)

    # Set the random seed so that model initialization
    # changes between runs but remains reproducible.
    torch.manual_seed(seed)

    graph_layer = GraphConvolution(
        input_features=len(FEATURE_COLUMNS),
        output_features=EMBEDDING_DIM,
    )

    decoder = torch.nn.Linear(
        EMBEDDING_DIM,
        len(FEATURE_COLUMNS)
    )

    optimizer = torch.optim.Adam(
        list(graph_layer.parameters())
        + list(decoder.parameters()),
        lr=LEARNING_RATE
    )

    loss_function = torch.nn.MSELoss()

    # Store training loss for every epoch.
    loss_history = []

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        embeddings = graph_layer(
            X_tensor,
            adjacency_edges,
            edge_weights
        )

        reconstruction = decoder(
            embeddings
        )

        loss = loss_function(
            reconstruction,
            X_tensor
        )

        loss.backward()

        optimizer.step()

        loss_history.append(
            float(loss.item())
        )

    # -----------------------------------------------------
    # Extract final embeddings
    # -----------------------------------------------------

    graph_layer.eval()

    with torch.no_grad():

        final_embeddings = graph_layer(
            X_tensor,
            adjacency_edges,
            edge_weights
        )

    embedding_matrix = (
        final_embeddings
        .cpu()
        .numpy()
    )

    # -----------------------------------------------------
    # K-Means clustering
    # -----------------------------------------------------

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        embedding_matrix
    )

    # -----------------------------------------------------
    # Silhouette score
    # -----------------------------------------------------

    silhouette = silhouette_score(
        embedding_matrix,
        labels
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results.append(
        {
            "seed": seed,
            "initial_loss": loss_history[0],
            "loss_epoch_50": loss_history[49],
            "final_loss": loss_history[-1],
            "silhouette": float(silhouette),
        }
    )

    cluster_labels[seed] = labels

    print(
        f"Seed {seed} | "
        f"Initial loss={loss_history[0]:.6f} | "
        f"Epoch 50 loss={loss_history[49]:.6f} | "
        f"Final loss={loss_history[-1]:.6f} | "
        f"Silhouette={silhouette:.4f}"
    )


# ---------------------------------------------------------
# Compare model stability
# ---------------------------------------------------------

print()
print("=" * 60)
print("MODEL STABILITY RESULTS")
print("=" * 60)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)

print()

print(
    f"Mean silhouette: "
    f"{results_df['silhouette'].mean():.4f}"
)

print(
    f"Silhouette standard deviation: "
    f"{results_df['silhouette'].std():.4f}"
)

print()

print(
    f"Mean final loss: "
    f"{results_df['final_loss'].mean():.6f}"
)

print(
    f"Final loss standard deviation: "
    f"{results_df['final_loss'].std():.6f}"
)


# ---------------------------------------------------------
# Pairwise Adjusted Rand Index
# ---------------------------------------------------------

print()
print("PAIRWISE ADJUSTED RAND INDEX")
print("--------------------------------")

ari_results = []

for i in range(len(SEEDS)):

    for j in range(i + 1, len(SEEDS)):

        seed_a = SEEDS[i]
        seed_b = SEEDS[j]

        ari = adjusted_rand_score(
            cluster_labels[seed_a],
            cluster_labels[seed_b]
        )

        ari_results.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ari": ari,
            }
        )

        print(
            f"Seed {seed_a} vs "
            f"Seed {seed_b}: "
            f"ARI={ari:.4f}"
        )


# ---------------------------------------------------------
# ARI summary
# ---------------------------------------------------------

ari_df = pd.DataFrame(
    ari_results
)

print()

print(
    f"Mean pairwise ARI: "
    f"{ari_df['ari'].mean():.4f}"
)

print(
    f"Minimum pairwise ARI: "
    f"{ari_df['ari'].min():.4f}"
)

print(
    f"Maximum pairwise ARI: "
    f"{ari_df['ari'].max():.4f}"
)


# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------

print()
print("=" * 60)
print("MODEL STABILITY TESTING COMPLETED")
print("=" * 60)