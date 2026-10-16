
"""
Evaluate GNN clustering stability across random seeds.

Outputs:
    outputs/gnn_stability_seed_metrics.csv
    outputs/gnn_stability_pairwise_ari.csv
    outputs/gnn_stability_pairwise_cka.csv
    outputs/gnn_stability_pairwise_comparison.csv
    outputs/gnn_embeddings_seed_<seed>.csv
    outputs/gnn_stability_embeddings_all_seeds.csv

Metrics:
    - Silhouette score
    - Reconstruction loss
    - Cluster sizes
    - Pairwise Adjusted Rand Index (ARI)
    - Pairwise centered linear CKA

Run from the repository root:
    python src/risk_analysis/evaluate_gnn_stability.py
"""

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from libpysal.weights import Queen
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODELING_PATH = ROOT_DIR / "data" / "processed" / "modeling_grid_2024.csv"
GRID_PATH = ROOT_DIR / "data" / "processed" / "coastal_ap_grid_1km.gpkg"

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
    "log_population",
    "max_hazard",
    "cumulative_hazard",
]

EMBEDDING_DIM = 8
N_CLUSTERS = 2
EPOCHS = 100
LEARNING_RATE = 0.001

SEEDS = [7, 21, 42, 84, 123, 202, 303, 404, 505, 606]

# Keep silhouette calculations manageable on a large dataset.
SILHOUETTE_SAMPLE_SIZE = 5000


# ============================================================
# 2. Graph convolution model
# ============================================================

class GraphConvolution(nn.Module):
    """One graph-convolution layer using a normalized adjacency matrix."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        # Transform node features, then aggregate neighboring information.
        transformed = self.linear(x)
        return torch.sparse.mm(adjacency, transformed)


class SpatialAutoencoder(nn.Module):
    """
    Small graph autoencoder.

    Encoder:
        Node features -> graph convolution -> embeddings

    Decoder:
        Embeddings -> reconstructed node features
    """

    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        self.encoder = GraphConvolution(input_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, input_dim)

    def encode(self, x, adjacency):
        return F.relu(self.encoder(x, adjacency))

    def forward(self, x, adjacency):
        embeddings = self.encode(x, adjacency)
        reconstruction = self.decoder(embeddings)
        return reconstruction, embeddings


# ============================================================
# 3. Data loading and alignment
# ============================================================

def load_and_prepare_data():
    """Load modeling data and spatial geometries in matching row order."""

    if not MODELING_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {MODELING_PATH}"
        )

    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"Spatial grid not found: {GRID_PATH}"
        )

    modeling_df = pd.read_csv(MODELING_PATH)
    grid_gdf = gpd.read_file(GRID_PATH)

    required_model_columns = ["grid_id"] + FEATURE_COLUMNS
    missing_model = [
        col for col in required_model_columns
        if col not in modeling_df.columns
    ]

    if missing_model:
        raise ValueError(
            f"Missing columns in modeling CSV: {missing_model}"
        )

    if "grid_id" not in grid_gdf.columns:
        raise ValueError(
            "The spatial grid GeoPackage must contain a 'grid_id' column."
        )

    if modeling_df["grid_id"].isna().any():
        raise ValueError("Modeling data contains missing grid_id values.")

    if grid_gdf["grid_id"].isna().any():
        raise ValueError("Spatial grid contains missing grid_id values.")

    if modeling_df["grid_id"].duplicated().any():
        raise ValueError("Modeling data contains duplicate grid_id values.")

    if grid_gdf["grid_id"].duplicated().any():
        raise ValueError("Spatial grid contains duplicate grid_id values.")

    # Compare IDs as strings to avoid int/string representation mismatches.
    modeling_df = modeling_df.copy()
    grid_gdf = grid_gdf.copy()

    modeling_df["_grid_key"] = modeling_df["grid_id"].astype(str)
    grid_gdf["_grid_key"] = grid_gdf["grid_id"].astype(str)

    model_ids = set(modeling_df["_grid_key"])
    spatial_ids = set(grid_gdf["_grid_key"])

    if model_ids != spatial_ids:
        missing_geometry = list(model_ids - spatial_ids)[:10]
        missing_modeling = list(spatial_ids - model_ids)[:10]

        raise ValueError(
            "Modeling and spatial grid IDs do not match.\n"
            f"IDs missing from spatial grid (first 10): "
            f"{missing_geometry}\n"
            f"IDs missing from modeling CSV (first 10): "
            f"{missing_modeling}"
        )

    # Reorder geometries to exactly match modeling CSV row order.
    grid_gdf = (
        grid_gdf
        .set_index("_grid_key")
        .loc[modeling_df["_grid_key"].tolist()]
        .reset_index()
    )

    X = modeling_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError(
            "Modeling features contain NaN or infinite values. "
            "Resolve them before training."
        )

    # Standardize features once, consistently across all seed runs.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    print(f"Loaded {len(modeling_df):,} grid cells.")
    print(f"Using {len(FEATURE_COLUMNS)} input features.")
    print(f"Spatial CRS: {grid_gdf.crs}")

    return modeling_df, grid_gdf, X_scaled


# ============================================================
# 4. Build normalized spatial adjacency
# ============================================================

def build_normalized_adjacency(grid_gdf):
    """
    Build Queen-contiguity adjacency with self-loops and symmetric
    degree normalization:

        A_hat = D^(-1/2) (A + I) D^(-1/2)
    """

    print("\nBuilding Queen-contiguity spatial weights...")

    try:
        weights = Queen.from_dataframe(grid_gdf)
    except Exception as exc:
        raise RuntimeError(
            "Could not build Queen-contiguity weights. "
            "Check the spatial grid geometries and libpysal installation."
        ) from exc

    adjacency = weights.sparse.tocsr().astype(np.float32)

    # Make the adjacency symmetric and add self-loops.
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(1.0)
    adjacency.eliminate_zeros()

    degrees = np.asarray(adjacency.sum(axis=1)).ravel()

    if np.any(degrees <= 0):
        raise ValueError(
            "At least one grid cell has zero degree after adding self-loops."
        )

    inv_sqrt_degree = 1.0 / np.sqrt(degrees)
    degree_matrix = sparse.diags(inv_sqrt_degree)

    normalized = degree_matrix @ adjacency @ degree_matrix
    normalized = normalized.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((normalized.row, normalized.col)).astype(np.int64)
    )
    values = torch.from_numpy(normalized.data)

    adjacency_tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size=normalized.shape,
        dtype=torch.float32,
    ).coalesce()

    print(
        f"Spatial graph built: {adjacency_tensor.shape[0]:,} nodes, "
        f"{adjacency_tensor._nnz():,} adjacency entries."
    )

    return adjacency_tensor


# ============================================================
# 5. Train one seed and obtain embeddings
# ============================================================

def run_single_seed(X_scaled, adjacency, seed):
    """Train the model for one seed and return metrics and embeddings."""

    print(f"\nTraining seed {seed}...")

    # Reproducible model initialization for each seed.
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tensor = torch.from_numpy(X_scaled)

    model = SpatialAutoencoder(
        input_dim=X_scaled.shape[1],
        embedding_dim=EMBEDDING_DIM,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        reconstruction, _ = model(X_tensor, adjacency)
        loss = F.mse_loss(reconstruction, X_tensor)

        loss.backward()
        optimizer.step()

    # Evaluate final reconstruction and obtain embeddings.
    model.eval()

    with torch.no_grad():
        reconstruction, embeddings_tensor = model(X_tensor, adjacency)
        final_loss = F.mse_loss(
            reconstruction, X_tensor
        ).item()

        embeddings = embeddings_tensor.cpu().numpy().astype(np.float64)

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"Seed {seed} produced NaN or infinite embeddings."
        )

    # K-Means is held fixed across runs so this experiment focuses on
    # variation introduced by GNN initialization/training.
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(embeddings)

    if len(np.unique(labels)) < 2:
        silhouette = float("nan")
        warnings.warn(
            f"Seed {seed} produced fewer than two non-empty clusters."
        )
    else:
        sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(labels))

        silhouette = silhouette_score(
            embeddings,
            labels,
            sample_size=sample_size,
            random_state=42,
        )

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts)
    }

    metrics = {
        "seed": seed,
        "silhouette_score": float(silhouette),
        "reconstruction_loss": float(final_loss),
        "cluster_0_size": cluster_sizes.get(0, 0),
        "cluster_1_size": cluster_sizes.get(1, 0),
    }

    print(
        f"Seed {seed}: silhouette={silhouette:.4f}, "
        f"loss={final_loss:.6f}, "
        f"cluster sizes={cluster_sizes}"
    )

    return metrics, labels, embeddings


# ============================================================
# 6. Linear centered Kernel Alignment (CKA)
# ============================================================

def linear_cka(X, Y):
    """
    Calculate centered linear CKA between two embedding matrices.

    Both matrices must contain the same observations in the same row
    order. Each matrix is centered column-wise.

    CKA is invariant to orthogonal transformations and isotropic scaling.
    It is a representation-similarity measure, not a clustering score.
    """

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("CKA inputs must be two-dimensional matrices.")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "CKA inputs must contain the same number of observations."
        )

    # Center each embedding dimension across observations.
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Linear-kernel CKA computed efficiently in feature space.
    cross = X.T @ Y
    x_gram = X.T @ X
    y_gram = Y.T @ Y

    numerator = np.sum(cross ** 2)
    denominator = np.sqrt(
        np.sum(x_gram ** 2) * np.sum(y_gram ** 2)
    )

    if denominator <= np.finfo(np.float64).eps:
        return float("nan")

    return float(numerator / denominator)


# ============================================================
# 7. Pairwise ARI and CKA
# ============================================================

def calculate_pairwise_comparisons(labels_by_seed, embeddings_by_seed):
    """
    Calculate pairwise ARI and linear CKA for all distinct seed pairs.

    Pair order is consistent across all output files.
    """

    seeds = list(labels_by_seed.keys())

    comparison_rows = []

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            seed_a = seeds[i]
            seed_b = seeds[j]

            ari = adjusted_rand_score(
                labels_by_seed[seed_a],
                labels_by_seed[seed_b],
            )

            cka = linear_cka(
                embeddings_by_seed[seed_a],
                embeddings_by_seed[seed_b],
            )

            comparison_rows.append({
                "seed_1": seed_a,
                "seed_2": seed_b,
                "pairwise_ari": float(ari),
                "linear_cka": float(cka),
            })

    comparisons_df = pd.DataFrame(comparison_rows)

    # Save combined comparison table.
    comparisons_df.to_csv(
        OUTPUT_DIR / "gnn_stability_pairwise_comparison.csv",
        index=False,
    )

    # Save ARI separately for compatibility with the previous script.
    comparisons_df[
        ["seed_1", "seed_2", "pairwise_ari"]
    ].to_csv(
        OUTPUT_DIR / "gnn_stability_pairwise_ari.csv",
        index=False,
    )

    # Save CKA separately.
    comparisons_df[
        ["seed_1", "seed_2", "linear_cka"]
    ].to_csv(
        OUTPUT_DIR / "gnn_stability_pairwise_cka.csv",
        index=False,
    )

    return comparisons_df


# ============================================================
# 8. Main experiment
# ============================================================

def main():
    print("=" * 65)
    print("GNN STABILITY EVALUATION: ARI + LINEAR CKA")
    print("=" * 65)

    modeling_df, grid_gdf, X_scaled = load_and_prepare_data()
    adjacency = build_normalized_adjacency(grid_gdf)

    all_metrics = []
    labels_by_seed = {}
    embeddings_by_seed = {}
    all_embedding_frames = []

    grid_ids = modeling_df["grid_id"].to_numpy()

    for seed in SEEDS:
        metrics, labels, embeddings = run_single_seed(
            X_scaled=X_scaled,
            adjacency=adjacency,
            seed=seed,
        )

        all_metrics.append(metrics)
        labels_by_seed[seed] = labels
        embeddings_by_seed[seed] = embeddings

        # Save the embeddings for this seed, including its cluster labels.
        embedding_columns = [
            f"embedding_{i}" for i in range(EMBEDDING_DIM)
        ]

        seed_embeddings_df = pd.DataFrame(
            embeddings,
            columns=embedding_columns,
        )

        seed_embeddings_df.insert(0, "grid_id", grid_ids)
        seed_embeddings_df["gnn_cluster"] = labels
        seed_embeddings_df["seed"] = seed

        seed_output_path = (
            OUTPUT_DIR / f"gnn_embeddings_seed_{seed}.csv"
        )

        seed_embeddings_df.to_csv(seed_output_path, index=False)

        print(f"Saved embeddings: {seed_output_path}")

        # Also create a long-format table containing every seed's
        # embedding for every grid cell.
        all_embedding_frames.append(seed_embeddings_df)

    # Save per-seed metrics.
    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = OUTPUT_DIR / "gnn_stability_seed_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save all seed embeddings in one long-format CSV.
    all_embeddings_df = pd.concat(
        all_embedding_frames,
        ignore_index=True,
    )

    all_embeddings_path = (
        OUTPUT_DIR / "gnn_stability_embeddings_all_seeds.csv"
    )

    all_embeddings_df.to_csv(
        all_embeddings_path,
        index=False,
    )

    # Calculate all pairwise ARI and CKA comparisons.
    comparisons_df = calculate_pairwise_comparisons(
        labels_by_seed=labels_by_seed,
        embeddings_by_seed=embeddings_by_seed,
    )

    # ========================================================
    # 9. Summary
    # ========================================================

    print("\n" + "=" * 65)
    print("SEED METRICS")
    print("=" * 65)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("PAIRWISE COMPARISON SUMMARY")
    print("=" * 65)

    print(
        f"Number of seed pairs: {len(comparisons_df)}"
    )

    print(
        "Mean pairwise ARI: "
        f"{comparisons_df['pairwise_ari'].mean():.4f}"
    )
    print(
        "Median pairwise ARI: "
        f"{comparisons_df['pairwise_ari'].median():.4f}"
    )
    print(
        "Minimum pairwise ARI: "
        f"{comparisons_df['pairwise_ari'].min():.4f}"
    )
    print(
        "Maximum pairwise ARI: "
        f"{comparisons_df['pairwise_ari'].max():.4f}"
    )

    print(
        "Mean pairwise linear CKA: "
        f"{comparisons_df['linear_cka'].mean():.4f}"
    )
    print(
        "Median pairwise linear CKA: "
        f"{comparisons_df['linear_cka'].median():.4f}"
    )
    print(
        "Minimum pairwise linear CKA: "
        f"{comparisons_df['linear_cka'].min():.4f}"
    )
    print(
        "Maximum pairwise linear CKA: "
        f"{comparisons_df['linear_cka'].max():.4f}"
    )

    print("\n" + "=" * 65)
    print("OUTPUT FILES")
    print("=" * 65)

    output_files = [
        metrics_path,
        OUTPUT_DIR / "gnn_stability_pairwise_ari.csv",
        OUTPUT_DIR / "gnn_stability_pairwise_cka.csv",
        OUTPUT_DIR / "gnn_stability_pairwise_comparison.csv",
        all_embeddings_path,
    ]

    for path in output_files:
        print(path)

    print(
        f"Individual embedding CSVs: "
        f"{len(SEEDS)} files in {OUTPUT_DIR}"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
