
"""Evaluate K-Means stability on the original standardized features."""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/processed/modeling_grid_2024.csv"
OUTPUT_FILE = ROOT / "outputs/baseline_stability_metrics.csv"

FEATURES = [
    "elevation_m",
    "slope_deg",
    "annual_rainfall_mm",
    "mean_ndvi",
    "log_population",
    "max_hazard",
    "cumulative_hazard",
]

SEEDS = [7, 21, 42, 84, 123, 202, 303, 404, 505, 606]


def main():
    data = pd.read_csv(INPUT_FILE)

    X_raw = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X_raw.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or non-finite values.")

    X = StandardScaler().fit_transform(X_raw)

    labels_by_seed = {}
    rows = []

    for seed in SEEDS:
        model = KMeans(
            n_clusters=2,
            random_state=seed,
            n_init=10,
        )
        labels = model.fit_predict(X)
        labels_by_seed[seed] = labels

        silhouette = silhouette_score(
            X,
            labels,
            sample_size=min(5000, len(X)),
            random_state=42,
        )

        rows.append({
            "seed": seed,
            "silhouette_sampled": silhouette,
            "cluster_0_cells": int((labels == 0).sum()),
            "cluster_1_cells": int((labels == 1).sum()),
        })

    pairwise_ari = [
        adjusted_rand_score(labels_by_seed[a], labels_by_seed[b])
        for a, b in combinations(SEEDS, 2)
    ]

    metrics = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_FILE, index=False)

    print("\nBASELINE PER-SEED METRICS")
    print(metrics.round(4).to_string(index=False))

    print("\nBASELINE PAIRWISE ARI")
    print(f"Mean ARI: {np.mean(pairwise_ari):.4f}")
    print(f"Median ARI: {np.median(pairwise_ari):.4f}")
    print(f"Minimum ARI: {np.min(pairwise_ari):.4f}")
    print(f"Maximum ARI: {np.max(pairwise_ari):.4f}")

    print(f"\nSaved metrics to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
