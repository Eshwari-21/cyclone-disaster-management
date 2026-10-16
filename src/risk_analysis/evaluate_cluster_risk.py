
"""Evaluate how GNN clusters relate to risk categories and scores."""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/risk_analysis_results_2024.csv"
)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = pd.read_csv(INPUT_PATH)

    required = {"grid_id", "risk_score", "risk_category", "gnn_cluster"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Compare risk scores within each GNN cluster.
    score_summary = (
        data.groupby("gnn_cluster", observed=True)
        .agg(
            grid_cells=("grid_id", "count"),
            mean_risk_score=("risk_score", "mean"),
            median_risk_score=("risk_score", "median"),
            minimum_risk_score=("risk_score", "min"),
            maximum_risk_score=("risk_score", "max"),
        )
        .round(3)
    )

    # Count each risk category within each cluster.
    category_counts = pd.crosstab(
        data["gnn_cluster"],
        data["risk_category"],
    )

    # Convert counts to percentages within each cluster.
    category_percentages = (
        pd.crosstab(
            data["gnn_cluster"],
            data["risk_category"],
            normalize="index",
        )
        .mul(100)
        .round(2)
    )

    print("\nRISK SCORES BY GNN CLUSTER")
    print(score_summary.to_string())

    print("\nRISK CATEGORY COUNTS BY GNN CLUSTER")
    print(category_counts.to_string())

    print("\nRISK CATEGORY PERCENTAGES WITHIN EACH CLUSTER")
    print(category_percentages.to_string())

    score_summary.to_csv(
        OUTPUT_DIR / "gnn_cluster_risk_summary.csv"
    )
    category_counts.to_csv(
        OUTPUT_DIR / "gnn_cluster_risk_category_counts.csv"
    )
    category_percentages.to_csv(
        OUTPUT_DIR / "gnn_cluster_risk_category_percentages.csv"
    )

    print("\nEvaluation files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
