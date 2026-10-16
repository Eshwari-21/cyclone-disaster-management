
"""Run the complete cyclone risk-analysis workflow."""

from pathlib import Path

import pandas as pd

from src.risk_analysis.risk_agent import run_risk_analysis
from src.risk_analysis.validation import validate_risk_results
from src.risk_analysis.create_risk_map import create_risk_map_data
from src.risk_analysis.render_risk_map import render_risk_map


def main():
    input_path = Path("data/processed/modeling_grid_2024.csv")
    cluster_path = Path(
        "data/processed/baseline_vs_gnn_clusters_2024.csv"
    )
    output_path = Path(
        "data/processed/risk_analysis_results_2024.csv"
    )

    # ---------------------------------------------------------
    # STEP 1: Load modeling data and GNN cluster assignments
    # ---------------------------------------------------------
    print("\nSTEP 1: Load modeling data and GNN clusters")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Modeling data not found: {input_path}"
        )

    if not cluster_path.exists():
        raise FileNotFoundError(
            f"GNN cluster data not found: {cluster_path}"
        )

    data = pd.read_csv(input_path)
    clusters = pd.read_csv(cluster_path)

    print(f"Modeling grid cells: {len(data):,}")
    print(f"GNN cluster records: {len(clusters):,}")

    # ---------------------------------------------------------
    # STEP 2: Calculate risk scores and attach GNN profiles
    # ---------------------------------------------------------
    print("\nSTEP 2: Calculate risk scores and attach GNN clusters")

    required_cluster_columns = {"grid_id", "gnn_cluster"}
    missing_cluster_columns = (
        required_cluster_columns - set(clusters.columns)
    )

    if missing_cluster_columns:
        raise ValueError(
            "GNN cluster file is missing required columns: "
            f"{sorted(missing_cluster_columns)}. "
            f"Available columns: {clusters.columns.tolist()}"
        )

    results = run_risk_analysis(
        data,
        cluster_data=clusters[["grid_id", "gnn_cluster"]],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Saved {len(results):,} grid cells to {output_path}")

    if "gnn_cluster" in results.columns:
        print("\nGNN cluster counts:")
        print(results["gnn_cluster"].value_counts().sort_index())

    if "gnn_profile" in results.columns:
        print("\nGNN spatial profiles:")
        print(results["gnn_profile"].value_counts())

    # ---------------------------------------------------------
    # STEP 3: Validate risk-analysis results
    # ---------------------------------------------------------
    print("\nSTEP 3: Validate risk-analysis results")

    report = validate_risk_results(results)

    for check, value in report.items():
        print(f"{check}: {value}")

    required_checks = [
        report["total_rows"] == report["unique_grid_ids"],
        report["missing_grid_ids"] == 0,
        report["duplicate_grid_ids"] == 0,
        report["missing_scores"] == 0,
        report["finite_scores"],
        report["scores_within_0_100"],
        report["missing_categories"] == 0,
        report["components_within_0_1"],
    ]

    # Validate GNN assignments as well.
    if "gnn_cluster" in results.columns:
        required_checks.extend([
            results["gnn_cluster"].notna().all(),
            results["gnn_profile"].notna().all(),
        ])

    if not all(required_checks):
        raise ValueError(
            "Validation failed. Pipeline stopped before map export."
        )

    print("\nAll risk-analysis validation checks passed.")

    # ---------------------------------------------------------
    # STEP 4: Create geographic risk data
    # ---------------------------------------------------------
    print("\nSTEP 4: Create geographic risk data")

    risk_map = create_risk_map_data()

    print(f"Mapped grid cells: {len(risk_map):,}")

    # ---------------------------------------------------------
    # STEP 5: Render risk map
    # ---------------------------------------------------------
    print("\nSTEP 5: Render risk map")

    render_risk_map()

    # ---------------------------------------------------------
    # STEP 6: Display final summary
    # ---------------------------------------------------------
    print("\nRISK ANALYSIS SUMMARY")
    print("-" * 45)

    print(f"Total grid cells: {len(results):,}")
    print(
        f"Mean risk score: {results['risk_score'].mean():.2f}"
    )
    print(
        f"Minimum risk score: {results['risk_score'].min():.2f}"
    )
    print(
        f"Maximum risk score: {results['risk_score'].max():.2f}"
    )

    print("\nRisk category counts:")
    print(results["risk_category"].value_counts().sort_index())

    print("\nRisk-analysis pipeline completed successfully.")
    print(f"Results CSV: {output_path}")
    print("Geographic output: data/processed/risk_map_2024.gpkg")
    print("Map image: outputs/risk_map_2024.png")


if __name__ == "__main__":
    main()
