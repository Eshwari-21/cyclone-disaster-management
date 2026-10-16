
from pathlib import Path

import pandas as pd

from src.risk_analysis.risk_agent import run_risk_analysis


def main():
    input_path = Path("data/processed/modeling_grid_2024.csv")
    output_path = Path(
        "data/processed/risk_analysis_results_2024.csv"
    )

    data = pd.read_csv(input_path)
    results = run_risk_analysis(data)

    # Save all original columns plus risk scores and categories.
    results.to_csv(output_path, index=False)

    print(f"Input grid cells: {len(data):,}")
    print(f"Output grid cells: {len(results):,}")
    print(f"Saved results to: {output_path}")

    print("\nRisk category counts:")
    print(results["risk_category"].value_counts().sort_index())

    print("\nTop 10 grid cells by risk score:")
    print(
        results.nlargest(10, "risk_score")[
            ["grid_id", "risk_score", "risk_category"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
