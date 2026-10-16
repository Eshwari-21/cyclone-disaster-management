
"""Validation checks for cyclone risk-analysis results."""

import numpy as np
import pandas as pd


def validate_risk_results(data: pd.DataFrame) -> dict:
    """Check risk scores, categories, and grid identifiers."""

    required_columns = [
        "grid_id",
        "risk_score",
        "risk_category",
        "max_hazard_norm",
        "cumulative_hazard_norm",
        "population_exposure_norm",
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    scores = pd.to_numeric(
        data["risk_score"], errors="raise"
    )

    checks = {
        "total_rows": len(data),
        "unique_grid_ids": data["grid_id"].nunique(),
        "missing_grid_ids": int(data["grid_id"].isna().sum()),
        "duplicate_grid_ids": int(data["grid_id"].duplicated().sum()),
        "missing_scores": int(scores.isna().sum()),
        "finite_scores": bool(np.isfinite(scores).all()),
        "scores_within_0_100": bool(
            scores.between(0, 100).all()
        ),
        "missing_categories": int(
            data["risk_category"].isna().sum()
        ),
        "components_within_0_1": bool(
            data[
                [
                    "max_hazard_norm",
                    "cumulative_hazard_norm",
                    "population_exposure_norm",
                ]
            ].apply(
                lambda column: (
                    column.between(0, 1).all()
                )
            ).all()
        ),
    }

    return checks


if __name__ == "__main__":
    results_path = (
        "data/processed/risk_analysis_results_2024.csv"
    )

    results = pd.read_csv(results_path)
    report = validate_risk_results(results)

    print("RISK ANALYSIS VALIDATION REPORT")
    print("-" * 40)

    for check, value in report.items():
        print(f"{check}: {value}")
