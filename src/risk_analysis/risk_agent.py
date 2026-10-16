
"""Risk Analysis Agent combining weighted scores and GNN profiles."""

import pandas as pd

from src.risk_analysis.risk_scoring import (
    RiskWeights,
    calculate_risk_scores,
)


def run_risk_analysis(
    data: pd.DataFrame,
    weights: RiskWeights = RiskWeights(),
    cluster_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate risk scores and optionally attach GNN clusters."""

    result = calculate_risk_scores(data, weights)

    result["risk_category"] = pd.cut(
        result["risk_score"],
        bins=[-float("inf"), 25, 50, 75, float("inf")],
        labels=["Low", "Moderate", "High", "Very High"],
        right=True,
        include_lowest=True,
    )

    if cluster_data is not None:
        required = {"grid_id", "gnn_cluster"}
        missing = required - set(cluster_data.columns)

        if missing:
            raise ValueError(
                f"Cluster data is missing columns: {sorted(missing)}"
            )

        clusters = cluster_data[["grid_id", "gnn_cluster"]].copy()

        if clusters["grid_id"].isna().any():
            raise ValueError("Cluster data contains missing grid IDs.")

        if clusters["grid_id"].duplicated().any():
            raise ValueError("Cluster data contains duplicate grid IDs.")

        if set(clusters["grid_id"]) != set(result["grid_id"]):
            raise ValueError(
                "Cluster IDs do not match the risk-analysis grid IDs."
            )

        result = result.merge(
            clusters,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

        # Descriptive profile, not a validated risk category.
        result["gnn_profile"] = result["gnn_cluster"].map({
            0: "Higher-average-hazard-and-exposure profile",
            1: "Lower-average-hazard-and-exposure profile",
        }).fillna("Unprofiled cluster")

    return result
