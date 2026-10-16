
"""Basic tests for the cyclone risk-analysis modules."""

import pandas as pd
import pytest

from src.risk_analysis.risk_scoring import (
    RiskWeights,
    calculate_risk_scores,
)
from src.risk_analysis.risk_agent import run_risk_analysis


def sample_data():
    return pd.DataFrame({
        "grid_id": ["grid_1", "grid_2", "grid_3"],
        "max_hazard": [0.1, 0.2, 0.3],
        "cumulative_hazard": [0.2, 0.4, 0.6],
        "population_sum": [100, 500, 1000],
    })


def test_scores_are_within_range():
    result = calculate_risk_scores(sample_data())
    assert result["risk_score"].between(0, 100).all()


def test_scores_increase_for_increasing_inputs():
    result = calculate_risk_scores(sample_data())
    assert result["risk_score"].is_monotonic_increasing


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        calculate_risk_scores(
            sample_data(),
            RiskWeights(0.5, 0.5, 0.5),
        )


def test_duplicate_grid_ids_are_rejected():
    data = sample_data()
    data.loc[1, "grid_id"] = data.loc[0, "grid_id"]

    with pytest.raises(ValueError):
        calculate_risk_scores(data)


def test_categories_are_assigned():
    result = run_risk_analysis(sample_data())

    assert result["risk_category"].notna().all()
    assert set(result["risk_category"].astype(str)).issubset({
        "Low", "Moderate", "High", "Very High"
    })
