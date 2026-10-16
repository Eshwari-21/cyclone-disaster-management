
"""Calculate transparent relative cyclone risk-screening scores."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class RiskWeights:
    """Weights used to calculate the relative risk score."""

    max_hazard: float = 0.30
    cumulative_hazard: float = 0.40
    population_exposure: float = 0.30

    def validate(self):
        weights = [
            self.max_hazard,
            self.cumulative_hazard,
            self.population_exposure,
        ]

        if not all(np.isfinite(weights)):
            raise ValueError("Weights must be finite.")

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative.")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0.")


def calculate_risk_scores(
    data: pd.DataFrame,
    weights: RiskWeights = RiskWeights(),
) -> pd.DataFrame:
    """
    Calculate a relative screening score from 0 to 100.

    This score is not a probability of cyclone damage.
    Normalization is relative to the supplied study-area dataset.
    """

    weights.validate()

    required_columns = [
        "grid_id",
        "max_hazard",
        "cumulative_hazard",
        "population_sum",
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        raise ValueError("Input dataset is empty.")

    if data["grid_id"].isna().any():
        raise ValueError("grid_id contains missing values.")

    if data["grid_id"].duplicated().any():
        raise ValueError("Duplicate grid_id values found.")

    result = data.copy()

    numeric_columns = [
        "max_hazard",
        "cumulative_hazard",
        "population_sum",
    ]

    for col in numeric_columns:
        result[col] = pd.to_numeric(
            result[col], errors="raise"
        )

        values = result[col].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{col} contains missing or non-finite values."
            )

        if (result[col] < 0).any():
            raise ValueError(f"{col} cannot contain negatives.")

    # Normalize the two hazard indicators to a 0–1 scale.
    scaler = MinMaxScaler()

    normalized_hazards = scaler.fit_transform(
        result[["max_hazard", "cumulative_hazard"]]
    )

    result["max_hazard_norm"] = normalized_hazards[:, 0]
    result["cumulative_hazard_norm"] = normalized_hazards[:, 1]

    # Log-transform population before normalization.
    log_population = np.log1p(
        result["population_sum"].to_numpy(dtype=float)
    )

    pop_min = log_population.min()
    pop_max = log_population.max()

    if np.isclose(pop_min, pop_max):
        result["population_exposure_norm"] = 0.0
    else:
        result["population_exposure_norm"] = (
            (log_population - pop_min) / (pop_max - pop_min)
        )

    # Weighted composite score.
    result["risk_score"] = 100 * (
        weights.max_hazard * result["max_hazard_norm"]
        + weights.cumulative_hazard
        * result["cumulative_hazard_norm"]
        + weights.population_exposure
        * result["population_exposure_norm"]
    )

    return result
