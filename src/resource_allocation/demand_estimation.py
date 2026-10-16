
import math

import pandas as pd


# Simulation assumptions, not official disaster forecasts.
DEFAULT_ASSUMPTIONS = {
    "affected_population_rate": 0.10,
    "water_units_person_day": 3,
    "water_days": 3,
}


def estimate_demand(
    data: pd.DataFrame,
    assumptions: dict | None = None,
) -> pd.DataFrame:
    """
    Estimate water demand for each geographic grid.

    Each water unit is assumed to provide one person with
    the specified number of water units per day.
    """

    assumptions = {
        **DEFAULT_ASSUMPTIONS,
        **(assumptions or {}),
    }

    required_columns = ["grid_id", "population_sum"]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if data.empty:
        raise ValueError("Input data cannot be empty.")

    result = data.copy()

    if result["grid_id"].isna().any():
        raise ValueError("grid_id contains missing values.")

    if result["grid_id"].duplicated().any():
        raise ValueError("grid_id contains duplicate values.")

    result["population_sum"] = pd.to_numeric(
        result["population_sum"],
        errors="raise",
    )

    if (
        result["population_sum"].isna().any()
        or not result["population_sum"].map(
            math.isfinite
        ).all()
        or (result["population_sum"] < 0).any()
    ):
        raise ValueError(
            "Population must contain finite, nonnegative values."
        )

    affected_rate = assumptions["affected_population_rate"]
    units_per_person_day = assumptions["water_units_person_day"]
    water_days = assumptions["water_days"]

    if not 0 <= affected_rate <= 1:
        raise ValueError(
            "affected_population_rate must be between 0 and 1."
        )

    if (
        not isinstance(units_per_person_day, (int, float))
        or isinstance(units_per_person_day, bool)
        or not math.isfinite(units_per_person_day)
        or units_per_person_day <= 0
    ):
        raise ValueError(
            "water_units_person_day must be a positive number."
        )

    if (
        not isinstance(water_days, (int, float))
        or isinstance(water_days, bool)
        or not math.isfinite(water_days)
        or water_days <= 0
    ):
        raise ValueError("water_days must be positive.")

    # Estimate the population affected by the cyclone.
    result["estimated_affected_population"] = (
        result["population_sum"] * affected_rate
    ).apply(math.ceil)

    # Estimate water demand per grid.
    result["water_units_demand"] = (
        result["estimated_affected_population"]
        * units_per_person_day
        * water_days
    ).apply(math.ceil)

    print("Water demand estimation completed.")
    print(
        "Estimated affected population:",
        int(result["estimated_affected_population"].sum()),
    )
    print(
        "Total estimated water demand:",
        int(result["water_units_demand"].sum()),
    )

    return result
