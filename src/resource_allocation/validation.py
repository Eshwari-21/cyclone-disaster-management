
import math

import pandas as pd

from src.resource_allocation.resource_inventory import ResourceInventory


def validate_input(data: pd.DataFrame) -> None:
    """Validate input data required for water allocation."""

    required_columns = [
        "grid_id",
        "risk_score",
        "population_sum",
        "population_exposure_norm",
        "water_units_demand",
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required input columns: {missing}"
        )

    if data.empty:
        raise ValueError("Input data cannot be empty.")

    if data["grid_id"].isna().any():
        raise ValueError("grid_id contains missing values.")

    if data["grid_id"].duplicated().any():
        raise ValueError("grid_id contains duplicate values.")

    numeric_columns = [
        "risk_score",
        "population_sum",
        "population_exposure_norm",
        "water_units_demand",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="raise")

        if values.isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if not values.map(math.isfinite).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if (values < 0).any():
            raise ValueError(
                f"{column} cannot contain negative values."
            )

    if (data["risk_score"] > 100).any():
        raise ValueError("risk_score cannot exceed 100.")

    if (data["population_exposure_norm"] > 1).any():
        raise ValueError(
            "population_exposure_norm cannot exceed 1."
        )

    demand = pd.to_numeric(
        data["water_units_demand"],
        errors="raise",
    )

    if not (demand == demand.round()).all():
        raise ValueError(
            "water_units_demand must contain whole units."
        )


def validate_allocations(
    result: pd.DataFrame,
    inventory: ResourceInventory,
) -> None:
    """Validate water allocations, demand caps, and inventory."""

    required_columns = [
        "grid_id",
        "water_units_demand",
        "water_units_allocated",
        "water_units_unmet_demand",
    ]

    missing = [
        column for column in required_columns
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing allocation columns: {missing}"
        )

    if result.empty:
        raise ValueError("Allocation results cannot be empty.")

    if result["grid_id"].isna().any():
        raise ValueError("grid_id contains missing values.")

    if result["grid_id"].duplicated().any():
        raise ValueError("grid_id contains duplicate values.")

    numeric_columns = [
        "water_units_demand",
        "water_units_allocated",
        "water_units_unmet_demand",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="raise")

        if values.isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if not values.map(math.isfinite).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if (values < 0).any():
            raise ValueError(
                f"{column} cannot contain negative values."
            )

        if not (values == values.round()).all():
            raise ValueError(
                f"{column} must contain whole units."
            )

    # Allocated water must not exceed demand in any grid.
    if (
        result["water_units_allocated"]
        > result["water_units_demand"]
    ).any():
        raise ValueError(
            "Water allocation exceeds grid-level demand."
        )

    # Allocated water plus unmet demand must equal total demand.
    if not (
        result["water_units_allocated"]
        + result["water_units_unmet_demand"]
        == result["water_units_demand"]
    ).all():
        raise ValueError(
            "Water allocation and unmet demand do not reconcile."
        )

    # Total allocated water must not exceed inventory.
    total_allocated = int(
        result["water_units_allocated"].sum()
    )

    if total_allocated > inventory.water_units:
        raise ValueError(
            f"Water inventory exceeded: allocated {total_allocated}, "
            f"available {inventory.water_units}."
        )

    print(f"Grid cells validated: {len(result):,}")
    print(f"Water demand: {int(result['water_units_demand'].sum()):,}")
    print(f"Water allocated: {total_allocated:,}")
    print(
        "Unmet water demand:",
        int(result["water_units_unmet_demand"].sum()),
    )
    print("Inventory constraint: PASS")
    print("Demand cap: PASS")
    print("Unmet-demand reconciliation: PASS")
