
import math

import pandas as pd

from src.resource_allocation.resource_inventory import ResourceInventory


def allocate_capped(
    total_units: int,
    demand: pd.Series,
    weights: pd.Series,
) -> pd.Series:
    """
    Allocate whole water units using proportional weights.

    Each grid's allocation is capped at its demand. When grids
    reach their caps, remaining units are redistributed among
    eligible grids.

    Uses deterministic largest-remainder rounding.
    """

    if (
        isinstance(total_units, bool)
        or not isinstance(total_units, int)
        or total_units < 0
    ):
        raise ValueError(
            "Available water units must be a nonnegative integer."
        )

    if len(demand) != len(weights):
        raise ValueError(
            "Demand and weights must have equal lengths."
        )

    demand_values = demand.to_numpy(dtype=float)
    weight_values = weights.to_numpy(dtype=float)

    if not all(math.isfinite(x) for x in demand_values):
        raise ValueError("Water demand must be finite.")

    if not all(math.isfinite(x) for x in weight_values):
        raise ValueError("Allocation weights must be finite.")

    if (demand_values < 0).any():
        raise ValueError("Water demand cannot be negative.")

    if (weight_values < 0).any():
        raise ValueError("Allocation weights cannot be negative.")

    if not all(x.is_integer() for x in demand_values):
        raise ValueError("Water demand must use whole units.")

    demand_values = demand_values.astype(int)
    allocations = [0] * len(demand_values)

    remaining = min(
        total_units,
        int(demand_values.sum()),
    )

    while remaining > 0:
        eligible = [
            i
            for i, amount in enumerate(demand_values)
            if allocations[i] < amount
        ]

        if not eligible:
            break

        eligible_weight_sum = sum(
            weight_values[i] for i in eligible
        )

        # If every eligible weight is zero, use equal weights.
        if eligible_weight_sum == 0:
            effective_weights = {
                i: 1.0 for i in eligible
            }
            eligible_weight_sum = float(len(eligible))
        else:
            effective_weights = {
                i: weight_values[i] for i in eligible
            }

        quotas = {
            i: remaining
            * effective_weights[i]
            / eligible_weight_sum
            for i in eligible
        }

        # Assign whole-unit quota floors, respecting demand caps.
        added = 0

        for i in eligible:
            available_demand = (
                demand_values[i] - allocations[i]
            )

            units = min(
                available_demand,
                math.floor(quotas[i]),
            )

            allocations[i] += units
            added += units

        remaining -= added

        if remaining == 0:
            break

        # Recalculate proportions if any grid has reached its cap.
        hit_cap = any(
            allocations[i] >= demand_values[i]
            for i in eligible
        )

        if hit_cap:
            continue

        # Assign leftover units by largest fractional remainder.
        eligible.sort(
            key=lambda i: (
                -(quotas[i] - math.floor(quotas[i])),
                i,
            )
        )

        distributed = 0

        for i in eligible:
            if remaining == 0:
                break

            if allocations[i] < demand_values[i]:
                allocations[i] += 1
                remaining -= 1
                distributed += 1

        if distributed == 0:
            break

    return pd.Series(
        allocations,
        index=demand.index,
        dtype=int,
    )


def allocate_resources(
    data: pd.DataFrame,
    inventory: ResourceInventory | None = None,
    risk_weight: float = 0.5,
    population_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Allocate water using risk and population exposure.

    Default priority weights:
        Risk: 50%
        Population exposure: 50%

    Allocation weight:
        priority * water demand

    Water allocation is proportional to this weight, subject
    to each grid's demand cap and available inventory.
    """

    if inventory is None:
        inventory = ResourceInventory()

    if (
        not math.isfinite(risk_weight)
        or not math.isfinite(population_weight)
    ):
        raise ValueError("Priority weights must be finite.")

    if risk_weight < 0 or population_weight < 0:
        raise ValueError("Priority weights cannot be negative.")

    if not math.isclose(
        risk_weight + population_weight,
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("Priority weights must sum to 1.")

    required_columns = [
        "grid_id",
        "risk_score",
        "population_exposure_norm",
        "water_units_demand",
    ]

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

    numeric_columns = [
        "risk_score",
        "population_exposure_norm",
        "water_units_demand",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

        if (
            result[column].isna().any()
            or not result[column].map(math.isfinite).all()
        ):
            raise ValueError(
                f"{column} must contain finite numeric values."
            )

        if (result[column] < 0).any():
            raise ValueError(
                f"{column} cannot contain negative values."
            )

    if (result["risk_score"] > 100).any():
        raise ValueError("risk_score cannot exceed 100.")

    if (result["population_exposure_norm"] > 1).any():
        raise ValueError(
            "population_exposure_norm cannot exceed 1."
        )

    if not (
        result["water_units_demand"]
        == result["water_units_demand"].round()
    ).all():
        raise ValueError(
            "water_units_demand must contain whole units."
        )

    # Normalize risk to [0, 1] and combine with population exposure.
    result["priority"] = (
        risk_weight * (result["risk_score"] / 100.0)
        + population_weight
        * result["population_exposure_norm"]
    )

    result["priority_rank"] = (
        result["priority"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # NEW: combine priority with estimated demand.
    # Higher priority and greater demand increase allocation weight.
    result["allocation_weight"] = (
        result["priority"]
        * result["water_units_demand"]
    )

    result["water_units_allocated"] = allocate_capped(
        total_units=inventory.water_units,
        demand=result["water_units_demand"],
        weights=result["allocation_weight"],
    )

    result["water_units_unmet_demand"] = (
        result["water_units_demand"]
        - result["water_units_allocated"]
    )

    # Final consistency checks.
    allocated_total = int(
        result["water_units_allocated"].sum()
    )

    if allocated_total > inventory.water_units:
        raise ValueError(
            "Water allocation exceeds available inventory."
        )

    if (
        result["water_units_allocated"]
        > result["water_units_demand"]
    ).any():
        raise ValueError(
            "Water allocation exceeds grid-level demand."
        )

    if (result["water_units_allocated"] < 0).any():
        raise ValueError(
            "Water allocation cannot be negative."
        )

    if not (
        result["water_units_allocated"]
        + result["water_units_unmet_demand"]
        == result["water_units_demand"]
    ).all():
        raise ValueError(
            "Allocation and unmet demand do not reconcile."
        )

    return result
