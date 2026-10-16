
from pathlib import Path

import pandas as pd

from src.resource_allocation.demand_estimation import estimate_demand
from src.resource_allocation.allocation_agent import allocate_resources
from src.resource_allocation.resource_inventory import ResourceInventory
from src.resource_allocation.validation import validate_allocations


# ==================================================
# CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "risk_analysis_results_2024.csv"
)

OUTPUT_DIR = ROOT_DIR / "outputs"

INVENTORY = ResourceInventory(
    food_kits=5000,
    water_units=10000,
    medical_kits=2000,
    rescue_teams=50,
)


# ==================================================
# WATER METRICS
# ==================================================

def calculate_water_metrics(
    allocation: pd.DataFrame,
    inventory: ResourceInventory,
) -> pd.DataFrame:
    """Calculate water demand, allocation, and utilization metrics."""

    total_demand = int(allocation["water_units_demand"].sum())
    total_allocated = int(
        allocation["water_units_allocated"].sum()
    )
    total_unmet = int(
        allocation["water_units_unmet_demand"].sum()
    )

    available = inventory.water_units

    if total_allocated > available:
        raise ValueError(
            f"Water allocation ({total_allocated}) exceeds "
            f"inventory ({available})."
        )

    if total_allocated + total_unmet != total_demand:
        raise ValueError(
            "Water demand reconciliation failed."
        )

    return pd.DataFrame([{
        "grid_count": len(allocation),
        "water_inventory_available": available,
        "total_water_demand": total_demand,
        "total_water_allocated": total_allocated,
        "total_unmet_water_demand": total_unmet,
        "demand_coverage_pct": (
            100.0 * total_allocated / total_demand
            if total_demand > 0 else 0.0
        ),
        "inventory_utilization_pct": (
            100.0 * total_allocated / available
            if available > 0 else 0.0
        ),
        "grids_receiving_water": int(
            (allocation["water_units_allocated"] > 0).sum()
        ),
        "grids_with_unmet_water_demand": int(
            (allocation["water_units_unmet_demand"] > 0).sum()
        ),
    }])


# ==================================================
# PRIORITY METRICS
# ==================================================

def calculate_priority_metrics(
    allocation: pd.DataFrame,
) -> pd.DataFrame:
    """Measure water reaching the top 10% of priority grids."""

    data = allocation.copy()

    if "priority" not in data.columns:
        data["priority"] = (
            0.5 * (data["risk_score"] / 100.0)
            + 0.5 * data["population_exposure_norm"]
        )

    data = data.sort_values(
        "priority",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    top_count = max(1, int(len(data) * 0.10))
    top_grids = data.head(top_count)

    total_allocated = int(data["water_units_allocated"].sum())
    top_allocated = int(top_grids["water_units_allocated"].sum())

    return pd.DataFrame([{
        "total_grid_count": len(data),
        "top_priority_grid_count": top_count,
        "total_water_allocated": total_allocated,
        "water_allocated_to_top_10_pct_grids": top_allocated,
        "allocation_share_to_top_10_pct_pct": (
            100.0 * top_allocated / total_allocated
            if total_allocated > 0 else 0.0
        ),
    }])


# ==================================================
# RISK CATEGORY METRICS
# ==================================================

def calculate_risk_category_metrics(
    allocation: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize water allocation by risk category."""

    if "risk_category" not in allocation.columns:
        raise ValueError(
            "Input data does not contain risk_category."
        )

    result = (
        allocation.groupby("risk_category", dropna=False)
        .agg(
            grid_count=("grid_id", "count"),
            water_demand=("water_units_demand", "sum"),
            water_allocated=("water_units_allocated", "sum"),
            unmet_water_demand=("water_units_unmet_demand", "sum"),
        )
        .reset_index()
    )

    result["water_demand_coverage_pct"] = (
        100.0 * result["water_allocated"]
        / result["water_demand"].replace(0, float("nan"))
    ).fillna(0.0)

    return result


# ==================================================
# STRATEGY COMPARISON
# ==================================================

def calculate_strategy_metrics(
    demand_data: pd.DataFrame,
    inventory: ResourceInventory,
) -> pd.DataFrame:
    """Compare water allocation under three priority strategies."""

    strategies = {
        "balanced": (0.5, 0.5),
        "risk_first": (1.0, 0.0),
        "population_first": (0.0, 1.0),
    }

    rows = []

    for strategy_name, weights in strategies.items():
        risk_weight, population_weight = weights

        result = allocate_resources(
            demand_data.copy(),
            inventory=inventory,
            risk_weight=risk_weight,
            population_weight=population_weight,
        )

        validate_allocations(result, inventory)

        total_demand = int(result["water_units_demand"].sum())
        total_allocated = int(
            result["water_units_allocated"].sum()
        )

        water_receivers = result[
            result["water_units_allocated"] > 0
        ]

        high_risk_mask = result["risk_category"].isin(
            ["High", "Very High"]
        )

        high_risk_water = int(
            result.loc[
                high_risk_mask,
                "water_units_allocated",
            ].sum()
        )

        rows.append({
            "strategy": strategy_name,
            "risk_weight": risk_weight,
            "population_weight": population_weight,
            "total_water_demand": total_demand,
            "total_water_allocated": total_allocated,
            "unmet_water_demand": (
                total_demand - total_allocated
            ),
            "demand_coverage_pct": (
                100.0 * total_allocated / total_demand
                if total_demand > 0 else 0.0
            ),
            "water_to_high_and_very_high_risk_grids": (
                high_risk_water
            ),
            "high_risk_allocation_share_pct": (
                100.0 * high_risk_water / total_allocated
                if total_allocated > 0 else 0.0
            ),
            "mean_risk_score_receiving_water": (
                float(water_receivers["risk_score"].mean())
                if not water_receivers.empty else 0.0
            ),
        })

    return pd.DataFrame(rows)


# ==================================================
# MAIN
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Risk results file not found: {INPUT_FILE}"
        )

    print("Loading risk analysis results...")
    risk_data = pd.read_csv(INPUT_FILE)

    print(f"Grid cells loaded: {len(risk_data):,}")

    print("Estimating water demand...")
    demand_data = estimate_demand(risk_data)

    print("Allocating water using balanced strategy...")
    allocation = allocate_resources(
        demand_data.copy(),
        inventory=INVENTORY,
        risk_weight=0.5,
        population_weight=0.5,
    )

    print("Validating water allocation...")
    validate_allocations(allocation, INVENTORY)

    print("Calculating water metrics...")
    water_metrics = calculate_water_metrics(
        allocation,
        INVENTORY,
    )

    print("Calculating priority metrics...")
    priority_metrics = calculate_priority_metrics(allocation)

    print("Calculating risk category metrics...")
    risk_metrics = calculate_risk_category_metrics(allocation)

    print("Comparing water allocation strategies...")
    strategy_metrics = calculate_strategy_metrics(
        demand_data,
        INVENTORY,
    )

    outputs = {
        "water_evaluation_metrics.csv": water_metrics,
        "water_priority_metrics.csv": priority_metrics,
        "water_risk_category_metrics.csv": risk_metrics,
        "water_strategy_comparison.csv": strategy_metrics,
    }

    for filename, dataframe in outputs.items():
        output_path = OUTPUT_DIR / filename
        dataframe.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    print("\n========== WATER EVALUATION ==========")
    print(water_metrics.round(3).to_string(index=False))

    print("\n========== WATER PRIORITY METRICS ==========")
    print(priority_metrics.round(3).to_string(index=False))

    print("\n========== WATER BY RISK CATEGORY ==========")
    print(risk_metrics.round(3).to_string(index=False))

    print("\n========== WATER STRATEGY COMPARISON ==========")
    print(strategy_metrics.round(3).to_string(index=False))

    print("\nWater-only evaluation completed successfully.")


if __name__ == "__main__":
    main()
