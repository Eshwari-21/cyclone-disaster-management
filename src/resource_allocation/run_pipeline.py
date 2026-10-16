
from pathlib import Path

import pandas as pd

from src.resource_allocation.demand_estimation import estimate_demand
from src.resource_allocation.allocation_agent import allocate_resources
from src.resource_allocation.resource_inventory import ResourceInventory
from src.resource_allocation.validation import (
    validate_input,
    validate_allocations,
)


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "risk_analysis_results_2024.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR / "resource_allocation_results_2024.csv"
)

# Water-only inventory: simulation configuration.
INVENTORY = ResourceInventory(
    food_kits=5000,
    water_units=10000,
    medical_kits=2000,
    rescue_teams=50,
)

RISK_WEIGHT = 0.5
POPULATION_WEIGHT = 0.5


# ==================================================
# PIPELINE
# ==================================================

def main():
    print("Loading risk analysis results...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Risk analysis results not found: {INPUT_FILE}"
        )

    risk_data = pd.read_csv(INPUT_FILE)

    print(f"Grid cells loaded: {len(risk_data):,}")

    # Validate existing risk-analysis inputs.
    # Demand is estimated immediately afterward, so validate
    # the source columns required for this step.
    required_columns = [
        "grid_id",
        "risk_score",
        "population_sum",
        "population_exposure_norm",
    ]

    missing = [
        column for column in required_columns
        if column not in risk_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing risk-analysis columns: {missing}"
        )

    print("Estimating water demand...")

    demand_data = estimate_demand(risk_data)

    validate_input(demand_data)

    print("Allocating water units...")

    allocation = allocate_resources(
        demand_data,
        inventory=INVENTORY,
        risk_weight=RISK_WEIGHT,
        population_weight=POPULATION_WEIGHT,
    )

    print("Validating water allocation...")

    validate_allocations(allocation, INVENTORY)

    # Keep only water-allocation outputs plus useful grid,
    # risk, and population fields for mapping and evaluation.
    output_columns = [
        column
        for column in [
            "grid_id",
            "risk_score",
            "risk_category",
            "population_sum",
            "population_exposure_norm",
            "estimated_affected_population",
            "water_units_demand",
            "water_units_allocated",
            "water_units_unmet_demand",
            "priority",
            "priority_rank",
        ]
        if column in allocation.columns
    ]

    water_results = allocation[output_columns].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    water_results.to_csv(OUTPUT_FILE, index=False)

    total_demand = int(water_results["water_units_demand"].sum())
    total_allocated = int(
        water_results["water_units_allocated"].sum()
    )
    total_unmet = int(
        water_results["water_units_unmet_demand"].sum()
    )

    print("\n========== WATER ALLOCATION SUMMARY ==========")
    print(f"Grid cells processed: {len(water_results):,}")
    print(f"Total water demand: {total_demand:,}")
    print(f"Water units available: {INVENTORY.water_units:,}")
    print(f"Water units allocated: {total_allocated:,}")
    print(f"Unmet water demand: {total_unmet:,}")

    coverage = (
        100 * total_allocated / total_demand
        if total_demand > 0
        else 0.0
    )

    print(f"Water demand coverage: {coverage:.3f}%")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nWater allocation pipeline completed successfully.")


if __name__ == "__main__":
    main()
