
from pathlib import Path

import pandas as pd

from src.resource_allocation.resource_inventory import ResourceInventory
from src.resource_allocation.validation import validate_allocations


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "resource_allocation_results_2024.csv"
)

INVENTORY = ResourceInventory(
    food_kits=5000,
    water_units=10000,
    medical_kits=2000,
    rescue_teams=50,
)


# ==================================================
# EVALUATION
# ==================================================

def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Allocation results not found: {RESULTS_FILE}"
        )

    print("Loading water allocation results...")
    results = pd.read_csv(RESULTS_FILE)

    print("Validating water allocation...")
    validate_allocations(results, INVENTORY)

    total_demand = int(results["water_units_demand"].sum())
    total_allocated = int(
        results["water_units_allocated"].sum()
    )
    total_unmet = int(
        results["water_units_unmet_demand"].sum()
    )

    coverage = (
        100.0 * total_allocated / total_demand
        if total_demand > 0
        else 0.0
    )

    utilization = (
        100.0 * total_allocated / INVENTORY.water_units
        if INVENTORY.water_units > 0
        else 0.0
    )

    print("\n========== WATER ALLOCATION EVALUATION ==========")
    print(f"Grid cells evaluated: {len(results):,}")
    print(f"Total water demand: {total_demand:,}")
    print(f"Water inventory: {INVENTORY.water_units:,}")
    print(f"Water allocated: {total_allocated:,}")
    print(f"Unmet water demand: {total_unmet:,}")
    print(f"Demand coverage: {coverage:.3f}%")
    print(f"Inventory utilization: {utilization:.3f}%")

    assert total_allocated <= INVENTORY.water_units
    assert total_allocated + total_unmet == total_demand

    print("\nInventory constraint: PASS")
    print("Demand reconciliation: PASS")
    print("Water allocation evaluation completed.")


if __name__ == "__main__":
    main()
