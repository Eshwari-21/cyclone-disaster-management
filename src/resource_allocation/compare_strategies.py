"""Compare balanced, risk-first, and population-first allocation."""

from pathlib import Path

import pandas as pd

from .demand_estimation import estimate_demand
from .allocation_agent import allocate_resources
from .resource_inventory import ResourceInventory


INPUT = Path("data/processed/risk_analysis_results_2024.csv")
OUTPUT = Path("outputs/resource_allocation_strategy_comparison.csv")


STRATEGIES = {
    "balanced": (0.5, 0.5),
    "risk_first": (1.0, 0.0),
    "population_first": (0.0, 1.0),
}


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    data = pd.read_csv(INPUT)
    demand_data = estimate_demand(data)
    inventory = ResourceInventory()
    rows = []

    for name, (risk_weight, population_weight) in STRATEGIES.items():
        result = allocate_resources(
            demand_data,
            inventory=inventory,
            risk_weight=risk_weight,
            population_weight=population_weight,
        )

        high_risk = result["risk_category"].isin(
            ["High", "Very High"]
        )

        row = {
            "strategy": name,
            "grid_cells": len(result),
            "high_risk_grid_count": int(high_risk.sum()),
            "food_allocated": int(result["food_kits_allocated"].sum()),
            "water_allocated": int(result["water_units_allocated"].sum()),
            "medical_allocated": int(
                result["medical_kits_allocated"].sum()
            ),
            "rescue_teams_allocated": int(
                result["rescue_teams_allocated"].sum()
            ),
            "food_to_high_risk": int(
                result.loc[high_risk, "food_kits_allocated"].sum()
            ),
            "water_to_high_risk": int(
                result.loc[high_risk, "water_units_allocated"].sum()
            ),
            "medical_to_high_risk": int(
                result.loc[high_risk, "medical_kits_allocated"].sum()
            ),
            "rescue_teams_to_high_risk": int(
                result.loc[high_risk, "rescue_teams_allocated"].sum()
            ),
            "mean_risk_score_receiving_food": (
                result.loc[
                    result["food_kits_allocated"] > 0, "risk_score"
                ].mean()
            ),
            "mean_risk_score_receiving_water": (
                result.loc[
                    result["water_units_allocated"] > 0, "risk_score"
                ].mean()
            ),
        }

        for resource, allocation_col, unmet_col in [
            (
                "food",
                "food_kits_allocated",
                "food_kits_unmet_demand",
            ),
            (
                "water",
                "water_units_allocated",
                "water_units_unmet_demand",
            ),
            (
                "medical",
                "medical_kits_allocated",
                "medical_kits_unmet_demand",
            ),
            (
                "rescue",
                "rescue_teams_allocated",
                "rescue_teams_unmet_demand",
            ),
        ]:
            row[f"{resource}_unmet_demand"] = int(
                result[unmet_col].sum()
            )
            row[f"{resource}_grids_receiving"] = int(
                (result[allocation_col] > 0).sum()
            )

        rows.append(row)

    summary = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print("Strategy comparison completed.")
    print(f"Results saved to: {OUTPUT}")
    print(
        summary[
            [
                "strategy",
                "food_to_high_risk",
                "water_to_high_risk",
                "medical_to_high_risk",
                "rescue_teams_to_high_risk",
                "mean_risk_score_receiving_food",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
