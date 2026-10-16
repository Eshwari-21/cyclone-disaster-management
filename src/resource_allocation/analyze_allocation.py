
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "resource_allocation_results_2024.csv"
)

OUTPUT_DIR = ROOT / "outputs"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Allocation results not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE)

    required = [
        "grid_id",
        "priority",
        "priority_rank",
        "water_units_demand",
        "water_units_allocated",
        "water_units_unmet_demand",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data["grid_id"].duplicated().any():
        raise ValueError("Duplicate grid IDs found.")

    if not (
        data["water_units_allocated"]
        + data["water_units_unmet_demand"]
        == data["water_units_demand"]
    ).all():
        raise ValueError("Water allocation does not reconcile.")

    # Rank grids by the priority score calculated by the agent.
    data = data.sort_values(
        ["priority", "grid_id"],
        ascending=[False, True],
    ).copy()

    data["cumulative_water_allocated"] = (
        data["water_units_allocated"].cumsum()
    )

    data["cumulative_water_demand"] = (
        data["water_units_demand"].cumsum()
    )

    data["allocation_share_pct"] = (
        100
        * data["water_units_allocated"]
        / max(data["water_units_allocated"].sum(), 1)
    )

    # Save all grids, ordered from highest to lowest priority.
    priority_report = OUTPUT_DIR / "water_priority_audit.csv"

    data.to_csv(priority_report, index=False)

    # Save the top 100 priority grids for easier inspection.
    top_grids = data.head(100).copy()

    top_report = OUTPUT_DIR / "top_100_water_priority_grids.csv"

    top_grids.to_csv(top_report, index=False)

    # Summarize whether the highest-priority grids received water.
    total_allocated = int(data["water_units_allocated"].sum())
    total_demand = int(data["water_units_demand"].sum())
    total_unmet = int(data["water_units_unmet_demand"].sum())

    positive = data[data["water_units_allocated"] > 0]

    summary = pd.DataFrame(
        [
            {
                "grid_count": len(data),
                "total_water_demand": total_demand,
                "total_water_allocated": total_allocated,
                "total_unmet_water_demand": total_unmet,
                "grids_receiving_water": len(positive),
                "grids_receiving_no_water": int(
                    (data["water_units_allocated"] == 0).sum()
                ),
                "maximum_allocation_per_grid": int(
                    data["water_units_allocated"].max()
                ),
                "mean_priority_receiving_water": (
                    float(positive["priority"].mean())
                    if not positive.empty else 0.0
                ),
                "mean_priority_receiving_no_water": (
                    float(
                        data.loc[
                            data["water_units_allocated"] == 0,
                            "priority",
                        ].mean()
                    )
                    if (data["water_units_allocated"] == 0).any()
                    else 0.0
                ),
                "demand_coverage_pct": (
                    100 * total_allocated / total_demand
                    if total_demand else 0.0
                ),
            }
        ]
    )

    summary_file = OUTPUT_DIR / "water_allocation_audit_summary.csv"

    summary.to_csv(summary_file, index=False)

    print("\n========== WATER ALLOCATION AUDIT ==========")
    print(summary.to_string(index=False))

    print(f"\nSaved: {priority_report}")
    print(f"Saved: {top_report}")
    print(f"Saved: {summary_file}")
    print("\nAllocation audit completed successfully.")


if __name__ == "__main__":
    main()
