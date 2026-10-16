
"""Run the complete cyclone disaster management workflow."""

from src.risk_analysis.run_pipeline import (
    main as run_risk_analysis_pipeline,
)
from src.resource_allocation.run_pipeline import (
    main as run_allocation,
)
from src.resource_allocation.evaluation_metrics import (
    main as run_evaluation,
)
from src.resource_allocation.render_water_map import (
    main as render_water_maps,
)


def main():
    print("\n" + "=" * 65)
    print("CYCLONE DISASTER MANAGEMENT")
    print("COMPLETE END-TO-END WORKFLOW")
    print("=" * 65)

    # STEP 1: Generate and validate risk-analysis results.
    print("\n[STEP 1/4] Running Risk Analysis Agent...")
    run_risk_analysis_pipeline()
    print("\nRisk analysis completed successfully.")

    # STEP 2: Estimate water demand and allocate available water.
    print("\n[STEP 2/4] Running Water Allocation Agent...")
    run_allocation()
    print("\nWater allocation completed successfully.")

    # STEP 3: Evaluate allocation performance and strategies.
    print("\n[STEP 3/4] Evaluating water allocation...")
    run_evaluation()
    print("\nAllocation evaluation completed successfully.")

    # STEP 4: Generate allocation and unmet-demand maps.
    print("\n[STEP 4/4] Generating water allocation maps...")
    render_water_maps()
    print("\nWater maps generated successfully.")

    print("\n" + "=" * 65)
    print("COMPLETE WORKFLOW FINISHED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
