
"""Join risk results to the spatial grid and export a GeoPackage."""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def create_risk_map_data():
    results_path = Path(
        "data/processed/risk_analysis_results_2024.csv"
    )
    grid_path = Path(
        "data/processed/decision_support_grid_2024.gpkg"
    )
    output_path = Path(
        "data/processed/risk_map_2024.gpkg"
    )

    print("Loading risk results...")
    results = pd.read_csv(results_path)

    print("Loading geographic grid...")
    grid = gpd.read_file(grid_path)

    # Ensure both datasets have valid, unique grid identifiers.
    for name, frame in [("results", results), ("grid", grid)]:
        if frame["grid_id"].isna().any():
            raise ValueError(f"{name} contains missing grid IDs.")
        if frame["grid_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate grid IDs.")

    # Verify that both datasets cover exactly the same grid cells.
    result_ids = set(results["grid_id"])
    grid_ids = set(grid["grid_id"])

    if result_ids != grid_ids:
        raise ValueError(
            "Grid IDs do not match between risk results and geometry."
        )

    # Keep geometry from the spatial grid and risk outputs from the CSV.
    
    risk_columns = [
    "grid_id",
    "risk_score",
    "risk_category",
    "max_hazard_norm",
    "cumulative_hazard_norm",
    "population_exposure_norm",
    "gnn_cluster",
    "gnn_profile",
]


    risk_data = results[risk_columns]

    # Remove any existing copies of these columns before joining.
    columns_to_replace = [
        col for col in risk_columns
        if col != "grid_id" and col in grid.columns
    ]
    grid = grid.drop(columns=columns_to_replace)

    risk_map = grid.merge(
        risk_data,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if risk_map["risk_score"].isna().any():
        raise ValueError("Some grid cells have no risk score.")

    if len(risk_map) != len(grid):
        raise ValueError("Grid row count changed after joining.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    risk_map.to_file(output_path, driver="GPKG")

    print(f"\nGrid cells mapped: {len(risk_map):,}")
    print(f"CRS: {risk_map.crs}")
    print(f"Saved geographic risk data to: {output_path}")

    print("\nRisk categories:")
    print(risk_map["risk_category"].value_counts().sort_index())

    return risk_map


if __name__ == "__main__":
    create_risk_map_data()
