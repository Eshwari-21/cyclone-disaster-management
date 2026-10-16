
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "resource_allocation_results_2024.csv"
)

GRID_FILE = (
    ROOT
    / "data"
    / "processed"
    / "risk_map_2024.gpkg"
)

OUTPUT_DIR = ROOT / "outputs"


# --------------------------------------------------
# DATA LOADING AND VALIDATION
# --------------------------------------------------

def load_map_data():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Allocation results not found: {RESULTS_FILE}"
        )

    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"Grid GeoPackage not found: {GRID_FILE}"
        )

    results = pd.read_csv(RESULTS_FILE)
    grids = gpd.read_file(GRID_FILE)

    required_columns = {
        "grid_id",
        "water_units_allocated",
        "water_units_unmet_demand",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in allocation CSV: {sorted(missing)}"
        )

    if "grid_id" not in grids.columns:
        raise ValueError(
            "The GeoPackage does not contain a grid_id column."
        )

    if results["grid_id"].isna().any():
        raise ValueError("Allocation CSV contains missing grid IDs.")

    if grids["grid_id"].isna().any():
        raise ValueError("GeoPackage contains missing grid IDs.")

    if results["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in allocation CSV."
        )

    if grids["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in GeoPackage."
        )

    numeric_columns = [
        "water_units_allocated",
        "water_units_unmet_demand",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="raise",
        )

        if not results[column].map(
            lambda value: pd.notna(value)
            and value >= 0
            and value != float("inf")
            and value != float("-inf")
        ).all():
            raise ValueError(
                f"{column} contains invalid values."
            )

        if not (results[column] % 1 == 0).all():
            raise ValueError(
                f"{column} must contain whole water units."
            )

    map_data = grids.merge(
        results[
            [
                "grid_id",
                "water_units_allocated",
                "water_units_unmet_demand",
            ]
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (map_data["_merge"] != "both").sum()

    if unmatched:
        raise ValueError(
            f"{unmatched} map grids have no matching allocation data."
        )

    map_data = map_data.drop(columns=["_merge"])

    if map_data.empty:
        raise ValueError("No map grids were loaded.")

    return map_data


# --------------------------------------------------
# PRINT DIAGNOSTICS
# --------------------------------------------------

def print_diagnostics(map_data):
    allocated = map_data["water_units_allocated"]
    unmet = map_data["water_units_unmet_demand"]

    print("\n========== WATER MAP DIAGNOSTICS ==========")

    print(f"Grid cells mapped: {len(map_data):,}")
    print(f"Total water allocated: {allocated.sum():,.0f}")
    print(f"Total unmet water demand: {unmet.sum():,.0f}")

    print(
        "Grids receiving water:",
        int((allocated > 0).sum()),
    )

    print(
        "Grids receiving no water:",
        int((allocated == 0).sum()),
    )

    print(
        "Distinct allocation values:",
        allocated.nunique(),
    )

    print(
        "Largest allocation to one grid:",
        f"{allocated.max():,.0f}",
    )

    print("\nAllocation distribution:")
    print(
        allocated.describe(
            percentiles=[0.50, 0.90, 0.99]
        ).to_string()
    )

    print("\nUnmet-demand distribution:")
    print(
        unmet.describe(
            percentiles=[0.50, 0.90, 0.99]
        ).to_string()
    )


# --------------------------------------------------
# COLOR NORMALIZATION
# --------------------------------------------------

def get_color_normalization(values, use_log=False):
    """
    Use actual values for color mapping.
    Log normalization helps show variation when a few
    grids have much larger values than the rest.
    """

    values = pd.to_numeric(values, errors="coerce")
    positive = values[values > 0]

    if positive.empty:
        return mcolors.Normalize(vmin=0, vmax=1)

    vmax = float(positive.max())

    if use_log:
        # LogNorm cannot represent zero. Zero-valued grids
        # are drawn separately in a neutral background color.
        vmin = float(positive.min())

        if vmax > vmin:
            return mcolors.LogNorm(
                vmin=vmin,
                vmax=vmax,
                clip=True,
            )

    if vmax <= 0:
        vmax = 1

    return mcolors.Normalize(
        vmin=0,
        vmax=vmax,
        clip=True,
    )


# --------------------------------------------------
# RENDER ONE MAP
# --------------------------------------------------

def render_map(
    map_data,
    filename,
    title,
    column,
    cmap,
    colorbar_label,
    use_log=False,
):
    output_path = OUTPUT_DIR / filename

    values = map_data[column]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw zero-valued grids in a neutral shade.
    zero_data = map_data[values == 0]

    if not zero_data.empty:
        zero_data.plot(
            ax=ax,
            color="#f2f2f2",
            linewidth=0,
        )

    # Draw positive values using a data-driven color scale.
    positive_data = map_data[values > 0]

    if not positive_data.empty:
        normalization = get_color_normalization(
            positive_data[column],
            use_log=use_log,
        )

        positive_data.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            norm=normalization,
            linewidth=0,
            legend=True,
            legend_kwds={
                "label": colorbar_label,
                "shrink": 0.72,
                "format": "%.2g",
            },
        )

    ax.set_title(title, fontsize=15, pad=16)
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Map saved to: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading allocation results and grid geometry...")

    map_data = load_map_data()

    print_diagnostics(map_data)

    print("\nGenerating water allocation map...")

    render_map(
        map_data=map_data,
        filename="water_allocation_map_2024.png",
        title="Water Allocated by Grid — 2024",
        column="water_units_allocated",
        cmap="Blues",
        colorbar_label="Water units allocated",
        use_log=False,
    )

    print("\nGenerating unmet water demand map...")

    render_map(
        map_data=map_data,
        filename="unmet_water_demand_map_2024.png",
        title="Unmet Water Demand by Grid — 2024",
        column="water_units_unmet_demand",
        cmap="OrRd",
        colorbar_label="Unmet water demand (units)",
        use_log=True,
    )

    print("\nWater map generation completed successfully.")


if __name__ == "__main__":
    main()
