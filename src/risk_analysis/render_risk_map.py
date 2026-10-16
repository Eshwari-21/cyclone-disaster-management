
"""Render a polished, presentation-ready cyclone risk-screening map."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Resolve paths from the project root, not the terminal's current directory.
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "risk_map_2024.gpkg"
OUTPUT_PATH = ROOT_DIR / "outputs" / "risk_map_2024.png"

CATEGORY_ORDER = ["Low", "Moderate", "High", "Very High"]

COLORS = {
    "Low": "#3182BD",
    "Moderate": "#FEE08B",
    "High": "#F46D43",
    "Very High": "#B2182B",
}


def render_risk_map():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Risk GeoPackage not found: {INPUT_PATH}"
        )

    grid = gpd.read_file(INPUT_PATH)

    required = {"grid_id", "risk_category", "geometry"}
    missing = required - set(grid.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if grid.crs is None:
        raise ValueError("The grid CRS is missing.")

    if grid["risk_category"].isna().any():
        raise ValueError("Some grid cells have no risk category.")

    if grid.geometry.isna().any() or grid.geometry.is_empty.any():
        raise ValueError("Some grid cells have empty geometries.")

    unknown = set(grid["risk_category"].astype(str)) - set(COLORS)
    if unknown:
        raise ValueError(f"Unexpected risk categories: {unknown}")

    # Project to metres for map display and the scale bar.
    projected = grid.to_crs("EPSG:32644")

    # Portrait layout gives the north-south coastal study area room.
    fig, ax = plt.subplots(figsize=(12, 13), facecolor="white")
    ax.set_facecolor("white")

    # Draw lower-risk cells first, then higher-risk cells.
    for category in CATEGORY_ORDER:
        subset = projected[
            projected["risk_category"] == category
        ]

        if not subset.empty:
            subset.plot(
                ax=ax,
                color=COLORS[category],
                edgecolor="none",
                linewidth=0,
            )

    # Keep the map proportions geographically correct.
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.015, y=0.015)

    # Title and subtitle.
    fig.suptitle(
        "Cyclone Risk Screening Map — 2024",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )

    ax.set_title(
        "Coastal Andhra Pradesh Study Area",
        fontsize=12,
        pad=12,
    )

    # North arrow.
    ax.annotate(
        "",
        xy=(0.94, 0.95),
        xytext=(0.94, 0.87),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.8,
            "color": "black",
        },
        zorder=10,
    )

    ax.text(
        0.94,
        0.955,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    # Scale bar: 10 km in the projected CRS.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    map_width = xmax - xmin
    map_height = ymax - ymin
    bar_length = 10_000  # metres

    # Place the scale bar inside the lower-left map area.
    bar_x = xmin + 0.06 * map_width
    bar_y = ymin + 0.08 * map_height

    ax.plot(
        [bar_x, bar_x + bar_length],
        [bar_y, bar_y],
        color="black",
        linewidth=3,
        solid_capstyle="butt",
        zorder=10,
    )

    ax.text(
        bar_x + bar_length / 2,
        bar_y + 0.012 * map_height,
        "10 km",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 2,
        },
        zorder=10,
    )

    # Legend in a dedicated footer area so it does not cover the map.
    handles = [
        Patch(
            facecolor=COLORS[category],
            edgecolor="none",
            label=category,
        )
        for category in CATEGORY_ORDER
    ]

    fig.legend(
        handles=handles,
        title="Relative risk category",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        borderpad=0.8,
        handlelength=1.5,
        columnspacing=1.6,
    )

    # Scientific disclaimer.
    fig.text(
        0.5,
        0.025,
        "Relative screening indicator; not a validated probability "
        "of cyclone damage.",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    ax.set_axis_off()

    # Reserve space for the title, legend, and disclaimer.
    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.91,
        bottom=0.16,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.15,
    )

    plt.close(fig)

    print(f"Grid cells mapped: {len(projected):,}")
    print(f"CRS used for display: {projected.crs}")
    print(f"Map saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    render_risk_map()
