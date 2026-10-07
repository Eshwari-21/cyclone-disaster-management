from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_results_summary.csv"
)


# ---------------------------------------------------------
# Final modeling results
# ---------------------------------------------------------

summary = [
    {
        "category": "Dataset",
        "metric": "Grid cells",
        "value": "17,737",
        "interpretation": "Number of 1 km spatial grid cells used for modeling.",
    },
    {
        "category": "Dataset",
        "metric": "Modeling features",
        "value": "7",
        "interpretation": (
            "Elevation, slope, annual rainfall, NDVI, "
            "log population, maximum hazard, cumulative hazard."
        ),
    },
    {
        "category": "Spatial graph",
        "metric": "Directed edges",
        "value": "136,112",
        "interpretation": (
            "Neighbor relationships connecting spatially adjacent grid cells."
        ),
    },
    {
        "category": "GNN",
        "metric": "Embedding dimensions",
        "value": "8",
        "interpretation": (
            "Each grid cell is represented by an 8-dimensional learned spatial embedding."
        ),
    },
    {
        "category": "GNN",
        "metric": "Selected cluster count",
        "value": "2",
        "interpretation": (
            "Two spatial profiles were obtained from the selected GNN representation."
        ),
    },
    {
        "category": "GNN",
        "metric": "Seed-42 silhouette",
        "value": "0.4883",
        "interpretation": (
            "Measures separation of the two clusters in the learned embedding space."
        ),
    },
    {
        "category": "GNN",
        "metric": "Spatial neighbor agreement",
        "value": "0.9423",
        "interpretation": (
            "Average fraction of neighboring cells belonging to the same cluster."
        ),
    },
    {
        "category": "PCA baseline",
        "metric": "Silhouette",
        "value": "0.4581",
        "interpretation": (
            "Deterministic baseline clustering performance after PCA."
        ),
    },
    {
        "category": "PCA baseline",
        "metric": "Spatial neighbor agreement",
        "value": "0.9167",
        "interpretation": (
            "Spatial coherence of the deterministic baseline clusters."
        ),
    },
    {
        "category": "Model comparison",
        "metric": "GNN silhouette improvement",
        "value": "+0.0302",
        "interpretation": (
            "Difference between selected GNN silhouette and PCA baseline silhouette."
        ),
    },
    {
        "category": "Model comparison",
        "metric": "PCA-GNN ARI",
        "value": "0.7655",
        "interpretation": (
            "Strong agreement between PCA and GNN cluster assignments."
        ),
    },
    {
        "category": "Stability",
        "metric": "Mean multi-seed silhouette",
        "value": "0.3460",
        "interpretation": (
            "Average silhouette across five GNN seeds in the 300-epoch stability experiment."
        ),
    },
    {
        "category": "Stability",
        "metric": "Mean pairwise ARI",
        "value": "0.3419",
        "interpretation": (
            "Indicates that GNN cluster assignments remain sensitive to initialization."
        ),
    },
    {
        "category": "Interpretation",
        "metric": "Cluster meaning",
        "value": "Spatial profiles",
        "interpretation": (
            "Clusters represent learned spatial/environmental profiles, "
            "not independently observed disaster-severity classes."
        ),
    },
    {
        "category": "Limitation",
        "metric": "Observed impact target",
        "value": "Not available",
        "interpretation": (
            "The current dataset does not contain an independent historical disaster-impact "
            "target, so the model should not be described as a validated damage/severity predictor."
        ),
    },
]


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

summary_df = pd.DataFrame(summary)


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

summary_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Display summary
# ---------------------------------------------------------

print("FINAL MODELING RESULTS SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    f"Saved modeling summary to:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Modeling results summary completed."
)