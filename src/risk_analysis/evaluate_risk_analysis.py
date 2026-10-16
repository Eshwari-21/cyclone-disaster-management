
"""Evaluate how sensitive risk rankings are to score weights."""

from pathlib import Path

import pandas as pd

from src.risk_analysis.risk_scoring import (
    RiskWeights,
    calculate_risk_scores,
)


INPUT = Path("data/processed/modeling_grid_2024.csv")

WEIGHT_SCENARIOS = {
    "baseline": RiskWeights(0.30, 0.40, 0.30),
    "equal_weights": RiskWeights(1 / 3, 1 / 3, 1 / 3),
    "hazard_heavy": RiskWeights(0.40, 0.40, 0.20),
    "population_heavy": RiskWeights(0.20, 0.30, 0.50),
}


def main():
    data = pd.read_csv(INPUT)
    scores = {}

    for name, weights in WEIGHT_SCENARIOS.items():
        result = calculate_risk_scores(data, weights)
        scores[name] = result.set_index("grid_id")["risk_score"]

    baseline = scores["baseline"]
    top_count = max(1, int(len(baseline) * 0.10))
    baseline_top = set(baseline.nlargest(top_count).index)

    print(f"Grid cells evaluated: {len(data):,}")
    print(f"Top-priority group size: {top_count:,}")

    print("\nScenario summary:")
    for name, values in scores.items():
        top_cells = set(values.nlargest(top_count).index)
        overlap = len(baseline_top & top_cells) / top_count

        correlation = baseline.corr(values, method="spearman")

        print(f"\n{name}")
        print(f"  Mean score: {values.mean():.2f}")
        print(f"  Minimum score: {values.min():.2f}")
        print(f"  Maximum score: {values.max():.2f}")
        print(f"  Spearman rank correlation with baseline: "
              f"{correlation:.4f}")
        print(f"  Top 10% overlap with baseline: {overlap:.1%}")


if __name__ == "__main__":
    main()
