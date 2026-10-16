
import pandas as pd
import pytest

from src.resource_allocation.allocation_agent import (
    allocate_capped,
    allocate_resources,
)
from src.resource_allocation.resource_inventory import ResourceInventory


def sample_data():
    """Create a predictable dataset for allocation tests."""
    return pd.DataFrame(
        {
            "grid_id": [1, 2, 3],
            "risk_score": [90, 60, 20],
            "population_exposure_norm": [0.9, 0.6, 0.2],
            "water_units_demand": [8, 8, 8],
        }
    )


def test_inventory_constraint():
    """Total allocation must not exceed available inventory."""
    inventory = ResourceInventory(water_units=10)

    result = allocate_resources(
        sample_data(),
        inventory=inventory,
        risk_weight=0.5,
        population_weight=0.5,
    )

    assert result["water_units_allocated"].sum() <= 10


def test_allocation_does_not_exceed_demand():
    """No grid should receive more water than it needs."""
    inventory = ResourceInventory(water_units=10)

    result = allocate_resources(
        sample_data(),
        inventory=inventory,
    )

    assert (
        result["water_units_allocated"]
        <= result["water_units_demand"]
    ).all()


def test_allocated_plus_unmet_equals_demand():
    """Allocated water plus unmet demand must equal total demand."""
    inventory = ResourceInventory(water_units=10)

    result = allocate_resources(
        sample_data(),
        inventory=inventory,
    )

    assert (
        result["water_units_allocated"]
        + result["water_units_unmet_demand"]
        == result["water_units_demand"]
    ).all()


def test_all_allocations_are_nonnegative_integers():
    """Allocated and unmet water must be nonnegative whole units."""
    inventory = ResourceInventory(water_units=10)

    result = allocate_resources(
        sample_data(),
        inventory=inventory,
    )

    for column in [
        "water_units_allocated",
        "water_units_unmet_demand",
    ]:
        values = result[column]

        assert (values >= 0).all()
        assert (values % 1 == 0).all()


def test_higher_risk_gets_more_under_risk_first_strategy():
    """With equal exposure, the higher-risk grid should receive
    at least as much water under risk-first weighting."""
    data = pd.DataFrame(
        {
            "grid_id": [1, 2],
            "risk_score": [100, 10],
            "population_exposure_norm": [0.5, 0.5],
            "water_units_demand": [100, 100],
        }
    )

    inventory = ResourceInventory(water_units=10)

    result = allocate_resources(
        data,
        inventory=inventory,
        risk_weight=1.0,
        population_weight=0.0,
    ).set_index("grid_id")

    assert (
        result.loc[1, "water_units_allocated"]
        >= result.loc[2, "water_units_allocated"]
    )


def test_priority_weighted_allocation_can_assign_multiple_units():
    """A high-priority grid with sufficient demand can receive
    multiple units under the updated allocation policy."""
    data = pd.DataFrame(
        {
            "grid_id": [1, 2, 3],
            "risk_score": [100, 50, 10],
            "population_exposure_norm": [1.0, 0.5, 0.1],
            "water_units_demand": [100, 100, 100],
        }
    )

    inventory = ResourceInventory(water_units=10)

    result = allocate_resources(
        data,
        inventory=inventory,
        risk_weight=0.5,
        population_weight=0.5,
    )

    assert result["water_units_allocated"].sum() == 10

    assert (
        result["water_units_allocated"]
        <= result["water_units_demand"]
    ).all()

    assert result["water_units_allocated"].max() > 1

    assert (
        result["water_units_allocated"]
        + result["water_units_unmet_demand"]
        == result["water_units_demand"]
    ).all()


def test_allocate_capped_respects_demand_caps():
    """Allocation must redistribute water when a grid reaches demand."""
    demand = pd.Series([1, 100, 100])
    weights = pd.Series([100.0, 1.0, 1.0])

    result = allocate_capped(
        total_units=10,
        demand=demand,
        weights=weights,
    )

    assert result.sum() == 10
    assert (result <= demand).all()
    assert (result >= 0).all()
    assert (result % 1 == 0).all()


def test_zero_inventory_allocates_no_water():
    """No water can be allocated when inventory is zero."""
    result = allocate_capped(
        total_units=0,
        demand=pd.Series([10, 20, 30]),
        weights=pd.Series([0.8, 0.5, 0.2]),
    )

    assert result.sum() == 0
    assert (result == 0).all()


def test_zero_weights_use_equal_fallback():
    """Eligible grids receive equal shares if all weights are zero."""
    result = allocate_capped(
        total_units=6,
        demand=pd.Series([10, 10, 10]),
        weights=pd.Series([0.0, 0.0, 0.0]),
    )

    assert result.sum() == 6
    assert result.tolist() == [2, 2, 2]


def test_invalid_priority_weights_are_rejected():
    """Priority weights must be nonnegative and sum to one."""
    inventory = ResourceInventory(water_units=10)

    with pytest.raises(ValueError):
        allocate_resources(
            sample_data(),
            inventory=inventory,
            risk_weight=0.8,
            population_weight=0.5,
        )

    with pytest.raises(ValueError):
        allocate_resources(
            sample_data(),
            inventory=inventory,
            risk_weight=-0.1,
            population_weight=1.1,
        )
