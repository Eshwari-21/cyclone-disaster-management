"""Simulation inventory for cyclone resource allocation.

These values are assumptions for testing, not verified real-world supplies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceInventory:
    food_kits: int = 5000
    water_units: int = 10000
    medical_kits: int = 2000
    rescue_teams: int = 50

    def as_dict(self) -> dict[str, int]:
        return {
            "food_kits": self.food_kits,
            "water_units": self.water_units,
            "medical_kits": self.medical_kits,
            "rescue_teams": self.rescue_teams,
        }

    def validate(self) -> None:
        for name, quantity in self.as_dict().items():
            if isinstance(quantity, bool) or not isinstance(quantity, int):
                raise TypeError(f"{name} must be an integer")

            if quantity < 0:
                raise ValueError(f"{name} cannot be negative")
