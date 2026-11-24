"""All the values we can change are here. Distances are in metres."""

from dataclasses import dataclass
from math import isfinite


@dataclass
class NavigationConfig:
    arena_width_m: float = 1.485
    arena_height_m: float = 1.05
    resolution_m: float = 0.005
    robot_radius_m: float = 0.079
    safety_margin_m: float = 0.002
    smoothing_radius_m: float = 0.04
    waypoint_spacing_m: float = 0.01

    @property
    def clearance_m(self):
        return self.robot_radius_m + self.safety_margin_m

    def __post_init__(self):
        positive = (
            self.arena_width_m,
            self.arena_height_m,
            self.resolution_m,
            self.waypoint_spacing_m,
        )
        nonnegative = (self.robot_radius_m, self.safety_margin_m, self.smoothing_radius_m)

        if any(not isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Arena size, resolution and waypoint spacing must be positive")
        if any(not isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError("Robot radius, margin and smoothing must be nonnegative")
