"""Remove unnecessary waypoints, then round corners when there is room."""

from dataclasses import dataclass
from math import ceil

import numpy as np

from .map_utils import GridMap, build_grid
from .path_planning import a_star_search


def clean_path(path):
    result = [np.asarray(path[0], dtype=float)]
    for point in path[1:]:
        if not np.array_equal(point, result[-1]):
            result.append(point)
    return np.asarray(result, dtype=float)


def path_is_collision_free(path, grid):
    if len(path) == 1:
        return grid.segment_is_free(path[0], path[0])
    return all(grid.segment_is_free(a, b) for a, b in zip(path[:-1], path[1:]))


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def simplify_path(path, grid):
    """Keep the furthest waypoint reachable by a straight, clear segment."""
    result = [path[0]]
    index = 0

    while index < len(path) - 1:
        target = len(path) - 1
        while target > index + 1 and not grid.segment_is_free(path[index], path[target]):
            target -= 1
        result.append(path[target])
        index = target

    return np.asarray(result)


def resample_path(path, spacing_m):
    """Add points along each segment, keeping all the existing corners."""
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("Waypoint spacing must be positive")

    path = clean_path(path)
    result = [path[0]]

    for a, b in zip(path[:-1], path[1:]):
        count = max(1, ceil(np.linalg.norm(b - a) / spacing_m))
        for t in np.linspace(0, 1, count + 1)[1:]:
            result.append((1 - t) * a + t * b)
    return np.asarray(result)


def smooth_path(path, grid):
    """Try one quadratic Bezier rounding per corner; keep it only if clear."""
    path = clean_path(path)
    radius = grid.config.smoothing_radius_m
    spacing = grid.config.waypoint_spacing_m
    result = [path[0]]

    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        incoming, outgoing = b - a, c - b
        before, after = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        trim = min(radius, before / 2, after / 2)

        if trim == 0:
            result.append(b)
            continue

        entry = b - trim * incoming / before
        exit_point = b + trim * outgoing / after
        count = max(2, ceil(2 * trim / spacing))
        curve = []

        for t in np.linspace(0, 1, count + 1):
            curve.append((1 - t)**2 * entry + 2 * (1 - t) * t * b + t**2 * exit_point)

        candidate = [result[-1]] + curve + [c]
        if path_is_collision_free(candidate, grid):
            result.extend(curve)
        else:
            result.append(b)

    if len(path) > 1:
        result.append(path[-1])

    return resample_path(result, spacing)


@dataclass
class NavigationPlan:
    grid: GridMap
    raw_path_m: np.ndarray
    simplified_path_m: np.ndarray
    waypoints: np.ndarray


def plan_path(frame, config=None):
    """Connect the vision result to the planner. All returned points are in metres."""
    if frame.thymio is None or frame.goal is None:
        raise ValueError("Robot or goal marker is missing")

    grid = build_grid(frame, config)
    start = grid.pixels_to_meters(frame.thymio[:2])
    goal = grid.pixels_to_meters(frame.goal[:2])

    for name, point in (("Start", start), ("Goal", goal)):
        if not grid.segment_is_free(point, point):
            raise ValueError(f"{name} is outside the map or in an obstacle/clearance zone")

    start_cell = grid.meters_to_cell(start)
    goal_cell = grid.meters_to_cell(goal)
    cells = a_star_search(grid.occupied, start_cell, goal_cell, grid.cell_size)
    if cells is None:
        raise ValueError("No route with the current clearance")

    if start_cell == goal_cell:
        raw = clean_path([start, goal])
    else:
        raw = clean_path(np.vstack((start, grid.cell_to_meters(cells), goal)))

    simplified = simplify_path(raw, grid)
    waypoints = smooth_path(simplified, grid)

    if not path_is_collision_free(waypoints, grid):
        raise ValueError("Processed path failed the collision check")

    return NavigationPlan(grid, raw, simplified, waypoints)
