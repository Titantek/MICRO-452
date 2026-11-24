"""Turn the obstacle polygons from vision into a grid.

Points are (x, y), with x right and y down, as in computer_vision.
Cells are (row, column); arrays are indexed [row, column].
"""

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from .config import NavigationConfig


def _points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite (x, y) points with shape (..., 2)")
    return points


def _segment_cells(start, end, cell_size):
    """Find every cell a line touches, even if it only touches a corner."""
    a, b = np.asarray(start) / cell_size, np.asarray(end) / cell_size
    delta = b - a
    times = [0.0, 1.0]

    for axis in (0, 1):
        if delta[axis] != 0:
            lo, hi = sorted((a[axis], b[axis]))
            for boundary in range(ceil(lo), floor(hi) + 1):
                t = (boundary - a[axis]) / delta[axis]
                if 0 < t < 1:
                    times.append(float(t))

    times = np.unique(times)
    samples = np.concatenate((times, (times[:-1] + times[1:]) / 2))
    cells = set()

    for t in samples:
        point = a + t * delta
        indices = []

        for coordinate in point:
            nearest = round(float(coordinate))
            if abs(coordinate - nearest) < 1e-10:
                indices.append((nearest - 1, nearest))
            else:
                indices.append((floor(coordinate),))

        for col in indices[0]:
            for row in indices[1]:
                cells.add((row, col))

    return cells


@dataclass
class GridMap:
    raw_occupied: np.ndarray
    occupied: np.ndarray
    config: NavigationConfig
    image_size: tuple

    @property
    def cell_size(self):
        """Actual (dx, dy) in metres, each <= configured resolution."""
        rows, cols = self.occupied.shape
        return (self.config.arena_width_m / cols, self.config.arena_height_m / rows)

    def pixels_to_meters(self, points):
        return _points(points) * (
            np.array([self.config.arena_width_m, self.config.arena_height_m])
            / self.image_size
        )

    def meters_to_pixels(self, points):
        return _points(points) * (
            np.asarray(self.image_size)
            / [self.config.arena_width_m, self.config.arena_height_m]
        )

    def meters_to_cell(self, point):
        point = _points(point)
        if point.shape != (2,):
            raise ValueError("Expected one (x, y) point")
        if (point < 0).any() or (
            point >= [self.config.arena_width_m, self.config.arena_height_m]
        ).any():
            raise ValueError("Point is outside the arena")
        col, row = np.floor(point / self.cell_size).astype(int)
        return int(row), int(col)

    def cell_to_meters(self, cells):
        """Convert (row, column) cells to (x, y) cell centres."""
        cells = _points(cells)
        if (
            (cells != np.floor(cells)).any()
            or (cells < 0).any()
            or (cells >= self.occupied.shape).any()
        ):
            raise ValueError("Cells must be integer indices inside the grid")
        return (cells[..., ::-1] + 0.5) * self.cell_size

    def segment_is_free(self, start, end):
        start, end = _points(start), _points(end)
        if start.shape != (2,) or end.shape != (2,):
            raise ValueError("Expected two (x, y) endpoints")
        extent = [self.config.arena_width_m, self.config.arena_height_m]
        if any((p <= 0).any() or (p >= extent).any() for p in (start, end)):
            return False
        rows, cols = self.occupied.shape
        return all(
            0 <= r < rows and 0 <= c < cols and not self.occupied[r, c]
            for r, c in _segment_cells(start, end, self.cell_size)
        )


def inflate_obstacles(raw_occupied, cell_size, clearance_m):
    """Inflate occupied cell rectangles by a metric disk, conservatively.

    A target cell is blocked if any part is within clearance of an occupied
    cell. This accounts for both cells' area, not just their centres.
    """
    raw = np.asarray(raw_occupied, dtype=bool)
    dx, dy = cell_size
    if (
        raw.ndim != 2
        or not raw.size
        or min(dx, dy) <= 0
        or not np.isfinite([dx, dy, clearance_m]).all()
        or clearance_m < 0
    ):
        raise ValueError("Expected a nonempty grid, positive cell sizes and nonnegative clearance")
    if clearance_m == 0 or not raw.any():
        return raw.copy()

    rx = min(ceil(clearance_m / dx) + 1, raw.shape[1])
    ry = min(ceil(clearance_m / dy) + 1, raw.shape[0])
    x = np.maximum(np.abs(np.arange(-rx, rx + 1)) - 1, 0) * dx
    y = np.maximum(np.abs(np.arange(-ry, ry + 1)) - 1, 0) * dy
    kernel = (np.hypot(x[None, :], y[:, None]) <= clearance_m + 1e-12).astype(np.uint8)
    return cv2.dilate(raw.astype(np.uint8), kernel).astype(bool)


def build_grid(frame, config=None):
    """Rasterize frame.obstacles, inflate for Thymio, and reserve arena edges.

    Marks polygon interiors and every cell touched by an edge; obstacles
    narrower than one cell are retained. Contours use frame.warped pixels.
    No image thresholding is repeated here.
    """
    config = NavigationConfig() if config is None else config
    height, width = frame.warped.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("frame.warped must be a nonempty image")

    rows = ceil(config.arena_height_m / config.resolution_m)
    cols = ceil(config.arena_width_m / config.resolution_m)
    raw = np.zeros((rows, cols), dtype=bool)
    grid = GridMap(raw, raw.copy(), config, (width, height))
    dx, dy = grid.cell_size
    xs, ys = np.meshgrid((np.arange(cols) + 0.5) * dx, (np.arange(rows) + 0.5) * dy)

    for obstacle in frame.obstacles:
        pixels = _points(obstacle.corners)
        if pixels.ndim != 2 or len(pixels) < 3:
            raise ValueError("Obstacle contours must contain at least three (x, y) points")
        if (pixels < 0).any() or (pixels > [width, height]).any():
            raise ValueError("Obstacle contour is outside frame.warped")

        polygon = grid.pixels_to_meters(pixels)
        contour = polygon.astype(np.float32)
        first_col, first_row = np.floor(polygon.min(axis=0) / grid.cell_size).astype(int)
        last_col, last_row = np.floor(polygon.max(axis=0) / grid.cell_size).astype(int)

        for row in range(first_row, min(last_row + 1, rows)):
            for col in range(first_col, min(last_col + 1, cols)):
                centre = ((col + 0.5) * dx, (row + 0.5) * dy)
                if cv2.pointPolygonTest(contour, centre, False) >= 0:
                    raw[row, col] = True

        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            for r, c in _segment_cells(a, b, grid.cell_size):
                if 0 <= r < rows and 0 <= c < cols:
                    raw[r, c] = True

    grid.occupied = inflate_obstacles(raw, grid.cell_size, config.clearance_m)
    margin = config.clearance_m
    boundary = (
        (xs - dx / 2 < margin - 1e-12)
        | (xs + dx / 2 > config.arena_width_m - margin + 1e-12)
        | (ys - dy / 2 < margin - 1e-12)
        | (ys + dy / 2 > config.arena_height_m - margin + 1e-12)
    )
    grid.occupied |= boundary
    return grid
