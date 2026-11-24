"""A* on a grid: True means occupied, False means free."""

from heapq import heappop, heappush
from math import hypot


def a_star_search(occupied, start, goal, cell_size=(1.0, 1.0)):
    """Return a list of (row, column) cells, or None if there is no path."""
    rows, cols = occupied.shape
    dx, dy = cell_size
    for row, col in (start, goal):
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("Start or goal is outside the grid")
        if occupied[row, col]:
            raise ValueError("Start or goal is in an obstacle or clearance zone")

    def heuristic(cell):
        return hypot((cell[1] - goal[1]) * dx, (cell[0] - goal[0]) * dy)

    open_set = [(heuristic(start), 0.0, start)]
    g_cost = {start: 0.0}
    parent = {}
    visited = set()
    moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while open_set:
        _, cost, current = heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]

        row, col = current
        for dr, dc in moves:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if occupied[nr, nc] or neighbor in visited:
                continue
            if dr != 0 and dc != 0:
                if occupied[row + dr, col] or occupied[row, col + dc]:
                    continue

            new_cost = cost + hypot(dc * dx, dr * dy)
            if new_cost < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = new_cost
                parent[neighbor] = current
                heappush(open_set, (new_cost + heuristic(neighbor), new_cost, neighbor))
    return None
