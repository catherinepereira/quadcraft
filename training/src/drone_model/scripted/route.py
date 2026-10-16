"""
Routes across the drone's world map (perception.worldmap): A* over cell centers around whatever the map has blocked,
then the farthest waypoint in clear line of sight
"""

from __future__ import annotations

import heapq
import math

# the search window reaches this many blocks past the drone and the goal
MARGIN = 8


def plan(start: tuple[float, float], goal: tuple[float, float], grid, bounds: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """A* over cell centers inside bounds (x0, z0, x1, z1), 8-connected with no corner cutting"""
    x0, z0, x1, z1 = bounds
    start_cell = (math.floor(start[0]), math.floor(start[1]))
    goal_cell = (math.floor(goal[0]), math.floor(goal[1]))

    def center(c):
        return (c[0] + 0.5, c[1] + 0.5)

    def ok(c):
        return c == goal_cell or (x0 <= c[0] <= x1 and z0 <= c[1] <= z1 and grid.free(*center(c)))

    frontier = [(0.0, start_cell)]
    came = {start_cell: None}
    cost = {start_cell: 0.0}
    while frontier:
        _, cell = heapq.heappop(frontier)
        if cell == goal_cell:
            break
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dz == 0:
                    continue
                nxt = (cell[0] + dx, cell[1] + dz)
                # the first move may start inside a clearance box (the drone drifted close), it only has to end clear
                if not ok(nxt) or (cell != start_cell and not grid.clear_line(*center(cell), *center(nxt))):
                    continue
                new_cost = cost[cell] + math.hypot(dx, dz)
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt] = new_cost
                    came[nxt] = cell
                    heapq.heappush(frontier, (new_cost + math.dist(nxt, goal_cell), nxt))
    if goal_cell not in came:
        return [goal]
    path = []
    cell = goal_cell
    while cell is not None:
        path.append(center(cell))
        cell = came[cell]
    path.reverse()
    path[-1] = goal
    return path


def waypoint(state: dict, goal: tuple[float, float], grid) -> tuple[float, float]:
    """Next point to steer at on the way to goal, planning around what the map has blocked"""
    pos = state["pos"]
    if not grid.boxes or grid.clear_line(pos[0], pos[2], *goal):
        return goal
    bounds = (
        math.floor(min(pos[0], goal[0])) - MARGIN,
        math.floor(min(pos[2], goal[1])) - MARGIN,
        math.floor(max(pos[0], goal[0])) + MARGIN,
        math.floor(max(pos[2], goal[1])) + MARGIN,
    )
    path = plan((pos[0], pos[2]), goal, grid, bounds)
    target = path[min(1, len(path) - 1)]
    for p in path[1:]:
        if grid.clear_line(pos[0], pos[2], *p):
            target = p
    return target
