import math

from drone_model.scripted.route import waypoint

CLEARANCE = 0.55


class Footprints:
    """A planner grid with square footprints blocked, the way the drone's world map blocks pillars it has seen"""

    def __init__(self, footprints: list[tuple[int, int, int]]) -> None:
        self.boxes = [(x - CLEARANCE, z - CLEARANCE, x + w + CLEARANCE, z + w + CLEARANCE) for x, z, w in footprints]

    def free(self, x: float, z: float) -> bool:
        return not any(x0 < x < x1 and z0 < z < z1 for x0, z0, x1, z1 in self.boxes)

    def clear_line(self, ax: float, az: float, bx: float, bz: float) -> bool:
        steps = max(1, int(math.hypot(bx - ax, bz - az) / 0.25))
        return all(self.free(ax + (bx - ax) * t / steps, az + (bz - az) * t / steps) for t in range(steps + 1))


def test_route_goes_around_a_pillar():
    # the pillar sits squarely between the drone and the goal
    grid = Footprints([(-1, 3, 2)])
    state = {"pos": [0.0, -58.0, 0.0]}
    assert not grid.clear_line(0.0, 0.0, 0.5, 8.5)
    target = waypoint(state, (0.5, 8.5), grid)
    assert target != (0.5, 8.5)
    assert grid.clear_line(0.0, 0.0, *target)


def test_route_escapes_when_starting_inside_clearance():
    # the drone hugs the east face of the pillar, the goal is far to the west
    grid = Footprints([(4, -1, 2)])
    state = {"pos": [6.3, -59.0, -0.7]}
    assert not grid.free(6.3, -0.7)
    target = waypoint(state, (-10.5, -6.5), grid)
    assert grid.free(*target)
    assert target != (-10.5, -6.5)
