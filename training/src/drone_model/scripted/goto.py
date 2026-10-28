"""
The planner for getting to a point given as coordinates. Each step it names one goal, which the scripted controller or
the learned goto policy flies, around whatever stands in the way
"""

from __future__ import annotations

from pathlib import Path

from .base import Planner, center, tool_action


class GotoPlanner(Planner):
    def __init__(self, mask_ids: dict, depth_max: float = 64.0, reader=None, schematics: Path | None = None, core=None) -> None:
        super().__init__(mask_ids, depth_max, reader, core=core)

    def decide(self, state: dict) -> dict:
        return self.go(state, center(state["job"]["point"]), "goto")

    def go(self, state: dict, point, mode: str, standoff: float = 0.0, hover: float = 0.0) -> dict:
        """Fly to point, or to standoff blocks from it across the ground with the camera hover over it, facing it"""
        self.intent = {"mode": mode, "aim": tuple(point), "standoff": standoff, "hover": hover}
        move, _ = self.fly(state, point, standoff=standoff, hover=hover)
        return tool_action(move)
