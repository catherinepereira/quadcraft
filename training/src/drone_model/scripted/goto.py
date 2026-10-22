"""
The planner for getting somewhere: a point given as coordinates, a block of a named kind it has to find with its
camera, or a player or mob to follow at a distance. A followed player is a beacon the drone knows, a followed mob one it
finds and keeps track of with its camera. Each step it names one goal, which the scripted controller or the learned
goto policy flies, around whatever stands in the way
"""

from __future__ import annotations

import math
from pathlib import Path

from ..perception.mobs import Mob
from .base import SEARCH_HEIGHT, SPIN_TICKS, Planner, center, tool_action

# a found block is reached hovering this far over its top face, inside the job's arrive radius from its center
BLOCK_HOVER = 0.7
# a follower holds its camera this far over the target's middle, where a player's or villager's arms can't reach
FOLLOW_HOVER = 1.5
# a follow's "at" point names the mob of its kind within this many blocks of it, and counts as reached this close
AT_RADIUS = 5.0
AT_ARRIVE = 2.0


class GotoPlanner(Planner):
    def __init__(self, mask_ids: dict, depth_max: float = 64.0, reader=None, schematics: Path | None = None, core=None) -> None:
        super().__init__(mask_ids, depth_max, reader, core=core)
        # the mob a follow is on, kept by identity so the drone stays with one of several of its kind
        self.followed: Mob | None = None
        self.reached_at = False

    def decide(self, state: dict) -> dict:
        job = state["job"]
        if job["kind"] == "goto":
            return self.go(state, center(job["point"]), "goto")
        if job["kind"] == "follow":
            target = state.get("followTarget") or self.follow_mob(state, job)
            if target is not None:
                return self.go(state, tuple(target), "follow", standoff=(job["near"] + job["far"]) / 2, hover=FOLLOW_HOVER)
            at = job.get("at")
            if at is not None and not self.reached_at:
                pos = state["pos"]
                if math.hypot(at[0] + 0.5 - pos[0], at[2] + 0.5 - pos[2]) > AT_ARRIVE:
                    return self.go(state, (at[0] + 0.5, self.cruise_y(state, SEARCH_HEIGHT), at[2] + 0.5), "goto")
                # look all around the point before searching farther out
                self.reached_at = True
                self.spin_left = SPIN_TICKS
            return self.search(state)
        found = self.nearest(state, self.known(job["block"]))
        if found is not None:
            top = center(found)
            return self.go(state, (top[0], top[1] + 0.5 + BLOCK_HOVER, top[2]), "goto")
        return self.search(state)

    def follow_mob(self, state: dict, job: dict) -> tuple | None:
        """
        Where the followed mob was last seen, the same mob while the tracker keeps it. A new pick is the one of its kind
        nearest the job's "at" point, or with none the one nearest the drone, once the drone has looked all around
        """
        kind = self.core.kind_of.get(job["entity"])
        mobs = [m for m in self.core.mobs.mobs if m.kind == kind]
        if self.followed not in mobs:
            at = job.get("at")
            if at is not None:
                spot = (at[0] + 0.5, at[2] + 0.5)
                near = [m for m in mobs if math.dist((m.pos[0], m.pos[2]), spot) <= AT_RADIUS]
                self.followed = min(near, key=lambda m: math.dist((m.pos[0], m.pos[2]), spot), default=None)
            else:
                # one seen first may not be the nearest, finish the opening turn first
                self.followed = None if self.spin_left > 0 else min(mobs, key=lambda m: math.dist(m.pos, state["pos"]), default=None)
        return None if self.followed is None else tuple(self.followed.pos)

    def search(self, state: dict) -> dict:
        """Explore for what the job names, the goal is the next spot of the search"""
        action = self.explore(state)
        if self.search_goal is not None and self.spin_left == 0:
            goal = self.search_goal
            self.intent = {"mode": "explore", "aim": (goal[0], self.cruise_y(state, SEARCH_HEIGHT), goal[1]), "standoff": 0.0, "hover": 0.0}
        return action

    def go(self, state: dict, point, mode: str, standoff: float = 0.0, hover: float = 0.0) -> dict:
        """Fly to point, or to standoff blocks from it across the ground with the camera hover over it, facing it"""
        self.intent = {"mode": mode, "aim": tuple(point), "standoff": standoff, "hover": hover}
        move, _ = self.fly(state, point, standoff=standoff, hover=hover)
        return tool_action(move)
