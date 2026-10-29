"""
Planners for the player's jobs. Each one reads the drone's voxel memory (perception.core) and picks the next goal from
it: a view to look at part of a box, a block to break, or a cell to place a block into. The scripted flight under
work_on and view carries the goals out, or the learned cell skill does when one is attached
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from ..energy import dock_intent
from ..perception.memory import AIR, MIN_VOTES, base_name
from ..perception.reader import CLASSES, INDEX
from .base import (
    SURVEY_LIMIT,
    Planner,
    center,
    tool_action,
)
from .goto import GotoPlanner

VIEW_SPACING = 4.0
VIEW_DIST = 4.0
TOP_HEIGHT = 4.0
GAP_VIEW_DIST = 2.2
# a viewpoint is where the camera goes, the drone's feet are about this far below it
VIA_EYE = 0.2
CARRIED_SHARE = 0.25
# trenches two wide every four blocks leave two-wide walls, so every block of a dug box has a face in a trench
# Two wide because the drone needs more than a block of room to fly down one
TRENCH_WIDTH = 2
MAX_STRIKES = 3
TRENCH_PERIOD = 4
# a work pose puts the camera straight out from a face's center, at the first of these distances with room
WORK_DIST = (2.5, 3.5)
# a side face with no room level with it is worked from 45 degrees above, still square-on, as (out, up) from the face,
# the closer ones and then level close in for a face down a two-wide trench
SIDE_FALLBACKS = ((1.77, 1.77), (1.5, 1.5), (1.5, 0.0))
# a top face with no room overhead is worked from 45 degrees off one side, the camera this far from the face
TOP_SLANT = 2.5
UP = (0, 1, 0)
SIDES = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))
# the drone's half width and height in blocks, its hitbox is 0.6 by 0.4
DRONE_HALF = 0.3
DRONE_HEIGHT = 0.4
# with nothing else left, cells given up on get this many more tries
RETRIES = 2
# the face of the support block that faces the cell, by the direction from the cell to the support
FACES = {(0, -1, 0): "up", (1, 0, 0): "west", (-1, 0, 0): "east", (0, 0, 1): "north", (0, 0, -1): "south", (0, 1, 0): "down"}


def neighbors(cell) -> list[tuple[int, int, int]]:
    x, y, z = cell
    return [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]


def spaced(lo: int, hi: int) -> list[float]:
    """Points spread at most VIEW_SPACING apart along the inclusive block range lo..hi"""
    width = hi - lo + 1
    n = max(1, math.ceil(width / VIEW_SPACING))
    return [lo + width * (i + 0.5) / n for i in range(n)]


def box_views(box) -> list[tuple[tuple, tuple]]:
    """Viewpoints and aim points that cover a box's four sides and its top"""
    x0, y0, z0, x1, y1, z1 = box
    cx, cz = (x0 + x1 + 1) / 2, (z0 + z1 + 1) / 2
    heights = list(np.arange(y0 + 1.6, y1 + 1.0, VIEW_SPACING)) or [y0 + 1.6]
    views = []
    for h in heights:
        look_y = min(max(h, y0 + 0.5), y1 + 0.5)
        for x in spaced(x0, x1):
            views.append(((x, h, z1 + 1 + VIEW_DIST), (x, look_y, cz)))
            views.append(((x, h, z0 - VIEW_DIST), (x, look_y, cz)))
        for z in spaced(z0, z1):
            views.append(((x1 + 1 + VIEW_DIST, h, z), (cx, look_y, z)))
            views.append(((x0 - VIEW_DIST, h, z), (cx, look_y, z)))
    for x in spaced(x0, x1):
        for z in spaced(z0, z1):
            views.append(((x, y1 + 1 + TOP_HEIGHT, z - 1.5), (x, y1 + 0.5, z)))
    return views


class JobPlanner(Planner):
    """Surveys boxes view by view, nearest view first, and keeps what the reader makes of them in memory"""

    def __init__(self, mask_ids: dict, depth_max: float = 64.0, reader=None, schematics: Path | None = None, core=None) -> None:
        super().__init__(mask_ids, depth_max, reader, core=core, schematics=schematics)
        self.schematics = schematics
        self.views: list[tuple[tuple, tuple]] | None = None
        self.view_steps = 0
        self.dwell = 0
        self.dug: set[tuple[int, int, int]] = set()
        # the block being dug for its own sake (ore, or a material for a build) and what it was read as
        self.target_cell: tuple[int, int, int] | None = None
        self.target_block: str | None = None
        self.trenches: dict[tuple, list[tuple[int, int, int]]] = {}
        # the work pose picked for each ("break" or "place", cell), kept so the drone doesn't switch faces on the way
        self.poses: dict[tuple, tuple] = {}
        self.retries = 0
        # blocks the drone's tier breaks without a drop, the server lists them with the job
        self.unharvestable: frozenset[str] = frozenset()

    def note_job(self, job: dict) -> None:
        """Takes what the job tells the planner beyond its boxes: the blocks the drone's tier breaks without a drop"""
        self.unharvestable = frozenset(job.get("unharvestable", ()))

    @property
    def scanned(self) -> dict[tuple[int, int, int], str]:
        """Scan perception: the block name of every cell of the job's scanned boxes, which the core read into memory"""
        return self.core.scanned

    @property
    def scanned_boxes(self) -> set[str]:
        return set(self.core.scans)

    def scanned_in(self, box, wanted) -> list[tuple[int, int, int]]:
        """Scanned cells in box holding one of the wanted blocks and not dug out since"""
        lo, hi = box[:3], box[3:]
        return [
            c for c, name in self.scanned.items()
            if name in wanted and c not in self.dug and c not in self.unreachable and all(lo[i] <= c[i] <= hi[i] for i in range(3))
        ]

    def shaft_blocker(self, cell, box) -> tuple[int, int, int] | None:
        """
        The next block to dig on a two-wide shaft from the top of box down to a buried cell: the cell's column and the
        one beside it, highest first, so the drone has room to follow it down
        """
        x0, x1, top = box[0], box[3], box[4]
        side = cell[0] + 1 if cell[0] + 1 <= x1 else cell[0] - 1
        for y in range(top, cell[1], -1):
            for x in (cell[0], side):
                c = (x, y, cell[2])
                if x0 <= x <= x1 and not self.empty(c):
                    return c
        side_cell = (side, cell[1], cell[2])
        return side_cell if x0 <= side <= x1 and not self.empty(side_cell) else None

    def start_survey(self, state: dict, box) -> None:
        pos = state["pos"]
        todo = [v for v in box_views(box) if self.free_at(v[0])]
        ordered = []
        here = (pos[0], pos[2])
        while todo:
            nxt = min(todo, key=lambda v: math.dist(here, (v[0][0], v[0][2])))
            todo.remove(nxt)
            ordered.append(nxt)
            here = (nxt[0][0], nxt[0][2])
        self.views = ordered

    def survey(self, state: dict) -> dict:
        """Fly to the next view and hold the aim there for a few frames"""
        via, look = self.views[0]
        self.view_steps += 1
        move, arrived = self.fly_to_view(state, via, look)
        if arrived or self.view_steps > SURVEY_LIMIT:
            self.dwell_on(state, look)
        return self.view(state, move, via, look)

    def believed(self, cell) -> str:
        return base_name(self.memory.label(cell))

    def wasted(self, cell) -> bool:
        """Whether breaking cell would destroy a block the drone gets nothing from, by its exact scanned name or its read"""
        name = self.scanned[cell] if cell in self.scanned and cell not in self.dug else self.believed(cell)
        return name in self.unharvestable

    def idle(self) -> dict:
        return tool_action([0, 0, 0, 0, 0])

    def retry_given_up(self) -> bool:
        """With nothing else left, gives the cells given up on another try, the digging since may have opened a way in"""
        if not self.unreachable or self.retries >= RETRIES:
            return False
        self.retries += 1
        self.unreachable.clear()
        self.bad_faces.clear()
        self.target_cell = None
        self.working_on = None
        return True

    def note_breaks(self, state: dict) -> None:
        for event in state.get("events", []):
            if event.get("type") == "break":
                self.dug.add(tuple(event["pos"]))

    def unseen(self, cell) -> bool:
        c = self.memory.cells.get(cell)
        return (c is None or c.total() < MIN_VOTES) and self.map.air.get(cell, 0) < 3

    def empty(self, cell) -> bool:
        return cell in self.dug or (self.believed(cell) == AIR and not self.unseen(cell))

    def next_dig(self, state: dict, box) -> tuple[int, int, int] | None:
        """The next trench cell to dig in box, the highest layer first and the nearest within it"""
        key = tuple(box)
        if key not in self.trenches:
            x0, y0, z0, x1, y1, z1 = box
            zs = [z for z in range(z0, z1 + 1) if (z - z0) % TRENCH_PERIOD < TRENCH_WIDTH]
            self.trenches[key] = [(x, y, z) for y in range(y1, y0 - 1, -1) for z in zs for x in range(x0, x1 + 1)]
        left = [c for c in self.trenches[key] if not self.empty(c) and c not in self.unreachable and not self.wasted(c)]
        if not left:
            return None
        top = max(c[1] for c in left)
        return self.stick(state, [c for c in left if c[1] == top])

    def committed(self, state: dict, targets) -> tuple[int, int, int] | None:
        """
        The target being dug until it is broken or given up on, then the next one from targets.
        A buried block's read can flicker between frames, this keeps the drone on it
        """
        if self.target_cell is not None and (self.target_cell in self.dug or self.target_cell in self.unreachable):
            self.target_cell = None
        if self.target_cell is None and targets:
            self.target_cell = self.stick(state, targets)
            self.target_block = base_name(self.memory.label(self.target_cell))
        return self.target_cell

    def stick(self, state: dict, cells) -> tuple[int, int, int] | None:
        """The cell already being worked on if it is still a choice, else the nearest, so moving doesn't flip the target"""
        return self.working_on if self.working_on in cells else self.nearest(state, cells)

    def dig(self, state: dict, cell, expect: str | None, box=None) -> dict:
        """
        Break cell through a face that is open, with the camera in front of it.
        Inside box a face only counts when the drone can get to it, otherwise a shaft gets dug down to the cell first
        """
        c = center(cell)

        def open_side(n) -> bool:
            if box is None:
                return self.empty(n)
            # above the top of the box is open air for digging, even before the camera has seen it
            if n[1] > box[4]:
                return True
            if not self.empty(n):
                return False
            if not (box[0] <= n[0] <= box[3] and box[2] <= n[2] <= box[5]):
                return True
            # a pocket under solid blocks is open but the camera can't get into it, the column above has to be clear
            return all(self.empty((n[0], y, n[2])) for y in range(n[1] + 1, box[4] + 1))

        if box is not None and not any(open_side(n) for n in neighbors(cell)):
            blocker = self.shaft_blocker(cell, box)
            if blocker is not None and blocker != cell:
                if blocker in self.unreachable or self.wasted(blocker):
                    # the shaft can't go on, so neither can the dig it was for
                    self.unreachable.add(cell)
                    return self.idle()
                return self.dig(state, blocker, None, box)
        # an exposed side first, the one toward the drone, the top only when no side is open
        normals = [d for d in (*self.toward(state, c, SIDES), UP) if open_side((cell[0] + d[0], cell[1] + d[1], cell[2] + d[2]))]
        if not normals and box is None:
            # a face whose neighbor the camera hasn't seen yet may still be open
            normals = [d for d in (*self.toward(state, c, SIDES), UP) if not self.solid((cell[0] + d[0], cell[1] + d[1], cell[2] + d[2]))]
        pose = self.work_pose(state, ("break", cell), [(cell, n) for n in normals])
        if pose is None:
            self.unreachable.add(cell)
            return self.idle()
        _, normal, via, point = pose
        return self.work_on(state, cell, "break", via, point, normal, expect=expect)

    @staticmethod
    def toward(state: dict, p, directions) -> list[tuple[int, int, int]]:
        """directions ordered by how near a step from p along each one ends to the drone, ties in the order given"""
        pos = state["pos"]
        return sorted(directions, key=lambda d: round(math.hypot(p[0] + d[0] - pos[0], p[2] + d[2] - pos[2]), 3))

    def work_pose(self, state: dict, key, faces) -> tuple[tuple, tuple, tuple, tuple] | None:
        """
        The pose to work one of faces from, each face a (block, normal) pair, as (block, normal, via, point): the camera
        at via, square-on to the face that normal points out of, aimed at point, the face's center.
        A side face's spots are straight out from it and level with it, then 45 degrees above it, a top face's straight
        overhead, then 45 degrees off a side. The first face with a spot that has room and a clear line to the face wins, and stays picked under
        key while it still has both. None when no face has one
        """
        held = self.poses.get(key)
        if held is not None and (held[0], held[1]) not in self.bad_faces and self.can_work_from(held[0], held[2], held[3]):
            return held
        for block, normal in faces:
            if (block, normal) in self.bad_faces:
                continue
            point = tuple(c + n * 0.49 for c, n in zip(center(block), normal))
            spots = [tuple(p + n * d for p, n in zip(point, normal)) for d in WORK_DIST]
            if normal[1] == 0:
                spots += [(point[0] + normal[0] * out, point[1] + up, point[2] + normal[2] * out) for out, up in SIDE_FALLBACKS]
            if normal == UP:
                s = TOP_SLANT / math.sqrt(2)
                spots += [(point[0] + d[0] * s, point[1] + s, point[2] + d[2] * s) for d in self.toward(state, point, SIDES)]
            for via in spots:
                if self.can_work_from(block, via, point):
                    self.poses[key] = (block, normal, via, point)
                    return self.poses[key]
        return None

    def can_work_from(self, block, via, point) -> bool:
        """
        Whether the drone fits with its camera at via and nothing it knows of stands between via and point.
        The world map's clearance isn't asked, it keeps the drone out of the trenches it digs
        """
        body = {
            (math.floor(via[0] + ox), math.floor(via[1] + oy), math.floor(via[2] + oz))
            for ox in (-DRONE_HALF, DRONE_HALF) for oz in (-DRONE_HALF, DRONE_HALF) for oy in (-VIA_EYE, DRONE_HEIGHT - VIA_EYE)
        }
        if any(self.solid(cell) for cell in body):
            return False
        steps = max(2, int(math.dist(via, point) / 0.25))
        for i in range(1, steps):
            p = [v + (q - v) * i / steps for v, q in zip(via, point)]
            cell = (math.floor(p[0]), math.floor(p[1]), math.floor(p[2]))
            if cell != tuple(block) and self.solid(cell):
                return False
        return True

    def solid(self, cell) -> bool:
        return self.believed(cell) != AIR

    def close_views(self, cells, box) -> list[tuple[tuple, tuple]]:
        """Up to two close views of each cell, from above or from sides that face open air"""
        lo, hi = box[:3], box[3:]
        views = []
        for cell in cells:
            sides = []
            c = center(cell)
            for d in ((0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)):
                n = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
                outside = not all(lo[i] <= n[i] <= hi[i] for i in range(3))
                if not outside and self.believed(n) != AIR:
                    continue
                if d[1] == 1:
                    via = (c[0] + 0.8, c[1] + GAP_VIEW_DIST, c[2])
                else:
                    via = (c[0] + d[0] * GAP_VIEW_DIST, c[1] + 0.2, c[2] + d[2] * GAP_VIEW_DIST)
                if self.free_at(via):
                    sides.append((via, c))
            views += sides[:2]
        return views

    def free_at(self, p) -> bool:
        """Whether the planner can fly to p, judged at p's own height, not wherever the drone is now"""
        saved = self.map.fly_y
        self.map.fly_y = p[1] - VIA_EYE
        try:
            return self.map.free(p[0], p[2])
        finally:
            self.map.fly_y = saved

    def share(self, cell, block: str) -> float:
        """The share of a cell's votes for block"""
        c = self.memory.cells.get(cell)
        return 0.0 if c is None else c.share(INDEX[block])


class BuildPlanner(JobPlanner):
    """
    Copy and build jobs. A copy reads its source box into memory and takes that read as the plan, a build loads the
    schematic it names. Then it clears what doesn't belong in the destination and places the plan bottom up,
    each block against a face of a block already there
    """

    def __init__(self, mask_ids: dict, depth_max: float = 64.0, reader=None, schematics: Path | None = None, core=None) -> None:
        super().__init__(mask_ids, depth_max, reader, schematics, core=core)
        self.plan: dict[tuple[int, int, int], str] | None = None
        self.read_schematic = None
        self.site_seen = False
        self.placed: dict[tuple[int, int, int], str] = {}
        self.target: tuple[int, int, int] | None = None
        self.skipped: set[tuple[int, int, int]] = set()
        self.gaps_checked = False
        self.gather_seen = False
        self.out_of: set[str] = set()
        # source cells being looked at again, the blocks already rechecked, and the source to destination offset
        self.recheck: list[tuple[int, int, int]] = []
        self.rechecked: set[str] = set()
        self.offset = (0, 0, 0)
        # how many times each support was given up on, it gets another try after each block placed, up to MAX_STRIKES
        self.strikes: Counter = Counter()

    def decide(self, state: dict) -> dict:
        job = state["job"]
        self.note_breaks(state)
        for event in state.get("events", []):
            kind = event.get("type")
            if kind == "place":
                self.placed[tuple(event["pos"])] = base_name(event["block"])
                self.retry_given_up()
            elif kind == "break":
                self.placed.pop(tuple(event["pos"]), None)
            elif kind == "place_failed" and self.target is not None:
                reason = event.get("reason", "")
                if reason.startswith("out of ") and "gather" in job and reason.removeprefix("out of ") not in self.unharvestable:
                    # gathering resupplies it
                    pass
                elif reason.startswith("out of "):
                    block = reason.removeprefix("out of ")
                    # every cell wanting that block would fail the same way. The first time, the read is the likelier
                    # mistake, so look at those source cells again up close
                    if block not in self.rechecked and job["kind"] == "copy" and "source" not in self.scanned_boxes:
                        self.start_recheck(block)
                    else:
                        self.out_of.add(block)
                elif "in the way" not in reason:
                    # the server refused the cell, a safe region or outside the job
                    self.skipped.add(self.target)
        self.note_job(job)
        if self.plan is None:
            if job["kind"] == "build":
                self.plan = self.load_plan(job)
            elif "source" in self.scanned_boxes:
                self.plan = self.scanned_plan(job)
            else:
                if self.views is None:
                    self.start_survey(state, job["source"])
                if self.views:
                    return self.survey(state)
                if not self.gaps_checked:
                    # cells under read blocks that no view reached get a close look through an open side
                    self.gaps_checked = True
                    self.views = self.gap_views(job["source"])
                    if self.views:
                        return self.survey(state)
                self.plan = self.read_plan(job, state)
                self.views = None
        if self.recheck:
            if self.views:
                return self.survey(state)
            self.finish_recheck()
        if not self.site_seen:
            if self.views is None:
                self.start_survey(state, job["dest"])
            if self.views:
                return self.survey(state)
            self.site_seen = True
        return self.build(state)

    def gaps(self, box) -> list[tuple[int, int, int]]:
        """Unseen cells right under a block read in the box"""
        lo, hi = box[:3], box[3:]
        read = self.memory.box(lo, hi)
        return sorted({(x, y - 1, z) for (x, y, z) in read if y - 1 >= lo[1] and (x, y - 1, z) not in read and self.unseen((x, y - 1, z))})

    def gap_views(self, box) -> list[tuple[tuple, tuple]]:
        return self.close_views(self.gaps(box), box)

    def start_recheck(self, block: str) -> None:
        """Forget what was read for the source cells planned as block, then look at each one up close"""
        self.rechecked.add(block)
        dx, dy, dz = self.offset
        cells = [(x - dx, y - dy, z - dz) for (x, y, z), b in self.plan.items() if b == block and not self.done((x, y, z))]
        # only cells with a close view start over, the rest keep what was read
        self.recheck = [c for c in cells if self.close_views([c], self.source)]
        for cell in self.recheck:
            if cell in self.memory.cells:
                self.memory.cells[cell].reset()
                self.memory.touch(cell)
        self.views = self.close_views(self.recheck, self.source)
        if not self.recheck:
            self.out_of.add(block)

    def finish_recheck(self) -> None:
        dx, dy, dz = self.offset
        for (x, y, z) in self.recheck:
            label = self.memory.label((x, y, z))
            dest = (x + dx, y + dy, z + dz)
            if label == AIR or label.startswith("<"):
                self.plan.pop(dest, None)
            else:
                self.plan[dest] = base_name(label)
        self.recheck = []
        self.views = None

    def read_plan(self, job: dict, state: dict) -> dict:
        """
        The source box as memory has it, moved onto the destination.
        A block the drone doesn't carry loses to a carried one with a fair share of the cell's votes, the materials it was
        given are evidence of what the build is made of.
        A cell no view reached under a read block takes the block above it, columns with a hole underneath are rare
        """
        carried = {name for name, count in state.get("inventory", []) if count > 0} if "gather" not in job else set()
        sx, sy, sz = job["source"][:3]
        lo, hi = job["source"][:3], job["source"][3:]
        dx, dy, dz = (d - s for d, s in zip(job["dest"][:3], (sx, sy, sz)))
        self.offset = (dx, dy, dz)
        self.source = job["source"]
        read = {}
        for c, label in self.memory.box(lo, hi).items():
            if label.startswith("<"):
                continue
            name = base_name(label)
            read[c] = (self.carried_alternative(c, carried) or name) if carried and name not in carried else name
        for cell in sorted(self.gaps(job["source"]), key=lambda c: -c[1]):
            for y in range(cell[1], lo[1] - 1, -1):
                below = (cell[0], y, cell[2])
                if below in read or not self.unseen(below):
                    break
                read[below] = read[(cell[0], y + 1, cell[2])]
        self.read_schematic = self.memory.schematic(lo, hi)
        return {(x + dx, y + dy, z + dz): name for (x, y, z), name in read.items()}

    def carried_alternative(self, cell, carried: set[str]) -> str | None:
        """The carried block with the most votes in cell, if it has at least CARRIED_SHARE of the leader's"""
        votes = self.memory.cells[cell].votes
        best = max((i for i in votes if CLASSES[i] in carried), key=votes.get, default=None)
        if best is not None and votes[best] > 0 and votes[best] >= CARRIED_SHARE * max(votes.values()):
            return CLASSES[best]
        return None

    def scanned_plan(self, job: dict) -> dict:
        """The scanned source moved onto the destination, any block, exact names"""
        sx, sy, sz = job["source"][:3]
        dx, dy, dz = (d - s for d, s in zip(job["dest"][:3], (sx, sy, sz)))
        self.offset = (dx, dy, dz)
        self.source = job["source"]
        return {
            (x + dx, y + dy, z + dz): name
            for (x, y, z), name in self.scanned.items()
            if name != AIR and all(job["source"][i] <= (x, y, z)[i] <= job["source"][i + 3] for i in range(3))
        }

    def load_plan(self, job: dict) -> dict:
        from quadcraft.schematic import load

        if self.schematics is None:
            raise ValueError(f"build job for {job['schematic']} needs the schematics folder, pass --schematics")
        schematic = load(self.schematics / job["schematic"])
        x0, y0, z0 = job["dest"][:3]
        w, h, l = schematic.size
        return {
            (x0 + x, y0 + y, z0 + z): base_name(schematic.get(x, y, z))
            for x in range(w) for y in range(h) for z in range(l)
            if base_name(schematic.get(x, y, z)) != AIR
        }

    def done(self, cell) -> bool:
        want = self.plan[cell]
        return self.placed.get(cell) == want or (cell not in self.placed and self.believed(cell) == want)

    def solid(self, cell) -> bool:
        return cell in self.placed or self.believed(cell) != AIR

    def retry_given_up(self) -> None:
        """A newly placed block changes what the drone can reach, so supports given up on get another try"""
        for cell in self.unreachable:
            self.strikes[cell] += 1
        self.unreachable = {c for c in self.unreachable if self.strikes[c] >= MAX_STRIKES}
        self.bad_faces.clear()

    def placement(self, state: dict, cell) -> tuple[tuple[int, int, int], str, tuple, tuple, tuple] | None:
        """A support for cell, its face to place against, and the work pose for that face as normal, via, and point"""
        faces = []
        for d in FACES:
            n = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
            if n not in self.unreachable and self.solid(n):
                faces.append((n, (-d[0], -d[1], -d[2])))
        pose = self.work_pose(state, ("place", cell), faces)
        if pose is None:
            return None
        support, normal, via, point = pose
        return support, FACES[tuple(-v for v in normal)], normal, via, point

    def build(self, state: dict) -> dict:
        lo, hi = self.job_box(state, "dest")
        for cell, label in sorted(self.memory.box(lo, hi).items(), key=lambda kv: -kv[0][1]):
            # a scanned name is exact, the memory only has the reader's classes
            name = self.scanned[cell] if cell in self.scanned and cell not in self.dug and cell not in self.placed else base_name(label)
            want = self.plan.get(cell)
            if cell in self.unreachable or (cell in self.placed and self.placed[cell] == want) or name == want:
                continue
            # top down, so nothing is left standing on a block about to go
            self.target = cell
            return self.dig(state, cell, name)
        todo = [c for c in self.plan if not self.done(c) and c not in self.skipped and self.plan[c] not in self.out_of]
        if "gather" in state["job"]:
            action = self.gather(state, todo)
            if action is not None:
                return action
        pos = state["pos"]
        # the lowest layer first and the nearest cell in it, checking viewpoints only until one has room
        for cell in sorted(todo, key=lambda c: (c[1], math.dist(center(c), pos))):
            found = self.placement(state, cell)
            if found is None:
                continue
            support, face, normal, via, point = found
            self.target = cell
            return self.work_on(state, support, "place", via, point, normal, face=face, place=self.plan[cell])
        return self.idle()

    def gather(self, state: dict, todo) -> dict | None:
        """
        Mine the gather box for blocks the rest of the build needs and the drone doesn't carry: ones it can see first,
        then trenches to uncover more. None when nothing is missing or the box has nothing left to give
        """
        have = Counter()
        for name, count in state.get("inventory", []):
            have[name] += count
        # mining a block the tier can't harvest gives nothing, so those wait for the drone's own stock
        need = Counter({b: n for b, n in (Counter(self.plan[c] for c in todo) - have).items() if b not in self.unharvestable})
        if not need:
            return None
        box = state["job"]["gather"]
        lo, hi = box[:3], box[3:]
        if "gather" in self.scanned_boxes:
            cell = self.committed(state, self.scanned_in(box, need))
            return self.dig(state, cell, self.target_block, box) if cell is not None else None
        if not self.gather_seen:
            if self.views is None:
                self.start_survey(state, box)
            if self.views:
                return self.survey(state)
            self.gather_seen = True
            self.views = None
        visible = [c for c, label in self.memory.box(lo, hi).items() if base_name(label) in need and c not in self.unreachable]
        cell = self.committed(state, visible)
        if cell is not None:
            return self.dig(state, cell, self.target_block, box)
        cell = self.next_dig(state, box)
        if cell is not None:
            # every trench block comes out whatever it is, a misread mustn't make the drone give up on it
            return self.dig(state, cell, None)
        return None

    @staticmethod
    def job_box(state: dict, key: str) -> tuple[tuple, tuple]:
        box = state["job"][key]
        return tuple(box[:3]), tuple(box[3:])


class DockPlanner(Planner):
    """Flies back to the charging station, the job ends once the drone docks on it"""

    def decide(self, state: dict) -> dict:
        # named as a goto goal, so the goto policy learns docking and flies it
        self.intent = dock_intent(state, state["job"]["station"])
        move, _ = self.fly(state, self.intent["aim"], standoff=0.0)
        return tool_action(move)


def episode_expert(task: str, job: dict | None, mask_ids: dict, reader, schematics: Path | None = None, core=None) -> Planner:
    """The job planner for an episode, every task with a teacher hands the drone a job"""
    if not job:
        raise ValueError(f"{task} hands the drone no job, there is no planner for it")
    return make_planner(job, mask_ids, reader, schematics, core=core)


def make_planner(job: dict, mask_ids: dict, reader, schematics: Path | None = None, core=None) -> Planner:
    """The planner for a job, reading the drone's core when given one"""
    if job["kind"] == "return_home":
        return DockPlanner(mask_ids, reader=reader, core=core)
    if job["kind"] in ("copy", "build"):
        return BuildPlanner(mask_ids, reader=reader, schematics=schematics, core=core)
    if job["kind"] == "goto":
        return GotoPlanner(mask_ids, reader=reader, core=core)
    raise ValueError(f"no planner for a {job['kind']} job")
