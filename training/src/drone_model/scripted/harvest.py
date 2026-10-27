"""
The harvest planner's field work: survey the field, break ripe crops, replant their plots, and sweep for bare farmland.
It knows only the drone's own pose and what its camera has shown, through the drone's perception core
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..perception.reader import CLASSES, SKY
from .base import (
    AIM_TOLERANCE,
    SURVEY_LIMIT,
    TURN_GAIN,
    Planner,
    aim_angle,
    aim_errors,
    carried,
    place_viewpoint,
)

FIELD_VIEW_HEIGHT = 5.0
FIELD_VIEW_PITCH = 60.0
# steps spent looking for one plot on a sweep, and sweeps before the expert only watches for ripe crops
SWEEP_LIMIT = 60
MAX_SWEEPS = 2
# aimed frames read at a plot before the sweep decides whether it is bare farmland
PLOT_READS = 3


class HarvestExpert(Planner):
    """
    Works a harvest job's field from its voxel memory, the block reader's classes and ripeness back-projected into cells.
    Looks the field over from above, then harvests one cycle at a time, break a ripe crop and replant its plot.
    With nothing ripe left it sweeps the plots not known to hold a crop, looks down at each one, and plants where the
    reader sees bare farmland
    """

    def __init__(self, mask_ids: dict, depth_max: float = 64.0, reader=None, core=None, schematics=None) -> None:
        if reader is None:
            raise ValueError("the harvest expert judges ripeness with the block reader, pass one")
        super().__init__(mask_ids, depth_max, reader, core=core, schematics=schematics)
        self.views: list[tuple[float, float, float]] | None = None
        self.view_steps = 0
        self.dwell = 0
        self.planted: set[tuple[int, int, int]] = set()
        self.broken: set[tuple[int, int, int]] = set()
        self.refused: set[tuple[int, int, int]] = set()
        # the crop being harvested and replanted
        self.cycle: tuple[int, int, int] | None = None
        # plots left to check on this sweep, and how many sweeps are done
        self.sweep: list[tuple[int, int, int]] | None = None
        self.sweep_steps = 0
        self.sweeps = 0
        self.plot_reads: Counter[str] = Counter()

    def field(self, state: dict) -> list[int]:
        return state["job"]["region"]

    def crop(self, state: dict) -> str:
        return state["job"].get("crop", "minecraft:wheat")

    def farm_y(self, state: dict) -> int | None:
        """The farmland's height, where most of the field's farmland cells are"""
        x0, y0, z0, x1, y1, z1 = self.field(state)
        ys = Counter(c[1] for c, label in self.memory.box((x0, y0 - 1, z0), (x1, y1, z1)).items() if label == "minecraft:farmland")
        return ys.most_common(1)[0][0] if ys else None

    def ripe_cells(self, state: dict) -> list[tuple[int, int, int]]:
        farm_y = self.farm_y(state)
        if farm_y is None:
            return []
        x0, _, z0, x1, _, z1 = self.field(state)
        ripe = self.crop(state) + " ripe"
        # crops grow one above the farmland
        cells = self.memory.box((x0, farm_y + 1, z0), (x1, farm_y + 1, z1))
        return [c for c, label in cells.items() if label == ripe and c not in self.refused and c not in self.unreachable]

    def decide(self, state: dict) -> dict:
        for event in state.get("events", []):
            pos = tuple(event.get("pos", ()))
            if event.get("type") == "break":
                self.planted.discard(pos)
                self.broken.add(pos)
            elif event.get("type") == "place":
                self.planted.add(pos)
                self.broken.discard(pos)
            elif event.get("type") == "break_failed" and "ripe" in event.get("reason", "") and self.working_on:
                self.refused.add(self.working_on)
        if self.views is None or self.views:
            return self.survey(state)
        seed = state["job"].get("seed", "minecraft:wheat_seeds")
        has_seed = bool(carried(state, {seed}))
        if self.cycle is None:
            ripe = self.ripe_cells(state)
            self.cycle = self.nearest(state, ripe) if ripe else None
        if self.cycle is not None:
            action = self.harvest(state, seed, has_seed)
            if action is not None:
                return action
        if has_seed and self.sweeps < MAX_SWEEPS:
            if self.sweep is None:
                self.sweep = self.sweep_plots(state)
            if self.sweep:
                return self.check_plot(state, seed)
            self.sweep = None
            self.sweeps += 1
        # look the field over again for crops that ripened or were missed
        self.views = None
        return self.survey(state)

    def harvest(self, state: dict, seed: str, has_seed: bool) -> dict | None:
        """Break the cycle's crop, then plant its plot. None once the cycle is over"""
        crop = self.cycle
        plot = (crop[0], crop[1] - 1, crop[2])
        if crop in self.planted or crop in self.unreachable or plot in self.unreachable or crop in self.refused:
            self.cycle = None
            return None
        if crop not in self.broken:
            return self.work_on(state, crop, "break", standoff=2.0, hover=2.0, expect="minecraft:wheat")
        if not has_seed:
            self.cycle = None
            return None
        return self.plant(state, plot, seed)

    def plant(self, state: dict, plot, seed: str) -> dict:
        point = (plot[0] + 0.5, plot[1] + 0.95, plot[2] + 0.5)
        via = place_viewpoint(state, (plot[0], plot[1] + 1, plot[2]), (0, -1, 0))
        return self.work_on(state, plot, "place", standoff=0.0, hover=0.0, point=point, face="up", place=seed, via=via)

    def sweep_plots(self, state: dict) -> list[tuple[int, int, int]]:
        """The plots of the field the memory doesn't know a crop on, row by row with every other row reversed"""
        x0, y0, z0, x1, y1, z1 = self.field(state)
        farm_y = self.farm_y(state)
        y = farm_y if farm_y is not None else y0
        crop = self.crop(state)
        rows = [[(x, y, z) for z in range(z0, z1 + 1)] for x in range(x0, x1 + 1)]
        ordered = [c for i, row in enumerate(rows) for c in (row if i % 2 == 0 else row[::-1])]
        return [c for c in ordered if (c[0], y + 1, c[2]) not in self.planted and not self.memory.label((c[0], y + 1, c[2])).startswith(crop)]

    def center_class(self) -> str | None:
        """What the reader sees under the crosshair, the most common class in the middle of the frame"""
        if self.classes is None:
            return None
        patch = self.core.crosshair(5)[2].ravel()
        patch = patch[patch != SKY]
        return CLASSES[int(np.bincount(patch).argmax())] if len(patch) else None

    def check_plot(self, state: dict, seed: str) -> dict:
        """Look down at the next plot of the sweep, plant it if the reader reads bare farmland there, else move on"""
        plot = self.sweep[0]
        crop = (plot[0], plot[1] + 1, plot[2])
        point = (plot[0] + 0.5, plot[1] + 0.95, plot[2] + 0.5)
        via = place_viewpoint(state, crop, (0, -1, 0))
        self.sweep_steps += 1
        move, arrived = self.fly_to_view(state, via, point)
        aimed = arrived and aim_angle(state, point) < AIM_TOLERANCE
        if aimed and (c := self.center_class()) is not None:
            self.plot_reads[c] += 1
        # one frame can misread bare farmland, the plot goes by the majority of PLOT_READS,
        # or by the memory, many frames from many angles, when that has it bare
        seen = self.plot_reads.most_common(1)[0][0] if self.plot_reads.total() >= PLOT_READS else None
        if seen is not None and self.memory.label(plot) == "minecraft:farmland" and not self.memory.label(crop).startswith(self.crop(state)):
            seen = "minecraft:farmland"
        done = crop in self.planted or plot in self.unreachable or self.sweep_steps > SWEEP_LIMIT
        if done or (seen is not None and seen != "minecraft:farmland"):
            self.sweep.pop(0)
            self.sweep_steps = 0
            self.plot_reads.clear()
        elif seen == "minecraft:farmland":
            return self.plant(state, plot, seed)
        return self.view(state, move, via, point)

    def survey(self, state: dict) -> dict:
        """Look the field over from a little above each side, so every cell is seen at a steep angle"""
        x0, y0, z0, x1, y1, z1 = self.field(state)
        cx, cz = (x0 + x1 + 1) / 2, (z0 + z1 + 1) / 2
        if self.views is None:
            half = max(x1 - x0, z1 - z0) / 2
            y = y1 + FIELD_VIEW_HEIGHT
            self.views = [(cx + dx * half, y, cz + dz * half) for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1))]
        view = self.views[0]
        self.view_steps += 1
        move, _ = self.fly(state, view, standoff=0.0, look_down=FIELD_VIEW_PITCH)
        if math.dist((view[0], view[2]), (state["pos"][0], state["pos"][2])) < 0.8 or self.view_steps > SURVEY_LIMIT:
            yaw_error, pitch_error, _ = aim_errors(state, (cx, y0 + 1, cz))
            move = [0.0, 0.0, float(np.clip(view[1] - state["pos"][1], -1, 1)), np.clip(yaw_error * TURN_GAIN, -1, 1), np.clip(pitch_error * TURN_GAIN, -1, 1)]
            self.dwell_on(state, (cx, y0 + 1, cz))
        return self.view(state, move, view, (cx, y0 + 1, cz))
