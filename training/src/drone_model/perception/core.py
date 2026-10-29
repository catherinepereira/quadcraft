"""
The drone's perception core: the block reader, the voxel memory of which block fills each cell, and the world map of
column heights and landmarks. A drone keeps one core across all its jobs and every frame goes through it.
A job's planner and the learned skills read from it, they don't keep maps of their own
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .classes import alias
from .memory import AIR, Cell, Change, VoxelMemory, base_name
from .reader import CLASSES, INDEX, OTHER, SKY, class_lut, mask_lut
from .worldmap import WorldMap, camera_axes

# a scanned cell counts as this many reads, far more than frames add
SCAN_VOTES = 1000.0


def read_scans(job: dict, schematics: Path | None) -> dict[str, dict[tuple[int, int, int], str]]:
    """Block states of each box the server scanned for the job, by box name and then by cell"""
    if not job.get("scan"):
        return {}
    from quadcraft.schematic import load

    if schematics is None:
        raise ValueError("a scanned job needs the schematics folder, pass --schematics")
    out = {}
    for key, rel in job["scan"].items():
        schematic = load(schematics / rel)
        x0, y0, z0 = job[key][:3]
        w, h, l = schematic.size
        out[key] = {(x0 + x, y0 + y, z0 + z): schematic.get(x, y, z) for x in range(w) for y in range(h) for z in range(l)}
    return out


class DroneCore:
    def __init__(self, mask_ids: dict, depth_max: float = 64.0, reader=None, schematics: Path | None = None) -> None:
        """
        With a reader (a drone_model.perception.reader.Reader) the core sees only what the block reader makes of the camera
        image and depth, without one it reads the mod's mask stream, which is ground truth. schematics is the game's
        schematics folder, where the server leaves the boxes it scans for a job in scan perception
        """
        self.schematics = schematics
        # scan perception: the job's scanned boxes as block states by box name and cell, their block names by cell,
        # and the files they came from
        self.scans: dict[str, dict[tuple[int, int, int], str]] = {}
        self.scanned: dict[tuple[int, int, int], str] = {}
        self.scan_files: dict[str, str] = {}
        self.depth_max = depth_max
        self.reader = reader
        self.map = WorldMap(mask_ids)
        self.memory = VoxelMemory()
        self.reader_lut = mask_lut(mask_ids) if reader is not None else None
        self.class_lut = class_lut(mask_ids) if reader is None else None
        # the reader's class per pixel for the current frame, and the memory cells that changed with it
        self.classes: np.ndarray | None = None
        # the latest frame as planners read it, with the mask in place
        self.obs: dict | None = None
        self.changes: list[Change] = []

    def see(self, state: dict, obs: dict) -> dict:
        """
        Takes one frame into the memory and the map, and returns it with the mask planners read: the reader's classes as
        mask ids, or the mod's own mask
        """
        if self.reader is not None:
            classes = self.reader.read(obs["rgb"], obs["depth"])
            obs = {**obs, "mask": self.reader_lut[classes]}
        else:
            classes = self.class_lut[np.minimum(obs["mask"].astype(np.int64), len(self.class_lut) - 1)]
        self.classes = classes
        self.take_scans(state.get("job") or {})
        events = state.get("events", [])
        self.changes = self.memory.apply_events(events) + self.memory.observe(state, obs["depth"], classes, self.depth_max)
        self.map.update(state, obs["depth"], obs["mask"], self.depth_max)
        for event in events:
            if event.get("type") == "break":
                self.map.forget(tuple(event["pos"]))
        self.obs = obs
        return obs

    def crosshair(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The mask, depth, and reader classes of the size by size pixels in the middle of the latest frame"""
        h, w = self.obs["depth"].shape
        middle = (slice(h // 2 - size // 2, h // 2 - size // 2 + size), slice(w // 2 - size // 2, w // 2 - size // 2 + size))
        return self.obs["mask"][middle], self.obs["depth"][middle], self.classes[middle]

    def aim(self, state: dict) -> dict | None:
        """
        What the crosshair rests on as the drone's camera reads it: the cell and face the center ray reaches through depth,
        how far, and the class the frame shows there. None before the first frame or when the ray reaches nothing in range
        """
        if self.obs is None:
            return None
        h, w = self.obs["depth"].shape
        middle = (slice(h // 2 - 1, h // 2 + 1), slice(w // 2 - 1, w // 2 + 1))
        dist = float(np.median(self.obs["depth"][middle]))
        if dist >= self.depth_max - 0.01:
            return None
        eye, forward, _, _ = camera_axes(state)
        point = eye + forward * dist
        # step into the surface so the hit lands in the block that owns the face
        cell = np.floor(point + forward * 0.15).astype(np.int64)
        local = point - cell
        axis = int(np.argmin(np.minimum(local, 1 - local)))
        face = (("west", "east"), ("down", "up"), ("north", "south"))[axis][int(local[axis] > 0.5)]
        classes = self.classes[middle].ravel()
        return {"pos": tuple(int(v) for v in cell), "face": face, "dist": dist, "block": CLASSES[int(np.bincount(classes).argmax())]}

    def take_scans(self, job: dict) -> None:
        """Reads the boxes the server scanned for a job into the memory once, each cell as a certain read"""
        files = job.get("scan") or {}
        if files == self.scan_files:
            return
        self.scan_files = dict(files)
        self.scans = read_scans(job, self.schematics)
        self.scanned = {}
        for cells in self.scans.values():
            for cell, block in cells.items():
                name = base_name(block)
                self.scanned[cell] = name
                self.memory.cells.setdefault(cell, Cell()).reset(SKY if name == AIR else INDEX.get(alias(name), OTHER), SCAN_VOTES)
                self.memory.touch(cell)
