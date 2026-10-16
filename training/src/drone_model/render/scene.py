"""
Random studio scenes for the offline renderer, staged the way the mod's task/Studio.java stages them: one subject block
on a pedestal or as a 2x2x2 cube, random neighbors, a ground patch, a wall, a light, and the time of day, on flat, rough,
or cave terrain with trees and plants around. Camera views use Studio's distances, angles, and aim jitter
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass

import numpy as np

from ..perception.classes import alias
from ..perception.reader import CROPS, INDEX, NOT_CROP, OTHER
from .assets import BIOMES, FLUIDS, Assets, Biome, light_of, split
from .mesh import Placed

RADIUS = 24
VIEW_TRIES = 60
NEAR, FAR = 1.4, 10.0
LOW, HIGH = -15.0, 70.0
AIM_JITTER_YAW, AIM_JITTER_PITCH = 14.0, 10.0
MAX_NEIGHBORS = 8
CAMERA_CLEARANCE = 0.32
TERRAINS = ("flat", "rough", "cave")
TERRAIN_SHARE = (0.5, 0.3, 0.2)
LIGHTS = ("minecraft:torch", "minecraft:lantern", "minecraft:glowstone", "minecraft:sea_lantern", "minecraft:shroomlight", "minecraft:soul_lantern")
SURFACES = (
    ("minecraft:grass_block", "minecraft:dirt"), ("minecraft:grass_block", "minecraft:dirt"), ("minecraft:grass_block", "minecraft:dirt"),
    ("minecraft:sand", "minecraft:sandstone"), ("minecraft:podzol", "minecraft:dirt"), ("minecraft:coarse_dirt", "minecraft:dirt"),
    ("minecraft:stone", "minecraft:stone"), ("minecraft:gravel", "minecraft:stone"), ("minecraft:snow_block", "minecraft:dirt"),
    ("minecraft:moss_block", "minecraft:dirt"), ("minecraft:red_sand", "minecraft:terracotta"), ("minecraft:mud", "minecraft:dirt"),
)
CAVE_STONES = ("minecraft:stone", "minecraft:deepslate", "minecraft:tuff", "minecraft:andesite", "minecraft:diorite", "minecraft:granite")
PLANTS = ("minecraft:short_grass", "minecraft:short_grass", "minecraft:short_grass", "minecraft:fern", "minecraft:dandelion", "minecraft:poppy")
WOODS = ("oak", "birch", "spruce", "jungle", "acacia", "dark_oak", "cherry", "mangrove")


@dataclass
class Environment:
    """The lightmap and sky settings for a time of day, from the game's overworld timeline"""

    sky_factor: float
    sky_light_color: tuple[float, float, float]
    sky_color: tuple[float, float, float]
    fog_color: tuple[float, float, float]
    ambient: tuple[float, float, float] = (10 / 255, 10 / 255, 10 / 255)
    block_tint: tuple[float, float, float] = (1.0, 216 / 255, 140 / 255)
    block_factor: float = 1.4
    # the client's brightness option
    brightness: float = 0.5
    fog_range: tuple[float, float] = (192.0, 256.0)


@dataclass
class View:
    eye: np.ndarray
    forward: np.ndarray


@dataclass
class Scene:
    cells: dict[tuple[int, int, int], Placed]
    subject: str
    box: tuple[np.ndarray, np.ndarray]
    env: Environment
    ground: dict[tuple[int, int], int]
    ceiling: dict[tuple[int, int], int] | None
    far_ground: tuple[str, np.ndarray] | None


def environment(rng: random.Random, biome: Biome) -> Environment:
    """Studio's setTime spread: most scenes in the day, some at dawn or dusk, some at night"""
    roll = rng.random()
    night = 0.0 if roll < 0.6 else rng.uniform(0.2, 0.8) if roll < 0.8 else 1.0
    lerp = lambda a, b: tuple(x + (y - x) * night for x, y in zip(a, b))
    f = min(max(biome.temperature / 3.0, -1.0), 1.0)
    sky = colorsys.hsv_to_rgb(0.62222224 - f * 0.05, 0.5 + f * 0.1, 1.0)
    return Environment(
        sky_factor=1.0 + (0.24 - 1.0) * night,
        sky_light_color=lerp((1.0, 1.0, 1.0), (0.48, 0.48, 1.0)),
        sky_color=tuple(c * (1 - night) for c in sky),
        fog_color=lerp((0xC0 / 255, 0xD8 / 255, 1.0), (0.06 * 0xC0 / 255, 0.06 * 0xD8 / 255, 0.09)),
        block_factor=1.4 + rng.uniform(-0.05, 0.05),
    )


class Stager:
    def __init__(self, assets: Assets, textures, blocks: list[str]) -> None:
        self.assets = assets
        self.textures = textures
        self.blocks = blocks
        self._full_part: dict[int, bool] = {}
        self._full_state: dict[str, tuple | None] = {}
        self._shared: dict[tuple, Placed] = {}
        self.full_cubes = [b for b in blocks if self.full_cube_state(b) is not None]

    def placed(self, block: str, props: dict[str, str], parts, biome: Biome) -> Placed:
        name = split(block)[1]
        cls = INDEX.get(alias(block), OTHER)
        ripe = NOT_CROP
        if block in CROPS:
            ripe = 1 if int(props.get("age", 0)) == self.assets.max_age(block) else 0
        occludes = any(self.is_full(part) for part in parts)
        return Placed(block, props, parts, cls, ripe, light_of(block, props), occludes, self.assets.tints(block, props, biome),
                      fluid=FLUIDS.index(name) + 1 if name in FLUIDS else 0)

    def is_full(self, part) -> bool:
        if id(part) not in self._full_part:
            self._full_part[id(part)] = self._is_full(part)
        return self._full_part[id(part)]

    def _is_full(self, part) -> bool:
        box = part.boxes
        full = np.any(np.all(np.abs(box[:, 0]) < 1e-4, axis=1) & np.all(np.abs(box[:, 1] - 1) < 1e-4, axis=1))
        return bool(full and len(set(part.face.tolist())) == 6 and all(self.textures.kind.get(t, 0) == 0 for t in part.texture))

    def full_cube_state(self, block: str):
        """A state of block that fills its cell, the same one every time, None when no state does"""
        if block not in self._full_state:
            rng = random.Random(0)
            self._full_state[block] = None
            for _ in range(4):
                props, parts = self.assets.choose(block, rng)
                if parts and any(self.is_full(p) for p in parts):
                    self._full_state[block] = (props, parts)
                    break
        return self._full_state[block]

    def put(self, cells, pos, block: str, rng: random.Random, biome: Biome, full: bool = False) -> bool:
        """A random state of block at pos, both halves of a two-block one, False when no state draws anything"""
        if full:
            key = (block, id(biome))
            if key not in self._shared:
                state = self.full_cube_state(block)
                if state is None:
                    return False
                self._shared[key] = self.placed(block, *state, biome)
            cells[pos] = self._shared[key]
            return True
        else:
            for _ in range(4):
                props, parts = self.assets.choose(block, rng)
                if parts:
                    break
            else:
                return False
        cells[pos] = self.placed(block, props, parts, biome)
        if props.get("half") in ("lower", "upper"):
            other = dict(props, half="upper" if props["half"] == "lower" else "lower")
            other_parts = self.assets.parts(block, other, rng)
            if other_parts:
                above = (pos[0], pos[1] + (1 if props["half"] == "lower" else -1), pos[2])
                cells[above] = self.placed(block, other, other_parts, biome)
        return True

    def stage(self, subject: str, seed: int, terrain: str) -> Scene:
        rng = random.Random(seed)
        biome = rng.choice(BIOMES)
        env = environment(rng, biome)
        cells: dict[tuple[int, int, int], Placed] = {}
        surface, under = rng.choice(SURFACES)
        if terrain == "cave":
            surface = under = rng.choice(CAVE_STONES)
            env = Environment(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), block_factor=env.block_factor)
        ground = self.ground(rng, terrain)
        for (x, z), h in ground.items():
            self.put(cells, (x, h, z), surface, rng, biome, full=True)
            # deep enough to close the side a lower neighbor column shows
            low = min(ground.get((x + dx, z + dz), h) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            for y in range(h - 1, min(low, h - 1) - 1, -1):
                self.put(cells, (x, y, z), under, rng, biome, full=True)
        ceiling = None
        if terrain == "cave":
            ceiling = {}
            for (x, z), h in ground.items():
                rim = math.hypot(x, z) >= RADIUS - 2
                top = h + 1 if rim else h + rng.randint(4, 6)
                ceiling[(x, z)] = top
                # the rim walls reach past the highest ceiling, so the cave is closed all round
                for y in range(top, h + 9 if rim else top + 2):
                    self.put(cells, (x, y, z), surface, rng, biome, full=True)
                if not rim and rng.random() < 0.03:
                    self.put(cells, (x, top - 1, z), rng.choice(LIGHTS), rng, biome)
        else:
            self.scatter(cells, rng, biome, ground, surface)

        base_h = ground[(0, 0)]
        box = self.subject(cells, rng, biome, subject, base_h)
        self.backdrop(cells, rng, biome, ground, ceiling)
        self.neighbors(cells, rng, biome, box)
        # under every column, so no camera looks at it from below
        far = None if terrain == "cave" else (surface, min(ground.values()))
        return Scene(cells, subject, box, env, ground, ceiling, far)

    def ground(self, rng: random.Random, terrain: str) -> dict[tuple[int, int], int]:
        heights = {}
        bumps = [(rng.uniform(-RADIUS, RADIUS), rng.uniform(-RADIUS, RADIUS), rng.uniform(1, 4), rng.uniform(3, 9)) for _ in range(rng.randint(3, 8))] if terrain == "rough" else []
        for x in range(-RADIUS, RADIUS + 1):
            for z in range(-RADIUS, RADIUS + 1):
                h = sum(a * math.exp(-((x - bx) ** 2 + (z - bz) ** 2) / (2 * s * s)) for bx, bz, a, s in bumps)
                heights[(x, z)] = int(round(h))
        return heights

    def scatter(self, cells, rng, biome, ground, surface: str) -> None:
        """Plants on grass and a few trees, away from the subject"""
        for (x, z), h in ground.items():
            if abs(x) <= 2 and abs(z) <= 2:
                continue
            if surface == "minecraft:grass_block" and rng.random() < 0.12:
                self.put(cells, (x, h + 1, z), rng.choice(PLANTS), rng, biome)
        for _ in range(rng.randint(0, 6)):
            r, a = rng.uniform(5, RADIUS - 3), rng.uniform(0, 2 * math.pi)
            x, z = int(r * math.cos(a)), int(r * math.sin(a))
            wood = rng.choice(WOODS)
            h = ground[(x, z)]
            trunk = rng.randint(4, 6)
            for y in range(h + 1, h + 1 + trunk):
                self.put(cells, (x, y, z), f"minecraft:{wood}_log", rng, biome, full=True)
            for dx in range(-2, 3):
                for dy in range(-1, 2):
                    for dz in range(-2, 3):
                        pos = (x + dx, h + trunk + dy, z + dz)
                        if pos not in cells and dx * dx + dy * dy * 2 + dz * dz <= 6 and rng.random() < 0.9:
                            self.put(cells, pos, f"minecraft:{wood}_leaves", rng, biome)

    def subject(self, cells, rng, biome, subject: str, base_h: int) -> tuple[np.ndarray, np.ndarray]:
        """Studio's placeSubject: fluids in a basin, full cubes sometimes as a 2x2x2, the rest on a pedestal 0 to 2 high"""
        if split(subject)[1] in FLUIDS:
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    rim = abs(dx) == 2 or abs(dz) == 2
                    self.put(cells, (dx, base_h + 1, dz), rng.choice(self.full_cubes) if rim else subject, rng, biome, full=rim)
            return np.array([0, base_h + 1, 0], dtype=np.float32), np.array([1, base_h + 2, 1], dtype=np.float32)
        if self.full_cube_state(subject) is not None and rng.random() < 0.3:
            for dx in range(2):
                for dz in range(2):
                    for dy in range(1, 3):
                        self.put(cells, (dx, base_h + dy, dz), subject, rng, biome, full=True)
            return np.array([0, base_h + 1, 0], dtype=np.float32), np.array([2, base_h + 3, 2], dtype=np.float32)
        pedestal = rng.choice(self.full_cubes)
        height = rng.randint(0, 2)
        for h in range(height):
            self.put(cells, (0, base_h + 1 + h, 0), pedestal, rng, biome, full=True)
        top = base_h + 1 + height
        if not self.put(cells, (0, top, 0), subject, rng, biome):
            raise ValueError(f"no state of {subject} draws anything")
        two = cells[(0, top, 0)].props.get("half") in ("lower", "upper")
        return np.array([0, top, 0], dtype=np.float32), np.array([1, top + (2 if two else 1), 1], dtype=np.float32)

    def backdrop(self, cells, rng, biome, ground, ceiling) -> None:
        if rng.random() < 0.5:
            patch = rng.choice(self.full_cubes)
            reach = rng.randint(2, 5)
            for dx in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    pos = (dx, ground[(dx, dz)], dz)
                    self.put(cells, pos, patch, rng, biome, full=True)
                    plant = (dx, ground[(dx, dz)] + 1, dz)
                    if plant in cells and not cells[plant].occludes and (abs(dx) > 1 or abs(dz) > 1):
                        del cells[plant]
        if rng.random() < 0.4:
            wall = rng.choice(self.full_cubes)
            side = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            along = (-side[1], side[0])
            dist, half, height = rng.randint(2, 4), rng.randint(1, 3), rng.randint(2, 4)
            for i in range(-half, half + 1):
                x, z = side[0] * dist + along[0] * i, side[1] * dist + along[1] * i
                g = ground[(x, z)]
                for y in range(g + 1, g + 1 + height):
                    if ceiling is not None and y >= ceiling[(x, z)]:
                        break
                    self.put(cells, (x, y, z), wall, rng, biome, full=True)
        if rng.random() < 0.3:
            dx, dz = rng.randint(-2, 2), rng.randint(-2, 2)
            if dx or dz:
                pos = (dx, ground[(dx, dz)] + 1, dz)
                if pos not in cells or not cells[pos].occludes:
                    self.put(cells, pos, rng.choice(LIGHTS), rng, biome)

    def neighbors(self, cells, rng, biome, box) -> None:
        lo, hi = box
        for _ in range(rng.randint(0, MAX_NEIGHBORS)):
            reach = 1 if rng.random() < 0.6 else 2
            cell = (
                int(math.floor(lo[0] + rng.random() * (hi[0] - lo[0]) + rng.randint(-reach, reach))),
                int(lo[1]) + rng.randint(0, max(1, int(math.ceil(hi[1] - lo[1])))) - 1,
                int(math.floor(lo[2] + rng.random() * (hi[2] - lo[2]) + rng.randint(-reach, reach))),
            )
            inside = all(lo[i] <= cell[i] < hi[i] for i in range(3))
            if inside or (cell in cells and cells[cell].occludes):
                continue
            self.put(cells, cell, rng.choice(self.blocks), rng, biome)

    def views(self, scene: Scene, rng: random.Random):
        """Camera poses in Studio's ranges, in the open and inside the scene, aimed near the subject with jitter"""
        lo, hi = scene.box
        center = (lo + hi) / 2
        for _ in range(VIEW_TRIES):
            dist = math.exp(math.log(NEAR) + rng.random() * (math.log(FAR) - math.log(NEAR))) + (hi[0] - lo[0]) / 2
            heading = rng.random() * 2 * math.pi
            elevation = math.radians(LOW + rng.random() * (HIGH - LOW))
            eye = center + dist * np.array([math.cos(elevation) * math.sin(heading), math.sin(elevation), math.cos(elevation) * math.cos(heading)])
            cx, cz = int(math.floor(eye[0])), int(math.floor(eye[2]))
            if abs(cx) > RADIUS - 2 or abs(cz) > RADIUS - 2:
                continue
            if eye[1] < scene.ground[(cx, cz)] + 2.2:
                continue
            if scene.ceiling is not None and eye[1] > scene.ceiling[(cx, cz)] - 0.6:
                continue
            if self.blocked(scene.cells, eye):
                continue
            d = center - eye
            yaw = math.atan2(d[0], d[2]) + math.radians(rng.uniform(-1, 1) * AIM_JITTER_YAW)
            pitch = math.atan2(d[1], math.hypot(d[0], d[2])) + math.radians(rng.uniform(-1, 1) * AIM_JITTER_PITCH)
            pitch = max(min(pitch, math.radians(89)), math.radians(-89))
            forward = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
            yield View(eye, forward)

    def blocked(self, cells, eye: np.ndarray) -> bool:
        """Whether anything in a block within the camera's clearance overlaps it, or the camera is in a fluid"""
        lo, hi = eye - CAMERA_CLEARANCE, eye + CAMERA_CLEARANCE
        for x in range(int(math.floor(lo[0])), int(math.floor(hi[0])) + 1):
            for y in range(int(math.floor(lo[1])), int(math.floor(hi[1])) + 1):
                for z in range(int(math.floor(lo[2])), int(math.floor(hi[2])) + 1):
                    p = cells.get((x, y, z))
                    if p is None:
                        continue
                    if p.fluid:
                        return True
                    for part in p.parts:
                        b0 = part.boxes[:, 0] + (x, y, z)
                        b1 = part.boxes[:, 1] + (x, y, z)
                        if np.any(np.all((b0 < hi) & (b1 > lo), axis=1)):
                            return True
        return False
