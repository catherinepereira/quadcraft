"""
Block models from the game's client jar: blockstates resolved to model parts, each baked once into numpy quads with
texture layers, tint layers, cull faces, and the boxes the label pass draws. Tint and light values follow the game's
BlockColors, BlockTintSources, and Blocks (26.3)
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

LOOM = Path(os.path.expanduser("~/.gradle/caches/fabric-loom"))
MOD_ASSETS = Path(__file__).resolve().parents[4] / "mod" / "src" / "main" / "resources"

# down, up, north, south, west, east, as unit vectors
DIRS = ("down", "up", "north", "south", "west", "east")
DIR_VEC = np.array([[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0]], dtype=np.int32)
# the game's per-face shading, ModelBlockRenderer
DIR_SHADE = np.array([0.5, 1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float32)

# a face's corners as top left, bottom left, bottom right, top right seen from outside, from the element's from (0) and to (1)
CORNERS = {
    "down": [(0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1)],
    "up": [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    "north": [(1, 1, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0)],
    "south": [(0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1)],
    "west": [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)],
    "east": [(1, 1, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0)],
}

FLUIDS = ("water", "lava")
# a source block's surface with no fluid above it, LiquidBlockRenderer
FLUID_HEIGHT = 8 / 9

# light emission, from Blocks.java, "lit" ones only while lit=true
LIGHT = {
    "lava": 15, "brown_mushroom": 1, "torch": 14, "wall_torch": 14, "fire": 15, "soul_fire": 10, "soul_torch": 10,
    "soul_wall_torch": 10, "copper_torch": 14, "copper_wall_torch": 14, "glowstone": 15, "nether_portal": 11,
    "jack_o_lantern": 15, "glow_lichen": 7, "enchanting_table": 7, "brewing_stand": 1, "lava_cauldron": 15,
    "end_portal_frame": 1, "dragon_egg": 1, "beacon": 15, "sea_lantern": 15, "end_rod": 14, "magma_block": 3,
    "sea_pickle": 6, "lantern": 15, "soul_lantern": 10, "shroomlight": 15, "crying_obsidian": 10, "amethyst_cluster": 5,
    "large_amethyst_bud": 4, "medium_amethyst_bud": 2, "small_amethyst_bud": 1, "sculk_sensor": 1, "sculk_catalyst": 6,
    "cave_vines": 14, "cave_vines_plant": 14, "ochre_froglight": 15, "verdant_froglight": 15,
    "pearlescent_froglight": 15, "firefly_bush": 2, "respawn_anchor": 15, "copper_lantern": 15,
}
LIT_LIGHT = {"furnace": 13, "redstone_ore": 9, "deepslate_redstone_ore": 9, "redstone_torch": 7, "redstone_wall_torch": 7,
             "redstone_lamp": 15, "smoker": 13, "blast_furnace": 13, "campfire": 15, "soul_campfire": 10, "candle_cake": 3}

GRASS = {"fern", "short_grass", "potted_fern", "bush", "large_fern", "tall_grass", "sugar_cane", "grass_block"}
FOLIAGE = {"oak_leaves", "jungle_leaves", "acacia_leaves", "dark_oak_leaves", "vine", "mangrove_leaves"}
WATER_TINTED = {"water", "water_cauldron", "bubble_column"}
CONSTANT_TINT = {"spruce_leaves": 0x619961, "birch_leaves": 0x80A755, "lily_pad": 0x208030, "attached_melon_stem": 0xE0C71C,
                 "attached_pumpkin_stem": 0xE0C71C}
# blocks whose first tint layer is blank and second is grass
SECOND_LAYER_GRASS = {"pink_petals", "wildflowers"}


def client_jar() -> Path:
    jars = sorted(LOOM.glob("*/minecraft-client.jar"))
    if not jars:
        raise SystemExit(f"no minecraft-client.jar under {LOOM}, build the mod once so Loom fetches the game")
    return jars[-1]


@dataclass
class Biome:
    temperature: float
    downfall: float
    water: int = 0x3F76E4
    grass: int | None = None
    foliage: int | None = None


BIOMES = (
    Biome(0.8, 0.4),
    Biome(0.7, 0.8),
    Biome(0.6, 0.6),
    Biome(0.25, 0.8, 0x3D57D6),
    Biome(0.0, 0.5, 0x3938C9),
    Biome(2.0, 0.0, 0x32A598),
    Biome(0.95, 0.9, 0x14A2C5),
    Biome(0.8, 0.9, 0x617B64, 0x6A7039, 0x6A7039),
    Biome(2.0, 0.0, 0x3F76E4, 0x90814D, 0x9E814D),
    Biome(0.5, 0.8, 0x5DB7EF, 0xB6DB61, 0xB6DB61),
    Biome(0.8, 0.9, 0x3A7A6A, None, 0x8DB127),
    Biome(0.5, 0.5, 0x43D5EE),
)


@dataclass
class Baked:
    """One model part of a block, positions in blocks (0 to 1) with four corners per quad"""

    pos: np.ndarray  # (Q, 4, 3) float32
    uv: np.ndarray  # (Q, 4, 2) float32, 0 to 1 within the texture
    texture: list[str]  # (Q) texture names
    tint: np.ndarray  # (Q) int, -1 for none
    cull: np.ndarray  # (Q) int, the direction whose neighbor hides the face, -1 for none
    face: np.ndarray  # (Q) int, the direction the face looks
    shade: np.ndarray  # (Q) float, the game's per-direction shading, 1 for unshaded faces
    emission: np.ndarray  # (Q) int, the element's own light level
    boxes: np.ndarray  # (B, 2, 3) float32, label boxes
    ao: bool = True


@dataclass
class BlockKind:
    name: str
    variants: dict | None = None
    multipart: list | None = None


class Assets:
    """The jar's block assets with the mod's own on top, read lazily and cached"""

    def __init__(self, jar: Path | None = None) -> None:
        self.jar = zipfile.ZipFile(jar or client_jar())
        self.names = set(self.jar.namelist())
        self.colormaps = {k: self.image(f"minecraft:colormap/{k}") for k in ("grass", "foliage", "dry_foliage")}

    def read(self, namespace: str, path: str) -> bytes:
        local = MOD_ASSETS / "assets" / namespace / path
        if local.exists():
            return local.read_bytes()
        return self.jar.read(f"assets/{namespace}/{path}")

    def exists(self, namespace: str, path: str) -> bool:
        return (MOD_ASSETS / "assets" / namespace / path).exists() or f"assets/{namespace}/{path}" in self.names

    @cache
    def json(self, namespace: str, path: str) -> dict:
        return json.loads(self.read(namespace, path))

    @cache
    def image(self, name: str) -> np.ndarray:
        """A texture as RGBA uint8, all frames stacked top to bottom for an animated one"""
        ns, path = split(name)
        return np.asarray(Image.open(io.BytesIO(self.read(ns, f"textures/{path}.png"))).convert("RGBA"))

    def blockstate(self, block: str) -> dict | None:
        ns, path = split(block)
        if not self.exists(ns, f"blockstates/{path}.json"):
            return None
        return self.json(ns, f"blockstates/{path}.json")

    @cache
    def model(self, name: str) -> dict:
        """A model with its parents merged in: textures resolved to texture names, the nearest elements"""
        ns, path = split(name)
        raw = self.json(ns, f"models/{path}.json")
        merged = {"textures": {}, "elements": None, "ambientocclusion": True}
        if "parent" in raw and not raw["parent"].startswith("builtin/"):
            parent = self.model(raw["parent"])
            merged = {"textures": dict(parent["textures"]), "elements": parent["elements"], "ambientocclusion": parent["ambientocclusion"]}
        merged["textures"].update(raw.get("textures", {}))
        if "elements" in raw:
            merged["elements"] = raw["elements"]
        if "ambientocclusion" in raw:
            merged["ambientocclusion"] = raw["ambientocclusion"]
        return merged

    def texture_of(self, model: dict, ref: str) -> str | None:
        seen = 0
        while seen < 10:
            if isinstance(ref, dict):
                ref = ref["sprite"]
            if not ref.startswith("#"):
                break
            ref = model["textures"].get(ref[1:], ref)
            seen += 1
        if ref.startswith("#"):
            return None
        return ref if ":" in ref else "minecraft:" + ref

    def texture_names(self, block: str) -> set[str]:
        """Every texture any state of block can draw with"""
        if split(block)[1] in FLUIDS:
            return {f"minecraft:block/{split(block)[1]}_still", f"minecraft:block/{split(block)[1]}_flow"}
        k = self.kind(block)
        if k is None:
            return set()
        applies = list(k.variants.values()) if k.variants is not None else [part["apply"] for part in k.multipart]
        out = set()
        for apply in applies:
            for a in apply if isinstance(apply, list) else [apply]:
                model = self.model(a["model"])
                if model["elements"]:
                    out |= {t for t in (self.texture_of(model, r) for r in model["textures"].values()) if t}
        return out

    def renderable(self, block: str) -> bool:
        if split(block)[1] in FLUIDS:
            return True
        k = self.kind(block)
        if k is None:
            return False
        apply = next(iter(k.variants.values())) if k.variants is not None else k.multipart[0]["apply"]
        apply = apply[0] if isinstance(apply, list) else apply
        return self.model(apply["model"])["elements"] is not None

    @cache
    def bake(self, model_name: str, x: int = 0, y: int = 0, uvlock: bool = False) -> Baked | None:
        """A model part's quads and label boxes after its blockstate rotation, None for a model with no elements"""
        model = self.model(model_name)
        if not model["elements"]:
            return None
        rot = variant_rotation(x, y)
        pos, uv, tex, tint, cull, face, shade, emission, boxes = [], [], [], [], [], [], [], [], []
        for el in model["elements"]:
            lo = np.array(el["from"], dtype=np.float32)
            hi = np.array(el["to"], dtype=np.float32)
            erot = element_rotation(el.get("rotation"))
            corners_all = []
            for fname, f in el.get("faces", {}).items():
                texture = self.texture_of(model, f.get("texture", ""))
                if texture is None:
                    continue
                corners = np.array([[lo[i] if c[i] == 0 else hi[i] for i in range(3)] for c in CORNERS[fname]], dtype=np.float32)
                if "uv" in f:
                    u1, v1, u2, v2 = f["uv"]
                    quad_uv = np.array([(u1, v1), (u1, v2), (u2, v2), (u2, v1)], dtype=np.float32)
                    quad_uv = np.roll(quad_uv, -(int(f.get("rotation", 0)) // 90), axis=0)
                else:
                    quad_uv = auto_uv(corners, fname)
                world = erot(corners)
                world = (world - 8.0) @ rot.T + 8.0
                fdir = DIRS.index(fname)
                new_dir = snap(rot @ DIR_VEC[fdir])
                if uvlock and (x or y):
                    quad_uv = auto_uv(world, DIRS[new_dir])
                pos.append(world / 16.0)
                uv.append(quad_uv / 16.0)
                tex.append(texture)
                tint.append(int(f.get("tintindex", -1)))
                cull.append(snap(rot @ DIR_VEC[DIRS.index(f["cullface"])]) if "cullface" in f else -1)
                face.append(new_dir)
                shade_dir = snap(rot @ DIR_VEC[DIRS.index(el["shade_direction_override"])]) if "shade_direction_override" in el else new_dir
                shade.append(DIR_SHADE[shade_dir] if el.get("shade", True) else 1.0)
                emission.append(int(el.get("light_emission", 0)))
                corners_all.append(world)
            if corners_all:
                pts = np.concatenate(corners_all) / 16.0
                boxes.append(np.stack([pts.min(0), pts.max(0)]))
        if not pos:
            return None
        return Baked(
            np.stack(pos).astype(np.float32), np.stack(uv).astype(np.float32), tex, np.array(tint), np.array(cull),
            np.array(face), np.array(shade, dtype=np.float32), np.array(emission), np.stack(boxes).astype(np.float32),
            bool(model["ambientocclusion"]),
        )

    @cache
    def kind(self, block: str) -> BlockKind | None:
        state = self.blockstate(block)
        if state is None:
            return None
        return BlockKind(block, variants=state.get("variants"), multipart=state.get("multipart"))

    def choose(self, block: str, rng: random.Random) -> tuple[dict[str, str], list[Baked]]:
        """A random state of block as its properties and the baked parts that draw it"""
        if split(block)[1] in FLUIDS:
            return {"level": "0"}, [fluid(split(block)[1])]
        k = self.kind(block)
        if k is None:
            return {}, []
        if k.variants is not None:
            key = rng.choice(list(k.variants))
            props = parse_props(key)
            parts = [pick(k.variants[key], rng)]
        else:
            values: dict[str, set[str]] = {}
            for part in k.multipart:
                for cond in conditions(part.get("when")):
                    for p, v in cond.items():
                        values.setdefault(p, set()).update(str(v).split("|"))
            props = {p: rng.choice(sorted(v)) for p, v in values.items()}
            parts = [pick(part["apply"], rng) for part in k.multipart if matches(part.get("when"), props)]
        baked = [self.bake(p["model"], int(p.get("x", 0)), int(p.get("y", 0)), bool(p.get("uvlock", False))) for p in parts]
        return props, [b for b in baked if b is not None]

    def parts(self, block: str, props: dict[str, str], rng: random.Random) -> list[Baked]:
        """The baked parts of block in the state props names"""
        k = self.kind(block)
        if k is None:
            return []
        if k.variants is not None:
            keys = [key for key in k.variants if parse_props(key).items() <= props.items()]
            if not keys:
                return []
            applies = [pick(k.variants[keys[0]], rng)]
        else:
            applies = [pick(part["apply"], rng) for part in k.multipart if matches(part.get("when"), props)]
        baked = [self.bake(a["model"], int(a.get("x", 0)), int(a.get("y", 0)), bool(a.get("uvlock", False))) for a in applies]
        return [b for b in baked if b is not None]

    def tints(self, block: str, props: dict[str, str], biome: Biome) -> list[np.ndarray]:
        """The RGB tint of each tint layer, white for an untinted layer"""
        name = split(block)[1]
        white = np.ones(3, dtype=np.float32)
        if name in GRASS:
            layer = rgb(biome.grass) if biome.grass is not None else self.colormap("grass", biome)
            return [layer, white]
        if name in SECOND_LAYER_GRASS:
            return [white, rgb(biome.grass) if biome.grass is not None else self.colormap("grass", biome)]
        if name in FOLIAGE:
            return [rgb(biome.foliage) if biome.foliage is not None else self.colormap("foliage", biome)]
        if name == "leaf_litter":
            return [self.colormap("dry_foliage", biome)]
        if name in WATER_TINTED:
            return [rgb(biome.water)]
        if name in CONSTANT_TINT:
            return [rgb(CONSTANT_TINT[name])]
        if name == "redstone_wire":
            f = int(props.get("power", 0)) / 15.0
            r = f * 0.6 + (0.4 if f > 0 else 0.3)
            return [np.array([r, min(max(f * f * 0.7 - 0.5, 0), 1), min(max(f * f * 0.6 - 0.7, 0), 1)], dtype=np.float32)]
        if name in ("melon_stem", "pumpkin_stem"):
            age = int(props.get("age", 0))
            return [np.array([age * 32, 255 - age * 8, age * 4], dtype=np.float32) / 255.0]
        return [white]

    def colormap(self, which: str, biome: Biome) -> np.ndarray:
        """The game's ColorMapColorUtil lookup"""
        t = min(max(biome.temperature, 0.0), 1.0)
        d = min(max(biome.downfall, 0.0), 1.0) * t
        img = self.colormaps[which]
        return img[int((1 - d) * 255), int((1 - t) * 255), :3].astype(np.float32) / 255.0


@cache
def fluid(name: str) -> Baked:
    """A still fluid's block, sunk to the height the game gives a source with nothing above it"""
    lo = np.zeros(3, dtype=np.float32)
    hi = np.array([16, 16 * FLUID_HEIGHT, 16], dtype=np.float32)
    pos, uv, tex = [], [], []
    for fname in DIRS:
        corners = np.array([[lo[i] if c[i] == 0 else hi[i] for i in range(3)] for c in CORNERS[fname]], dtype=np.float32)
        pos.append(corners / 16.0)
        uv.append(auto_uv(corners, fname) / 16.0)
        tex.append(f"minecraft:block/{name}_{'still' if fname in ('up', 'down') else 'flow'}")
    n = len(DIRS)
    return Baked(
        np.stack(pos), np.stack(uv), tex, np.full(n, 0 if name == "water" else -1), np.arange(n), np.arange(n),
        DIR_SHADE.copy(), np.full(n, 15 if name == "lava" else 0), np.stack([lo, hi])[None] / 16.0, ao=False,
    )


def light_of(block: str, props: dict[str, str]) -> int:
    name = split(block)[1]
    if name in LIT_LIGHT:
        return LIT_LIGHT[name] if props.get("lit") == "true" else 0
    if name.endswith("candle"):
        return 3 * int(props.get("candles", 1)) if props.get("lit") == "true" else 0
    return LIGHT.get(name, 0)


def split(name: str) -> tuple[str, str]:
    ns, _, path = name.rpartition(":")
    return ns or "minecraft", path


def rgb(color: int) -> np.ndarray:
    return np.array([(color >> 16) & 255, (color >> 8) & 255, color & 255], dtype=np.float32) / 255.0


def parse_props(key: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in key.split(",") if "=" in p)


def pick(apply, rng: random.Random) -> dict:
    if isinstance(apply, dict):
        return apply
    return rng.choices(apply, weights=[a.get("weight", 1) for a in apply])[0]


def conditions(when) -> list[dict]:
    if when is None:
        return []
    if "OR" in when:
        return [c for w in when["OR"] for c in conditions(w)]
    if "AND" in when:
        return [c for w in when["AND"] for c in conditions(w)]
    return [when]


def matches(when, props: dict[str, str]) -> bool:
    if when is None:
        return True
    if "OR" in when:
        return any(matches(w, props) for w in when["OR"])
    if "AND" in when:
        return all(matches(w, props) for w in when["AND"])
    for p, v in when.items():
        allowed = str(v).split("|")
        negate = allowed[0].startswith("!")
        if negate:
            if props.get(p) in [a.lstrip("!") for a in allowed]:
                return False
        elif props.get(p) not in allowed:
            return False
    return True


def auto_uv(corners: np.ndarray, face: str) -> np.ndarray:
    """The game's default UVs for a face without them, per corner from its position in pixels"""
    x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
    u, v = {
        "down": (x, 16 - z), "up": (x, z), "north": (16 - x, 16 - y),
        "south": (x, 16 - y), "west": (z, 16 - y), "east": (16 - z, 16 - y),
    }[face]
    return np.stack([u, v], axis=1).astype(np.float32)


def element_rotation(spec: dict | None):
    if not spec:
        return lambda p: p
    if "axis" not in spec:
        # newer models rotate about several axes at once, x then y then z
        steps = [element_rotation({**spec, "axis": a, "angle": spec[a]}) for a in "xyz" if spec.get(a)]
        def apply(p):
            for step in steps:
                p = step(p)
            return p
        return apply
    if not spec.get("angle"):
        return lambda p: p
    axis = "xyz".index(spec["axis"])
    a = math.radians(spec["angle"])
    origin = np.array(spec.get("origin", [8, 8, 8]), dtype=np.float32)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(3, dtype=np.float32)
    i, j = [k for k in range(3) if k != axis]
    if axis == 1:
        # about y the game turns x toward -z for a positive angle
        m[i, i], m[i, j], m[j, i], m[j, j] = c, s, -s, c
    else:
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    scale = np.ones(3, dtype=np.float32)
    if spec.get("rescale"):
        scale[[i, j]] = 1.0 / abs(c)
    return lambda p: ((p - origin) @ m.T) * scale + origin


def variant_rotation(x: int, y: int) -> np.ndarray:
    """The blockstate's x then y rotation, both clockwise looking down the axis, as the game's BlockModelRotation"""
    ax, ay = math.radians(-x), math.radians(-y)
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]], dtype=np.float32)
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]], dtype=np.float32)
    return (ry @ rx).round(6)


def snap(v: np.ndarray) -> int:
    return int(np.argmax(DIR_VEC @ np.asarray(v, dtype=np.float32)))
