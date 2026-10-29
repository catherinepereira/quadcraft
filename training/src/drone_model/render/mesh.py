"""
A scene of placed blocks to vertex arrays: faces hidden behind full blocks culled, sky and block light flooded through a
voxel grid, and each corner lit and shaded by its neighbors as the game's smooth lighting and ambient occlusion do
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .assets import DIR_VEC, Baked

COLOR = np.dtype([("pos", "<f4", 3), ("uv", "<f4", 2), ("layer", "<f4"), ("color", "<f4", 3), ("light", "<f4", 2)])
LABEL = np.dtype([("pos", "<f4", 3), ("cls", "<u4")])
TRIANGLES = np.array([0, 1, 2, 0, 2, 3])
# the 36 corners of a box's 12 triangles as 0 (min) or 1 (max) per axis
BOX = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1],
    [0, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0],
    [0, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 0, 1], [1, 0, 0],
    [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1],
], dtype=np.float32)
# the game darkens a corner by this much per full block around it
AO_DARK = 0.2


@dataclass
class Placed:
    block: str
    props: dict[str, str]
    parts: list[Baked]
    cls: int
    light: int
    occludes: bool
    tints: list[np.ndarray]
    fluid: int = 0


@dataclass
class Mesh:
    opaque: np.ndarray  # (V,) COLOR
    translucent: np.ndarray  # (T, 6) COLOR, one row of two triangles per quad
    translucent_centers: np.ndarray  # (T, 3)
    label: np.ndarray  # (V,) LABEL


class Grid:
    """Occlusion, fluids, and light levels over the scene's bounds, out of bounds reads as open air in full sky"""

    def __init__(self, cells: dict[tuple[int, int, int], Placed]) -> None:
        keys = np.array(list(cells), dtype=np.int32)
        self.lo = keys.min(0) - 2
        shape = keys.max(0) + 3 - self.lo
        self.occ = np.zeros(shape, dtype=bool)
        self.fluid = np.zeros(shape, dtype=np.int8)
        emit = np.zeros(shape, dtype=np.int16)
        for (x, y, z), p in cells.items():
            i = (x - self.lo[0], y - self.lo[1], z - self.lo[2])
            self.occ[i] = p.occludes
            self.fluid[i] = p.fluid
            emit[i] = p.light
        # sky light falls straight down until the first full block, then spreads sideways
        below = np.flip(np.maximum.accumulate(np.flip(self.occ, 1), axis=1), 1)
        self.sky = spread(np.where(below, 0, 15).astype(np.int16), self.occ)
        self.block = spread(emit, self.occ, fixed=emit)

    def at(self, cells: np.ndarray):
        """occ, fluid, sky, block at integer cells (..., 3)"""
        i = cells - self.lo
        inside = np.all((i >= 0) & (i < self.occ.shape), axis=-1)
        i = np.where(inside[..., None], i, 0)
        ix = (i[..., 0], i[..., 1], i[..., 2])
        return (
            np.where(inside, self.occ[ix], False),
            np.where(inside, self.fluid[ix], 0),
            np.where(inside, self.sky[ix], 15),
            np.where(inside, self.block[ix], 0),
        )


def spread(level: np.ndarray, occ: np.ndarray, fixed: np.ndarray | None = None) -> np.ndarray:
    """Light flooding one level dimmer per step into every cell a full block doesn't fill"""
    level = level.copy()
    for _ in range(15):
        nxt = level.copy()
        for axis in range(3):
            for step in (1, -1):
                shifted = np.roll(level, step, axis=axis)
                edge = [slice(None)] * 3
                edge[axis] = 0 if step == 1 else -1
                shifted[tuple(edge)] = 0
                np.maximum(nxt, shifted - 1, out=nxt)
        nxt[occ] = 0
        if fixed is not None:
            np.maximum(nxt, fixed, out=nxt)
        if np.array_equal(nxt, level):
            break
        level = nxt
    return level


def build(cells: dict[tuple[int, int, int], Placed], textures, frames: dict[str, int], extra: list[np.ndarray] = ()) -> Mesh:
    grid = Grid(cells)
    # ground cells share one Placed, so group by it before the per-part keys
    by_placed: dict[int, tuple[Placed, list]] = {}
    for pos, p in cells.items():
        by_placed.setdefault(id(p), (p, []))[1].append(pos)
    groups: dict[tuple, list] = defaultdict(list)
    labels: dict[tuple, list] = defaultdict(list)
    for p, positions in by_placed.values():
        tint_key = tuple(t.tobytes() for t in p.tints)
        for part in p.parts:
            groups[(id(part), tint_key, p.light > 0, p.fluid)].append((positions, part, p.tints))
            labels[(id(part), p.cls)].append((positions, part))
    opaque, translucent = list(extra), []
    for (_, _, emits, fluid), members in groups.items():
        part, tints = members[0][1], members[0][2]
        positions = np.array([pos for m in members for pos in m[0]], dtype=np.int32)
        quads = color_quads(grid, part, positions, tints, emits, fluid, textures, frames)
        if quads is None:
            continue
        verts, is_trans = quads
        opaque.append(verts[~is_trans].reshape(-1))
        if is_trans.any():
            translucent.append(verts[is_trans])
    label = [label_boxes(members[0][1], np.array([pos for m in members for pos in m[0]], dtype=np.int32), cls) for (_, cls), members in labels.items()]
    trans = np.concatenate(translucent) if translucent else np.zeros((0, 6), dtype=COLOR)
    return Mesh(
        np.concatenate(opaque) if opaque else np.zeros(0, dtype=COLOR),
        trans,
        trans["pos"].mean(axis=1) if len(trans) else np.zeros((0, 3)),
        np.concatenate(label),
    )


def color_quads(grid: Grid, part: Baked, positions: np.ndarray, tints, emits: bool, fluid: int, textures, frames):
    n_blocks, n_quads = len(positions), len(part.texture)
    keep = np.ones((n_blocks, n_quads), dtype=bool)
    culls = part.cull >= 0
    if culls.any():
        nb = positions[:, None, :] + DIR_VEC[part.cull[culls]][None]
        occ, nfluid, _, _ = grid.at(nb)
        keep[:, culls] = ~occ & ~((nfluid == fluid) & (fluid > 0))
    b, q = np.nonzero(keep)
    if len(b) == 0:
        return None
    base = positions[b]
    local = part.pos[q]
    normal = DIR_VEC[part.face[q]]
    axis = np.abs(normal).argmax(1)
    plane = np.take_along_axis(local, axis[:, None, None].repeat(4, 1), axis=2)[..., 0]
    outward = (normal.sum(1) > 0)[:, None]
    boundary = np.all(np.where(outward, plane > 1 - 1e-4, plane < 1e-4), axis=1)

    # smooth light and ambient occlusion, the four cells in front of each corner
    front = base + normal
    t1 = (axis + 1) % 3
    t2 = (axis + 2) % 3
    side1 = corner_sign(local, t1)
    side2 = corner_sign(local, t2)
    e1 = np.eye(3, dtype=np.int32)[t1][:, None, :] * side1[..., None]
    e2 = np.eye(3, dtype=np.int32)[t2][:, None, :] * side2[..., None]
    f = front[:, None, :]
    cells = np.stack([f + 0 * e1, f + e1, f + e2, f + e1 + e2], axis=2)
    occ, _, sky, block = grid.at(cells)
    corner_hidden = occ[..., 3] | (occ[..., 1] & occ[..., 2])
    occ_eff = np.concatenate([occ[..., :3], corner_hidden[..., None]], axis=-1)
    valid = ~occ_eff
    count = np.maximum(valid.sum(-1), 1)
    smooth_sky = (sky * valid).sum(-1) / count
    smooth_block = (block * valid).sum(-1) / count
    ao = np.where(occ_eff, AO_DARK, 1.0).mean(-1)

    # faces inside the block take the light of the block's own cell
    own_occ, _, own_sky, own_block = grid.at(base)
    _, _, front_sky, front_block = grid.at(front)
    flat_sky = np.where(own_occ, front_sky, own_sky)[:, None].repeat(4, 1)
    flat_block = np.where(own_occ, front_block, own_block)[:, None].repeat(4, 1)
    sky_l = np.where(boundary[:, None], smooth_sky, flat_sky)
    block_l = np.where(boundary[:, None], smooth_block, flat_block)
    block_l = np.maximum(block_l, part.emission[q][:, None])
    if not part.ao or emits:
        ao = np.ones_like(ao)
    ao = np.where(boundary[:, None], ao, 1.0)

    tint_table = np.stack([np.ones(3, dtype=np.float32), *tints])
    tint_index = np.where((part.tint[q] >= 0) & (part.tint[q] < len(tints)), part.tint[q] + 1, 0)
    color = tint_table[tint_index][:, None, :] * (part.shade[q][:, None] * ao)[..., None]

    names = [part.texture[i] for i in q]
    layer = np.array([textures.layer(name, frames.get(name, 0)) for name in names], dtype=np.float32)
    is_trans = np.array([textures.kind.get(name, 0) == 2 for name in names])

    v = np.zeros((len(q), 4), dtype=COLOR)
    v["pos"] = local + base[:, None, :]
    v["uv"] = part.uv[q]
    v["layer"] = layer[:, None]
    v["color"] = color
    v["light"] = np.stack([sky_l, block_l], axis=-1)
    return v[:, TRIANGLES], is_trans


def far_ground(layer: int, tint: np.ndarray, y: float, reach: float = 256.0) -> np.ndarray:
    """One wide quad of the ground's top texture under the scene, standing in for the terrain beyond it"""
    corners = np.array([[-reach, y, -reach], [-reach, y, reach], [reach, y, reach], [reach, y, -reach]], dtype=np.float32)
    v = np.zeros(4, dtype=COLOR)
    v["pos"] = corners
    v["uv"] = corners[:, [0, 2]]
    v["layer"] = layer
    v["color"] = tint
    v["light"] = (15, 0)
    return v[TRIANGLES]


def corner_sign(local: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """-1, 0, or 1 per corner along axis, which neighbor the corner sits against"""
    c = np.take_along_axis(local, axis[:, None, None].repeat(4, 1), axis=2)[..., 0]
    return np.where(c < 1e-4, -1, np.where(c > 1 - 1e-4, 1, 0)).astype(np.int32)


def label_boxes(part: Baked, positions: np.ndarray, cls: int) -> np.ndarray:
    lo, hi = part.boxes[:, 0], part.boxes[:, 1]
    corners = lo[:, None, :] + BOX[None] * (hi - lo)[:, None, :]
    v = np.zeros((len(positions), len(part.boxes), 36), dtype=LABEL)
    v["pos"] = corners[None] + positions[:, None, None, :]
    v["cls"] = cls
    return v.reshape(-1)
