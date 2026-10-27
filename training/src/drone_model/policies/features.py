"""What the learned policies share: the conv block of their image encoders, the drone state vector, and the downscaled frames"""

from __future__ import annotations

import math

import numpy as np
from torch import nn

SLOTS = 27
# velocity (3), sin and cos of yaw, pitch / 90, geofence position (3), mining progress, container open, selected slot / 26
STATE_DIM = 12
HEIGHT, WIDTH = 60, 80
DEPTH_MAX = 64.0


def conv(cin: int, cout: int, k: int, s: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


def state_features(state: dict) -> np.ndarray:
    """The state vector a policy reads, only what the drone knows of itself"""
    vel = state.get("vel", [0.0, 0.0, 0.0])
    yaw = math.radians(state.get("yaw", 0.0))
    box, pos = state.get("bounds"), state.get("pos")
    fence = [(pos[i] - box[i]) / (box[i + 3] - box[i]) * 2 - 1 for i in range(3)] if box and pos else [0.0, 0.0, 0.0]
    breaking = state.get("breaking")
    return np.asarray(
        [*vel, math.sin(yaw), math.cos(yaw), state.get("pitch", 0.0) / 90.0, *fence,
         breaking["progress"] if breaking else 0.0, 1.0 if state.get("container") else 0.0, state.get("selectedSlot", 0) / (SLOTS - 1)],
        dtype=np.float32,
    )


def frame_features(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-size rgb and depth averaged and sampled down to HEIGHT x WIDTH, depth as 0 to 255 over DEPTH_MAX"""
    sy = rgb.shape[0] // HEIGHT
    sx = rgb.shape[1] // WIDTH
    rgb = rgb.reshape(HEIGHT, sy, WIDTH, sx, 3).mean(axis=(1, 3)).astype(np.uint8)
    depth = (np.clip(depth[::sy, ::sx], 0, DEPTH_MAX) / DEPTH_MAX * 255).astype(np.uint8)
    return rgb, depth
