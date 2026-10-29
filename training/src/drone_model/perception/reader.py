"""
The block reader: a ConvNeXt-Tiny encoder pretrained on ImageNet with a U-Net decoder, which labels each pixel of the camera
image and depth with the block or kind of mob it shows.
It is trained on the mod's mask and state streams and needs neither when it runs
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..torch_utils import load_weights, pick_device
from .classes import alias, load

SKY, OTHER = 0, 1
_spec = load()
# every block a camera can show and every kind of mob, see classes.py
BLOCK_KINDS = tuple(b["name"] for b in _spec["blocks"])
MOB_KINDS = tuple(_spec["mobs"])
CLASSES = ("<sky>", "<other>", *BLOCK_KINDS, *MOB_KINDS)
INDEX = {name: i for i, name in enumerate(CLASSES)}
# names the class list, labels saved as class indices only fit the list they were made for
CLASS_HASH = hashlib.sha1("\n".join(CLASSES).encode()).hexdigest()[:8]
MOBS = np.asarray([INDEX[name] for name in MOB_KINDS], dtype=np.uint16)
MOB_SET = frozenset(MOB_KINDS)
# blocks a mob can't stand inside, from the game's collision shapes
SOLID = frozenset(b["name"] for b in _spec["blocks"] if b["solid"])
DEPTH_MAX = 64.0
# ImageNet's channel statistics, the encoder was pretrained on inputs scaled by them
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
# the encoder's deepest features are a 32nd of the frame, the input is padded to a multiple of this
STRIDE = 32
# the camera size the reader trains and runs at, (height, width)
FRAME = (240, 320)


def state_table(state_names: list[str]) -> np.ndarray:
    """Lookup table from the state stream (0 sky, 1 + state id) to a class index"""
    classes = np.full(len(state_names) + 1, OTHER, dtype=np.uint16)
    classes[0] = SKY
    for i, name in enumerate(state_names):
        classes[i + 1] = INDEX.get(alias(name.partition("[")[0]), OTHER)
    return classes


def entity_table(entities: list[str]) -> np.ndarray:
    """
    Reader class per entity in a mask_ids entities list: the mob's own class for MOB_KINDS, SKY for everything else
    (items, drones, players), which the state stream never labeled as a block either
    """
    return np.asarray([INDEX[name] if name in INDEX and name in MOB_SET else SKY for name in entities], dtype=np.uint16)


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
    )


class BlockReader(nn.Module):
    """
    ConvNeXt-Tiny encoder and U-Net decoder over rgb and depth, outputs class logits (B, len(CLASSES), H, W).
    Depth is a fourth input channel beside rgb.
    pretrained starts the encoder from torchvision's ImageNet weights, a trained reader's checkpoint replaces them anyway
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None).features
        rgb_stem = net[0][0]
        stem = nn.Conv2d(4, 96, 4, stride=4)
        with torch.no_grad():
            stem.weight[:, :3] = rgb_stem.weight
            # the depth channel's weights start as the mean of the color channels'
            stem.weight[:, 3:] = rgb_stem.weight.mean(1, keepdim=True)
            stem.bias.copy_(rgb_stem.bias)
        net[0][0] = stem
        self.encoder = nn.ModuleList([net[0:2], net[2:4], net[4:6], net[6:8]])
        # each decoder stage upsamples by 2 and joins the encoder's features at that size, dec1 is full size
        # ConvNeXt starts at a quarter of the frame, so the half and full size features come from small conv stages
        self.half_size = block(4, 64, stride=2)
        self.full = block(4, 32)
        self.dec16 = block(768 + 384, 256)
        self.dec8 = block(256 + 192, 128)
        self.dec4 = block(128 + 96, 96)
        self.dec2 = block(96 + 64, 64)
        self.dec1 = block(64 + 32, 32)
        self.classes = nn.Conv2d(32, len(CLASSES), 1)
        self.register_buffer("rgb_mean", torch.tensor(RGB_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(RGB_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """rgb (B, H, W, 3) uint8, depth (B, H, W) in blocks"""
        return self.classes(self.features(rgb, depth))

    def features(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """The decoder's full-size features (B, 32, H, W), which the class head reads"""
        color = (rgb.permute(0, 3, 1, 2).float() / 255.0 - self.rgb_mean) / self.rgb_std
        near = (depth.float().clamp(0, DEPTH_MAX) / DEPTH_MAX - 0.5) / 0.25
        x = torch.cat([color, near.unsqueeze(1)], dim=1)
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % STRIDE, 0, -h % STRIDE), mode="replicate")
        skips = [self.half_size(x)]
        e = x
        for stage in self.encoder:
            e = stage(e)
            skips.append(e)
        d = skips.pop()
        for dec, skip in zip((self.dec16, self.dec8, self.dec4, self.dec2, self.dec1), (*reversed(skips), self.full(x))):
            d = dec(torch.cat([F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False), skip], dim=1))
        return d[..., :h, :w]


def mask_lut(mask_ids: dict) -> np.ndarray:
    """Reader classes to the mod's mask ids, so code written for the mask stream can run on the reader's output"""
    id_of = {name: i + 1 for i, name in enumerate(mask_ids["blocks"])}
    base = mask_ids["entityBase"]
    id_of.update({name: base + j for j, name in enumerate(mask_ids["entities"]) if name in MOB_SET})
    # some solid block for "<other>", it still counts as something in the way
    other = id_of.get("minecraft:bedrock", 1)
    return np.asarray([0, other, *(id_of.get(name, other) for name in CLASSES[2:])], dtype=np.uint16)


def class_lut(mask_ids: dict) -> np.ndarray:
    """The mod's mask ids to reader classes, so ground-truth frames go into the voxel memory like the reader's output"""
    base = mask_ids["entityBase"]
    lut = np.full(base + len(mask_ids["entities"]), OTHER, dtype=np.uint16)
    lut[0] = SKY
    for i, name in enumerate(mask_ids["blocks"]):
        lut[i + 1] = SKY if name in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air") else INDEX.get(alias(name), OTHER)
    # mobs to their own classes, items and drones to nothing
    for j, name in enumerate(mask_ids["entities"]):
        lut[base + j] = INDEX[name] if name in MOB_SET else SKY
    return lut


class Reader:
    """A trained BlockReader for one frame at a time, returning class indices"""

    def __init__(self, checkpoint, device: torch.device | None = None) -> None:
        self.device = device or pick_device()
        self.model = BlockReader().to(self.device)
        load_weights(self.model, checkpoint, self.device)
        self.model.eval()

    @torch.no_grad()
    def read(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        logits = self.model(torch.from_numpy(rgb)[None].to(self.device), torch.from_numpy(depth)[None].to(self.device))
        return logits[0].argmax(0).cpu().numpy().astype(np.uint16)
