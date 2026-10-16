"""
The GPU side: every block texture in one texture array, a color pass drawn at twice the frame size and box-filtered
down as FrameGrabber does, and a label pass that writes the class, ripeness, and z-depth of each pixel's center, as
the mod's Raycaster casts one ray per pixel center
"""

from __future__ import annotations

import math

import moderngl
import numpy as np
from PIL import Image

from .assets import Assets, split

TILE = 16
# the client's fov option is 70 degrees, vertical
FOV = 70.0
NEAR = 0.05
FAR = 1024.0
SUPERSAMPLE = 2

COLOR_VS = """
#version 430
uniform mat4 mvp;
uniform vec3 cam;
uniform float sky_factor;
uniform vec3 sky_color;
uniform vec3 ambient;
uniform vec3 block_tint;
uniform float block_factor;
uniform float brightness;
in vec3 in_pos;
in vec2 in_uv;
in float in_layer;
in vec3 in_color;
in vec2 in_light;
out vec2 uv;
flat out float layer;
out vec4 vcolor;
out float dist;

float bright(float l) { return l / (4.0 - 3.0 * l); }

// the game's lightmap.fsh, sky level in x and block level in y
vec3 lightmap(vec2 level) {
    float sky = clamp(level.x / 15.0, 0.0, 1.0);
    float block = clamp(level.y / 15.0, 0.0, 1.0);
    vec3 c = ambient + sky_color * bright(sky) * sky_factor;
    float p = (2.0 * block - 1.0) * (2.0 * block - 1.0);
    c += mix(block_tint, vec3(1.0), 0.9 * p) * bright(block) * block_factor;
    c = clamp(c, 0.0, 1.0);
    float m = max(max(c.r, c.g), c.b);
    float inv = 1.0 - m;
    vec3 lifted = m > 0.0 ? c * ((1.0 - inv * inv * inv * inv) / m) : c;
    return mix(c, lifted, brightness);
}

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    uv = in_uv;
    layer = in_layer;
    vcolor = vec4(in_color * lightmap(in_light), 1.0);
    dist = length(in_pos.xz - cam.xz);
}
"""

COLOR_FS = """
#version 430
uniform sampler2DArray atlas;
uniform float cutout;
uniform vec3 fog_color;
uniform vec2 fog_range;
in vec2 uv;
flat in float layer;
in vec4 vcolor;
in float dist;
out vec4 frag;

void main() {
    vec4 c = texture(atlas, vec3(uv, layer)) * vcolor;
    if (c.a < cutout) discard;
    float f = clamp((dist - fog_range.x) / (fog_range.y - fog_range.x), 0.0, 1.0);
    frag = vec4(mix(c.rgb, fog_color, f), c.a);
}
"""

SKY_VS = """
#version 430
in vec2 in_pos;
out vec2 ndc;
void main() {
    ndc = in_pos;
    gl_Position = vec4(in_pos, 0.9999, 1.0);
}
"""

SKY_FS = """
#version 430
uniform mat4 inv_vp;
uniform vec3 sky_color;
uniform vec3 fog_color;
in vec2 ndc;
out vec4 frag;
void main() {
    vec4 far = inv_vp * vec4(ndc, 1.0, 1.0);
    vec4 near = inv_vp * vec4(ndc, -1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - near.xyz / near.w);
    frag = vec4(mix(fog_color, sky_color, smoothstep(-0.02, 0.45, dir.y)), 1.0);
}
"""

LABEL_VS = """
#version 430
uniform mat4 mvp;
uniform vec3 cam;
uniform vec3 forward;
in vec3 in_pos;
in uint in_cls;
in uint in_ripe;
flat out uint cls;
flat out uint ripe;
out float zdepth;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    cls = in_cls;
    ripe = in_ripe;
    zdepth = dot(in_pos - cam, forward);
}
"""

LABEL_FS = """
#version 430
flat in uint cls;
flat in uint ripe;
in float zdepth;
layout(location = 0) out uint out_cls;
layout(location = 1) out uint out_ripe;
layout(location = 2) out float out_depth;
void main() {
    out_cls = cls;
    out_ripe = ripe;
    out_depth = zdepth;
}
"""

CLEAR_FS = """
#version 430
uniform float depth_max;
layout(location = 0) out uint out_cls;
layout(location = 1) out uint out_ripe;
layout(location = 2) out float out_depth;
void main() {
    out_cls = 0u;
    out_ripe = 255u;
    out_depth = depth_max;
}
"""


class Textures:
    """Every texture a renderable block draws with, animation frames as consecutive layers"""

    def __init__(self, ctx: moderngl.Context, assets: Assets, names: set[str]) -> None:
        layers = [np.full((TILE, TILE, 4), (255, 0, 255, 255), dtype=np.uint8)]
        self.base: dict[str, int] = {}
        self.frames: dict[str, int] = {}
        # 0 opaque, 1 cutout, 2 translucent
        self.kind: dict[str, int] = {}
        for name in sorted(names):
            img = assets.image(name)
            ns, path = split(name)
            w = img.shape[1]
            n = img.shape[0] // w if img.shape[0] > w and assets.exists(ns, f"textures/{path}.png.mcmeta") else 1
            frames = [img[i * w : (i + 1) * w] for i in range(n)] if n > 1 else [img]
            self.base[name] = len(layers)
            self.frames[name] = n
            alpha = frames[0][..., 3]
            partial = ((alpha > 0) & (alpha < 255)).mean()
            self.kind[name] = 0 if alpha.min() == 255 else (2 if partial > 0.02 else 1)
            for f in frames:
                if f.shape[:2] != (TILE, TILE):
                    f = np.asarray(Image.fromarray(f).resize((TILE, TILE), Image.NEAREST))
                layers.append(f)
        self.texture = ctx.texture_array((TILE, TILE, len(layers)), 4, np.stack(layers).tobytes())
        self.texture.build_mipmaps(max_level=4)
        self.texture.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
        self.texture.repeat_x = self.texture.repeat_y = True

    def layer(self, name: str, frame: int) -> int:
        return self.base.get(name, 0) + frame % self.frames.get(name, 1)


class Renderer:
    def __init__(self, assets: Assets, names: set[str], frame: tuple[int, int], depth_max: float) -> None:
        self.ctx = moderngl.create_standalone_context(require=430)
        self.textures = Textures(self.ctx, assets, names)
        self.h, self.w = frame
        self.depth_max = depth_max
        ctx = self.ctx
        self.color_prog = ctx.program(vertex_shader=COLOR_VS, fragment_shader=COLOR_FS)
        self.sky_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.label_prog = ctx.program(vertex_shader=LABEL_VS, fragment_shader=LABEL_FS)
        self.clear_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=CLEAR_FS)
        screen = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32).tobytes())
        self.sky_vao = ctx.vertex_array(self.sky_prog, [(screen, "2f", "in_pos")])
        self.clear_vao = ctx.vertex_array(self.clear_prog, [(screen, "2f", "in_pos")])
        size = (self.w * SUPERSAMPLE, self.h * SUPERSAMPLE)
        self.color_fbo = ctx.framebuffer([ctx.texture(size, 4)], ctx.depth_renderbuffer(size))
        self.label_fbo = ctx.framebuffer(
            [ctx.texture((self.w, self.h), 1, dtype="u2"), ctx.texture((self.w, self.h), 1, dtype="u1"), ctx.texture((self.w, self.h), 1, dtype="f4")],
            ctx.depth_renderbuffer((self.w, self.h)),
        )
        self.proj = perspective(FOV, self.w / self.h, NEAR, FAR)
        self.scene_vaos: list = []

    def load(self, mesh) -> None:
        """Uploads a scene's mesh, replacing the last one"""
        for obj in self.scene_vaos:
            obj.release()
        ctx = self.ctx
        self.mesh = mesh
        self.opaque_buf = ctx.buffer(mesh.opaque.tobytes()) if len(mesh.opaque) else None
        self.label_buf = ctx.buffer(mesh.label.tobytes())
        fmt = "3f 2f 1f 3f 2f"
        names = ("in_pos", "in_uv", "in_layer", "in_color", "in_light")
        self.opaque_vao = ctx.vertex_array(self.color_prog, [(self.opaque_buf, fmt, *names)]) if self.opaque_buf else None
        self.label_vao = ctx.vertex_array(self.label_prog, [(self.label_buf, "3f 1u 1u", "in_pos", "in_cls", "in_ripe")])
        self.trans_buf = ctx.buffer(reserve=mesh.translucent.nbytes) if len(mesh.translucent) else None
        self.trans_vao = ctx.vertex_array(self.color_prog, [(self.trans_buf, fmt, *names)]) if self.trans_buf else None
        self.scene_vaos = [o for o in (self.opaque_buf, self.label_buf, self.opaque_vao, self.label_vao, self.trans_buf, self.trans_vao) if o is not None]

    def labels(self, eye: np.ndarray, forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Class, ripeness, and z-depth per pixel, rows top to bottom"""
        mvp = self.proj @ look(eye, forward)
        ctx = self.ctx
        self.label_fbo.use()
        ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        self.label_fbo.clear(depth=1.0)
        self.clear_prog["depth_max"].value = self.depth_max
        self.clear_vao.render(moderngl.TRIANGLES)
        self.label_prog["mvp"].write(mvp.T.astype(np.float32).tobytes())
        self.label_prog["cam"].value = tuple(eye)
        self.label_prog["forward"].value = tuple(forward)
        self.label_vao.render(moderngl.TRIANGLES)
        cls = np.frombuffer(self.label_fbo.read(attachment=0, components=1, dtype="u2"), dtype=np.uint16).reshape(self.h, self.w)[::-1]
        ripe = np.frombuffer(self.label_fbo.read(attachment=1, components=1, dtype="u1"), dtype=np.uint8).reshape(self.h, self.w)[::-1]
        depth = np.frombuffer(self.label_fbo.read(attachment=2, components=1, dtype="f4"), dtype=np.float32).reshape(self.h, self.w)[::-1]
        return cls.copy(), ripe.copy(), np.minimum(depth, self.depth_max)

    def color(self, eye: np.ndarray, forward: np.ndarray, env) -> np.ndarray:
        """The frame as uint8 RGB, rows top to bottom"""
        view = look(eye, forward)
        mvp = self.proj @ view
        ctx = self.ctx
        self.color_fbo.use()
        self.color_fbo.clear(*env.fog_color, 1.0, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        self.sky_prog["inv_vp"].write(np.linalg.inv(mvp).T.astype(np.float32).tobytes())
        self.sky_prog["sky_color"].value = tuple(env.sky_color)
        self.sky_prog["fog_color"].value = tuple(env.fog_color)
        self.sky_vao.render(moderngl.TRIANGLES)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.depth_func = "<="
        p = self.color_prog
        p["mvp"].write(mvp.T.astype(np.float32).tobytes())
        p["cam"].value = tuple(eye)
        p["sky_factor"].value = env.sky_factor
        p["sky_color"].value = tuple(env.sky_light_color)
        p["ambient"].value = tuple(env.ambient)
        p["block_tint"].value = tuple(env.block_tint)
        p["block_factor"].value = env.block_factor
        p["brightness"].value = env.brightness
        p["fog_color"].value = tuple(env.fog_color)
        p["fog_range"].value = env.fog_range
        self.textures.texture.use(0)
        p["atlas"].value = 0
        if self.opaque_vao is not None:
            p["cutout"].value = 0.5
            self.opaque_vao.render(moderngl.TRIANGLES)
        if self.trans_vao is not None:
            # back to front from this eye, quad by quad
            order = np.argsort(-np.linalg.norm(self.mesh.translucent_centers - eye, axis=1))
            self.trans_buf.write(self.mesh.translucent[order].tobytes())
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ctx.depth_mask = False
            p["cutout"].value = 0.01
            self.trans_vao.render(moderngl.TRIANGLES)
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)
        raw = np.frombuffer(self.color_fbo.read(components=3), dtype=np.uint8).reshape(self.h, SUPERSAMPLE, self.w, SUPERSAMPLE, 3)
        return (raw.mean(axis=(1, 3)) + 0.5).astype(np.uint8)[::-1].copy()


def perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float64)


def look(eye: np.ndarray, forward: np.ndarray) -> np.ndarray:
    f = forward / np.linalg.norm(forward)
    right = np.cross(f, [0.0, 1.0, 0.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, up, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m
