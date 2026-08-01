"""Deterministic, headless fantasy town generator for Blender 5.1.

Full run:
  blender -b --factory-startup -P scripts/generate_town.py
Quick validation:
  set TOWN_TEST=1
  blender -b --factory-startup -P scripts/generate_town.py
"""

import bpy
import math
import os
import random
import shutil
import time
import numpy as np
from mathutils import Vector
from pathlib import Path


SEED = 517042
RNG = random.Random(SEED)
TEST_MODE = os.environ.get("TOWN_TEST", "0") == "1"
DUSK_MODE = os.environ.get("TOWN_DUSK", "0") == "1"
TURNTABLE_MODE = os.environ.get("TOWN_TURNTABLE", "0") == "1"
WALKTHROUGH_MODE = os.environ.get("TOWN_WALKTHROUGH", "0") == "1"
SNOW_MODE = os.environ.get("TOWN_SNOW", "0") == "1"
ROOT = Path(__file__).resolve().parent.parent
RENDER_DIR = ROOT / "renders"
EXPORT_DIR = ROOT / "export"
TEXTURE_DIR = ROOT / "textures"
RENDER_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
START_TIME = time.time()


def log(message):
    print("[TOWN] " + message, flush=True)


def clean_scene():
    for datablocks in (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            datablocks.remove(block)

    root = bpy.context.scene.collection
    for child in list(root.children):
        root.children.unlink(child)
        bpy.data.collections.remove(child)


clean_scene()

COLLECTIONS = {}
for name in ("Castle", "Town", "Walls", "Props", "Ground", "Environment"):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    COLLECTIONS[name] = coll


def periodic_noise(size, seed, octaves=5):
    """Small deterministic seamless value field made from periodic waves."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    out = np.zeros((size, size), dtype=np.float32)
    weight = 1.0
    for octave in range(octaves):
        freq = 2 ** octave
        phase1, phase2 = rng.random(2) * math.tau
        # Integer wave vectors guarantee mathematically exact tiling at the
        # image borders (arbitrary rotation angles do not).
        kx1, ky1 = int(rng.integers(0, freq + 1)), int(rng.integers(1, freq + 1))
        kx2, ky2 = int(rng.integers(1, freq + 1)), int(rng.integers(0, freq + 1))
        out += weight * (
            np.sin(math.tau * (kx1 * xx + ky1 * yy) + phase1)
            + np.cos(math.tau * (kx2 * xx - ky2 * yy) + phase2)
        )
        weight *= 0.5
    out -= out.min()
    out /= max(float(out.max()), 1e-6)
    return out


def save_texture(name, rgb):
    path = TEXTURE_DIR / f"{name}.png"
    rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    rgba[:, :, 3] = 1.0
    image = bpy.data.images.new("TEX_" + name, width=rgb.shape[1], height=rgb.shape[0], alpha=True)
    image.pixels.foreach_set(rgba.reshape(-1))
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return path


def make_block_texture(size, base, mortar, cols, rows, seed, bevel=0.08, variation=0.12):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cell_w, cell_h = size / cols, size / rows
    row = np.floor(yy / cell_h).astype(np.int32)
    shifted_x = np.mod(xx + (row % 2) * cell_w * 0.5, size)
    col = np.floor(shifted_x / cell_w).astype(np.int32)
    lx = np.mod(shifted_x, cell_w) / cell_w
    ly = np.mod(yy, cell_h) / cell_h
    edge = (lx < bevel) | (lx > 1.0 - bevel) | (ly < bevel) | (ly > 1.0 - bevel)
    cell_hash = np.mod(np.sin((col + 1) * 12.9898 + (row + 2) * 78.233) * 43758.5453, 1.0)
    n = periodic_noise(size, seed, 5)
    shade = (cell_hash - 0.5) * variation + (n - 0.5) * 0.13
    rgb = np.asarray(base, dtype=np.float32)[None, None, :] + shade[:, :, None]
    rgb[edge] = np.asarray(mortar, dtype=np.float32)
    return rgb


def make_roof_texture(size, base, seed):
    # Horizontal tile courses: soft row shadow + slight per-tile tint. Low
    # contrast on purpose — high contrast reads as checkerboard from afar.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cols, rows = 16, 24
    cw, ch = size / cols, size / rows
    row = np.floor(yy / ch).astype(np.int32)
    sx = np.mod(xx + (row % 2) * cw * 0.5, size)
    col = np.floor(sx / cw).astype(np.int32)
    lx = np.mod(sx, cw) / cw
    ly = np.mod(yy, ch) / ch
    course_shadow = np.clip(1.0 - ly * 2.2, 0.0, 1.0) * 0.17
    seam = ((lx < 0.045) | (lx > 0.955)) * 0.11
    tile_hash = np.mod(np.sin((col + 3) * 12.9898 + (row + 7) * 78.233) * 43758.5453, 1.0)
    n = periodic_noise(size, seed, 4) - 0.5
    shade = (tile_hash - 0.5) * 0.11 + n * 0.07 - course_shadow - seam
    rgb = np.asarray(base, dtype=np.float32)[None, None, :] * (1.0 + shade)[:, :, None]
    return rgb


def generate_textures():
    size = 512
    log("Generating seamless 512px textures")
    outputs = {}
    outputs["cobble"] = save_texture(
        "cobblestone",
        make_block_texture(size, (0.34, 0.35, 0.34), (0.185, 0.19, 0.185), 9, 15, 11, 0.08, 0.11),
    )
    noise = periodic_noise(size, 21, 6)
    stains = periodic_noise(size, 22, 3)
    for key, base in (
        ("plaster_cream", (0.78, 0.68, 0.51)),
        ("plaster_rose", (0.57, 0.31, 0.27)),
        ("plaster_ochre", (0.65, 0.44, 0.20)),
    ):
        rgb = np.asarray(base)[None, None, :] + ((noise - 0.5) * 0.15 - np.maximum(stains - 0.72, 0) * 0.35)[:, :, None]
        outputs[key] = save_texture(key, rgb)
    outputs["stone"] = save_texture(
        "stone_blocks",
        make_block_texture(size, (0.43, 0.43, 0.405), (0.13, 0.135, 0.13), 7, 13, 31, 0.075, 0.17),
    )
    outputs["roof_tile"] = save_texture("roof_tile_terracotta", make_roof_texture(size, (0.46, 0.13, 0.055), 41))
    outputs["roof_slate"] = save_texture("roof_tile_slate", make_roof_texture(size, (0.12, 0.15, 0.17), 42))
    outputs["roof_green"] = save_texture("roof_tile_green", make_roof_texture(size, (0.065, 0.19, 0.115), 43))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    wood_n = periodic_noise(size, 51, 5)
    # Straight grain along X with only mild waviness — strong phase distortion
    # reads as marble, not wood.
    grain = np.sin(math.tau * (xx * 26 + wood_n * 0.35))
    streaks = periodic_noise(size, 52, 3)
    wood = np.asarray((0.24, 0.095, 0.032))[None, None, :] + (
        grain * 0.05 + (streaks - 0.5) * 0.09
    )[:, :, None]
    outputs["wood"] = save_texture("wood_grain", wood)
    # Mottled two-tone grass with dry patches.
    g1 = periodic_noise(size, 61, 6)
    g2 = periodic_noise(size, 62, 3)
    grass = np.asarray((0.15, 0.26, 0.11))[None, None, :] + (
        (g1 - 0.5) * 0.09
    )[:, :, None] + np.maximum(g2 - 0.66, 0)[:, :, None] * np.asarray((0.22, 0.16, 0.02))[None, None, :]
    outputs["grass"] = save_texture("grass_mottled", grass)
    if SNOW_MODE:
        # Seamless blue-white powder with broad wind-packed variation and a
        # fine crystalline grain. Integer wave counts retain exact tiling.
        snow_large = periodic_noise(size, SEED + 701, 5)
        snow_fine = periodic_noise(size, SEED + 702, 7)
        yy_n, xx_n = np.mgrid[0:size, 0:size].astype(np.float32) / size
        wind = np.sin(math.tau * (xx_n * 3 + yy_n * 1)) * 0.018
        snow_rgb = np.asarray((0.82, 0.89, 0.94), dtype=np.float32)[None, None, :]
        snow_rgb = snow_rgb + (
            (snow_large - 0.5) * 0.105
            + (snow_fine - 0.5) * 0.035
            + wind
        )[:, :, None]
        outputs["snow_surface"] = save_texture("snow_surface", snow_rgb)
        packed = np.asarray((0.64, 0.70, 0.74), dtype=np.float32)[None, None, :]
        packed = packed + (
            (snow_large - 0.5) * 0.09
            + (snow_fine - 0.5) * 0.025
            - np.maximum(periodic_noise(size, SEED + 703, 3) - 0.63, 0) * 0.16
        )[:, :, None]
        outputs["snow_path"] = save_texture("snow_path", packed)
    return outputs


TEXTURES = generate_textures()


def material(
    name,
    color,
    roughness=0.75,
    metallic=0.0,
    emission=None,
    emission_strength=2.5,
    texture=None,
):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if texture:
        image = bpy.data.images.load(str(TEXTURES[texture]), check_existing=True)
        image.pack()
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = "Linear"
        tex.extension = "REPEAT"
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.use_backface_culling = True
    if emission:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission_strength
        else:
            bsdf.inputs["Emission"].default_value = (*emission, 1.0)
    return mat


def unlit_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 0.8
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


MAT = {
    "cream": material("MAT_CreamPlaster", (0.73, 0.61, 0.43), 0.86, texture="plaster_cream"),
    "cream2": material("MAT_WarmPlaster", (0.88, 0.76, 0.55), 0.82, texture="plaster_cream"),
    "rose": material("MAT_FadedRose", (0.56, 0.27, 0.22), 0.82, texture="plaster_rose"),
    "ochre": material("MAT_Ochre", (0.62, 0.40, 0.16), 0.84, texture="plaster_ochre"),
    "timber": material("MAT_DarkTimber", (0.115, 0.052, 0.024), 0.72, texture="wood"),
    "wood": material("MAT_Wood", (0.28, 0.12, 0.045), 0.72, texture="wood"),
    "stone": material("MAT_Stone", (0.31, 0.32, 0.30), 0.91, texture="stone"),
    "stone2": material("MAT_LightStone", (0.46, 0.45, 0.40), 0.9, texture="stone"),
    "tunnel": material("MAT_TunnelStone", (0.055, 0.060, 0.065), 0.96),
    "slate": material("MAT_SlateRoof", (0.095, 0.12, 0.14), 0.83, texture="roof_slate"),
    "tile": material("MAT_Terracotta", (0.43, 0.105, 0.045), 0.82, texture="roof_tile"),
    "green": material("MAT_DeepGreen", (0.055, 0.15, 0.095), 0.82, texture="roof_green"),
    "glass": material("MAT_WindowBlue", (0.07, 0.16, 0.20), 0.42, metallic=0.08),
    "glass_lit": material(
        "MAT_WindowGlowAmber", (0.64, 0.25, 0.055), 0.48,
        emission=(1.0, 0.22, 0.025), emission_strength=5.5,
    ),
    "glass_lit_soft": material(
        "MAT_WindowGlowGold", (0.58, 0.31, 0.09), 0.52,
        emission=(1.0, 0.38, 0.07), emission_strength=3.8,
    ),
    "gold": material("MAT_Gold", (0.72, 0.42, 0.08), 0.3, metallic=0.55),
    "road": material("MAT_Cobble", (0.24, 0.255, 0.25), 0.96, texture="cobble"),
    "road2": material("MAT_CobbleLight", (0.34, 0.34, 0.31), 0.96, texture="cobble"),
    "soil": material("MAT_Soil", (0.22, 0.18, 0.105), 0.98),
    "grass": material("MAT_Grass", (0.14, 0.245, 0.10), 0.96, texture="grass"),
    "leaf": material(
        "MAT_Leaves",
        (0.10, 0.15, 0.11) if SNOW_MODE else (0.075, 0.21, 0.075),
        0.92,
    ),
    "water": material("MAT_Water", (0.06, 0.25, 0.31), 0.25, metallic=0.1),
    "iron": material("MAT_Iron", (0.035, 0.04, 0.04), 0.42, metallic=0.72),
    "gate_iron": material("MAT_GateIron", (0.13, 0.14, 0.145), 0.38, metallic=0.64),
    "light": material(
        "MAT_LanternGlow", (1.0, 0.43, 0.08), 0.35,
        emission=(1.0, 0.22, 0.035),
        emission_strength=9.0 if DUSK_MODE else 2.5,
    ),
    "crystal": material(
        "MAT_ArcaneCrystal", (0.055, 0.42, 0.54), 0.24,
        emission=(0.035, 0.75, 1.0),
        emission_strength=8.0 if DUSK_MODE else 4.0,
    ),
    "mountain": unlit_material(
        "MAT_DistantMountainHaze",
        (0.10, 0.14, 0.22) if DUSK_MODE else (0.52, 0.64, 0.74),
    ),
    "snow": unlit_material(
        "MAT_SnowCaps",
        (0.34, 0.39, 0.50) if DUSK_MODE else (0.82, 0.88, 0.91),
    ),
    "horizon": unlit_material(
        "MAT_HorizonMist",
        (0.16, 0.22, 0.34) if DUSK_MODE else (0.58, 0.69, 0.78),
    ),
}
if SNOW_MODE:
    MAT["snow_surface"] = material(
        "MAT_SnowSurface", (0.82, 0.89, 0.94), 0.94,
        texture="snow_surface",
    )
    MAT["snow_path"] = material(
        "MAT_PackedSnow", (0.64, 0.70, 0.74), 0.96,
        texture="snow_path",
    )
ALL_MATERIALS = list(MAT.values())
MAT_INDEX = {m.name: i for i, m in enumerate(ALL_MATERIALS)}


def add_point_light(name, location, energy=320.0, color=(1.0, 0.28, 0.055), radius=1.0):
    """Create a dusk-only practical light without context-dependent operators."""
    if not DUSK_MODE:
        return None
    data = bpy.data.lights.new(name + "_Data", "POINT")
    data.energy = energy
    data.color = color
    data.shadow_soft_size = radius
    # Practical bulbs sit inside deliberately simple closed lantern meshes.
    # Disabling their own shadows prevents the casing from trapping all light.
    data.use_shadow = False
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    COLLECTIONS["Props"].objects.link(obj)
    return obj


# --- Main street centerline -------------------------------------------------
# Gentle S-curve running the full length of town. Endpoints return to x=0 so
# both the south gate and the castle gate stay on axis. Every building row,
# road strip, lantern and camera shares this lateral shift, which preserves
# alley widths and inter-row spacing while bending the whole town.
STREET_AMP = 6.5
STREET_Y0, STREET_Y1 = -92.0, 84.0


def street_x(y):
    return STREET_AMP * math.sin(math.tau * (y - STREET_Y0) / (STREET_Y1 - STREET_Y0))


def street_rot(y):
    slope = (STREET_AMP * math.tau / (STREET_Y1 - STREET_Y0)
             * math.cos(math.tau * (y - STREET_Y0) / (STREET_Y1 - STREET_Y0)))
    return -math.atan(slope)


class MeshBuilder:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.mat_ids = []

    def rotate_about(self, cx, cy, angle):
        if abs(angle) < 1e-6:
            return
        c, s = math.cos(angle), math.sin(angle)
        for i, (x, y, z) in enumerate(self.vertices):
            dx, dy = x - cx, y - cy
            self.vertices[i] = (cx + dx * c - dy * s, cy + dx * s + dy * c, z)

    def face(self, verts, mat):
        start = len(self.vertices)
        self.vertices.extend(verts)
        self.faces.append(tuple(range(start, start + len(verts))))
        self.mat_ids.append(MAT_INDEX[mat.name])

    def box(self, center, size, mat, rot=0.0):
        cx, cy, cz = center
        sx, sy, sz = (v * 0.5 for v in size)
        c, s = math.cos(rot), math.sin(rot)
        verts = []
        for x, y, z in (
            (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
            (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
        ):
            verts.append((cx + x * c - y * s, cy + x * s + y * c, cz + z))
        for q in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.faces.append(tuple(len(self.vertices) + i for i in q))
            self.mat_ids.append(MAT_INDEX[mat.name])
        self.vertices.extend(verts)

    def gable_roof(self, cx, cy, base_z, depth, width, height, mat):
        x0, x1 = cx - depth / 2, cx + depth / 2
        y0, y1 = cy - width / 2, cy + width / 2
        v = [(x0, y0, base_z), (x1, y0, base_z), (x0, y1, base_z),
             (x1, y1, base_z), (cx, y0, base_z + height), (cx, y1, base_z + height)]
        start = len(self.vertices)
        self.vertices.extend(v)
        for q in ((0, 1, 4), (2, 5, 3), (0, 4, 5, 2), (4, 1, 3, 5), (0, 2, 3, 1)):
            self.faces.append(tuple(start + i for i in q))
            self.mat_ids.append(MAT_INDEX[mat.name])

    def hip_roof(self, cx, cy, base_z, depth, width, height, mat):
        x0, x1 = cx - depth / 2, cx + depth / 2
        y0, y1 = cy - width / 2, cy + width / 2
        inset = min(width * 0.25, 1.8)
        v = [(x0, y0, base_z), (x1, y0, base_z), (x1, y1, base_z), (x0, y1, base_z),
             (cx, y0 + inset, base_z + height), (cx, y1 - inset, base_z + height)]
        start = len(self.vertices)
        self.vertices.extend(v)
        for q in ((0, 1, 4), (1, 2, 5, 4), (2, 3, 5), (3, 0, 4, 5), (0, 3, 2, 1)):
            self.faces.append(tuple(start + i for i in q))
            self.mat_ids.append(MAT_INDEX[mat.name])

    def cylinder(self, center, radius, depth, mat, segments=12, axis="Z"):
        cx, cy, cz = center
        start = len(self.vertices)
        for side in (-1, 1):
            for i in range(segments):
                a = math.tau * i / segments
                p, q = radius * math.cos(a), radius * math.sin(a)
                if axis == "Z":
                    self.vertices.append((cx + p, cy + q, cz + side * depth / 2))
                elif axis == "X":
                    self.vertices.append((cx + side * depth / 2, cy + p, cz + q))
                else:
                    self.vertices.append((cx + p, cy + side * depth / 2, cz + q))
        for i in range(segments):
            j = (i + 1) % segments
            self.faces.append((start + i, start + j, start + segments + j, start + segments + i))
            self.mat_ids.append(MAT_INDEX[mat.name])
        self.faces.append(tuple(start + i for i in reversed(range(segments))))
        self.mat_ids.append(MAT_INDEX[mat.name])
        self.faces.append(tuple(start + segments + i for i in range(segments)))
        self.mat_ids.append(MAT_INDEX[mat.name])

    def frustum(self, center_xy, base_z, bottom_radius, top_radius, height, mat, segments=16):
        """Capped vertical frustum, used where stacked cylinders look too blocky."""
        cx, cy = center_xy
        start = len(self.vertices)
        for z, radius in ((base_z, bottom_radius), (base_z + height, top_radius)):
            for i in range(segments):
                a = math.tau * i / segments
                self.vertices.append((cx + radius * math.cos(a),
                                      cy + radius * math.sin(a), z))
        for i in range(segments):
            j = (i + 1) % segments
            self.faces.append((start + i, start + j,
                               start + segments + j, start + segments + i))
            self.mat_ids.append(MAT_INDEX[mat.name])
        self.faces.append(tuple(start + i for i in reversed(range(segments))))
        self.mat_ids.append(MAT_INDEX[mat.name])
        self.faces.append(tuple(start + segments + i for i in range(segments)))
        self.mat_ids.append(MAT_INDEX[mat.name])

    def sphere(self, center, radius, mat, segments=10, rings=6):
        cx, cy, cz = center
        start = len(self.vertices)
        self.vertices.append((cx, cy, cz - radius))
        for ring in range(1, rings):
            phi = math.pi * ring / rings - math.pi / 2
            for i in range(segments):
                a = math.tau * i / segments
                self.vertices.append((cx + radius * math.cos(phi) * math.cos(a),
                                      cy + radius * math.cos(phi) * math.sin(a),
                                      cz + radius * math.sin(phi)))
        self.vertices.append((cx, cy, cz + radius))
        top = len(self.vertices) - 1
        for i in range(segments):
            ni = (i + 1) % segments
            self.faces.append((start, start + 1 + ni, start + 1 + i))
            self.mat_ids.append(MAT_INDEX[mat.name])
        for ring in range(rings - 2):
            base = start + 1 + ring * segments
            for i in range(segments):
                ni = (i + 1) % segments
                self.faces.append((base + i, base + ni, base + segments + ni, base + segments + i))
                self.mat_ids.append(MAT_INDEX[mat.name])
        last = start + 1 + (rings - 2) * segments
        for i in range(segments):
            ni = (i + 1) % segments
            self.faces.append((last + i, last + ni, top))
            self.mat_ids.append(MAT_INDEX[mat.name])

    def cone(self, center_xy, base_z, radius, height, mat, segments=12):
        cx, cy = center_xy
        start = len(self.vertices)
        for i in range(segments):
            a = math.tau * i / segments
            self.vertices.append((cx + radius * math.cos(a), cy + radius * math.sin(a), base_z))
        self.vertices.append((cx, cy, base_z + height))
        apex = start + segments
        for i in range(segments):
            self.faces.append((start + i, start + (i + 1) % segments, apex))
            self.mat_ids.append(MAT_INDEX[mat.name])
        self.faces.append(tuple(start + i for i in reversed(range(segments))))
        self.mat_ids.append(MAT_INDEX[mat.name])

    def beam(self, p1, p2, thickness, depth, mat):
        a, b = Vector(p1), Vector(p2)
        vec = b - a
        length = vec.length
        if length < 0.001:
            return
        tangent = vec / length
        normal = Vector((0, -tangent.z, tangent.y))
        across = Vector((depth / 2, 0, 0))
        normal *= thickness / 2
        verts = []
        for p in (a, b):
            verts += [tuple(p - across - normal), tuple(p + across - normal),
                      tuple(p + across + normal), tuple(p - across + normal)]
        start = len(self.vertices)
        self.vertices.extend(verts)
        for q in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                  (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
            self.faces.append(tuple(start + i for i in q))
            self.mat_ids.append(MAT_INDEX[mat.name])

    def object(self, name, collection):
        mesh = bpy.data.meshes.new(name + "_Mesh")
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.materials.clear()
        # Register only the materials this mesh actually uses; a full 24-slot
        # table on every object explodes into hundreds of submeshes in Unity.
        used = sorted(set(self.mat_ids))
        remap = {global_id: local_id for local_id, global_id in enumerate(used)}
        for global_id in used:
            mesh.materials.append(ALL_MATERIALS[global_id])
        for poly, idx in zip(mesh.polygons, self.mat_ids):
            poly.material_index = remap[idx]
        mesh.update()
        # Box-projected UVs are deterministic, tile cleanly, and work on the
        # already-batched meshes without context-sensitive unwrap operators.
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uv_scale = 0.14
        for poly in mesh.polygons:
            n = poly.normal
            dominant = max(range(3), key=lambda axis: abs(n[axis]))
            for loop_index in poly.loop_indices:
                co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
                if dominant == 0:
                    uv = (co.y * uv_scale, co.z * uv_scale)
                elif dominant == 1:
                    uv = (co.x * uv_scale, co.z * uv_scale)
                else:
                    uv = (co.x * uv_scale, co.y * uv_scale)
                uv_layer.data[loop_index].uv = uv
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        return obj


def make_house(name, cx, cy, width, depth, front_sign, style_seed, stone=False, narrow=False, rot=0.0):
    r = random.Random(style_seed)
    light_r = random.Random(SEED * 17 + style_seed)
    mb = MeshBuilder()
    floors = r.choice((2, 2, 3, 3, 4))
    floor_h = r.uniform(2.75, 3.15)
    wall_h = floors * floor_h
    floor_lights = []
    for fl in range(floors):
        chance = 0.34 if fl == 0 else (0.52 if fl == floors - 1 else 0.68)
        if (DUSK_MODE or SNOW_MODE) and light_r.random() < chance:
            floor_lights.append(
                MAT["glass_lit"] if light_r.random() < 0.58 else MAT["glass_lit_soft"]
            )
        else:
            floor_lights.append(None)
    facade_x = cx + front_sign * depth / 2
    wall_mat = MAT["stone2"] if stone else r.choice((MAT["cream"], MAT["cream2"], MAT["ochre"], MAT["rose"]))
    roof_mat = r.choice((MAT["tile"], MAT["slate"], MAT["green"]))
    jetty = 0.34 if not stone and floors > 2 else 0.16
    if narrow:
        jetty = 0.10

    mb.box((cx, cy, wall_h / 2), (depth, width, wall_h), wall_mat)
    if jetty:
        mb.box((cx + front_sign * jetty / 2, cy, floor_h + (wall_h - floor_h) / 2),
               (depth + jetty, width, wall_h - floor_h), wall_mat)
        facade_x += front_sign * jetty

    roof_depth = depth + 1.55
    roof_width = width + 0.80
    roof_h = r.uniform(2.1, 3.8)
    is_gable = r.random() < 0.72
    if is_gable:
        mb.gable_roof(cx, cy, wall_h, roof_depth, roof_width, roof_h, roof_mat)
    else:
        mb.hip_roof(cx, cy, wall_h, roof_depth, roof_width, roof_h, roof_mat)
    # Deep timber fascia makes the eave overhang readable from street level.
    for sx in (-1, 1):
        mb.box((cx + sx * roof_depth / 2, cy, wall_h + 0.03),
               (0.18, roof_width, 0.30), MAT["timber"])

    # Occasional dormers on the street-facing slope break up the roofscape.
    # Gable roofs only: the slope height at any X is analytic, so the dormer
    # can sit ON the surface instead of sinking into it.
    if is_gable and not narrow and r.random() < 0.24:
        def slope_z(x):
            return wall_h + roof_h * max(0.0, 1.0 - abs(x - cx) / (roof_depth / 2))
        count = 1 if width < 8 or r.random() < 0.5 else 2
        for k in range(count):
            dy = cy + (k - (count - 1) / 2) * width * 0.30 + r.uniform(-0.3, 0.3)
            x_front = cx + front_sign * roof_depth * 0.30
            base_z = slope_z(x_front) - 0.35
            dx = x_front - front_sign * 0.45
            mb.box((dx, dy, base_z + 0.55), (1.15, 1.2, 1.1), wall_mat)
            dormer_glass = floor_lights[-1] or MAT["glass"]
            mb.box((x_front + front_sign * 0.13, dy, base_z + 0.62),
                   (0.06, 0.62, 0.68), dormer_glass)
            mb.box((x_front + front_sign * 0.16, dy, base_z + 0.62), (0.05, 0.08, 0.68), MAT["timber"])
            # Tiny ridge-forward roof: two slopes + front gable triangle.
            rw, rh = 0.85, 0.62
            f = x_front + front_sign * 0.25
            b = dx - front_sign * 0.75
            top = base_z + 1.06
            mb.face([(b, dy - rw, top), (f, dy - rw, top),
                     (f, dy, top + rh), (b, dy, top + rh)], roof_mat)
            mb.face([(f, dy + rw, top), (b, dy + rw, top),
                     (b, dy, top + rh), (f, dy, top + rh)], roof_mat)
            mb.face([(f, dy - rw, top), (f, dy + rw, top), (f, dy, top + rh)], wall_mat)

    chimney_w = r.uniform(0.5, 0.8)

    # Shared panel grid: timber studs, diagonal braces and windows all live on
    # the same bays, so braces never cross a window opening. Roughly every
    # third upper-floor panel takes a brace; the rest take windows.
    bays = max(2, int(width / 2.8))

    def is_diag_panel(bay, fl):
        return (not stone) and fl >= 1 and (bay * 2 + fl) % 3 == 0

    if not stone:
        beam_x = facade_x + front_sign * 0.11
        for z in (floor_h, wall_h - 0.22):
            mb.box((beam_x, cy, z), (0.24, width, 0.22), MAT["timber"])
        for y in (cy - width / 2 + 0.18, cy + width / 2 - 0.18):
            mb.box((beam_x, y, wall_h * 0.55), (0.24, 0.22, wall_h * 0.88), MAT["timber"])
        for bay in range(1, bays):
            y = cy - width / 2 + width * bay / bays
            mb.box((beam_x, y, (floor_h + wall_h) / 2), (0.24, 0.18, wall_h - floor_h), MAT["timber"])
        for bay in range(bays):
            y0 = cy - width / 2 + width * bay / bays + 0.28
            y1 = cy - width / 2 + width * (bay + 1) / bays - 0.28
            for fl in range(1, floors):
                if not is_diag_panel(bay, fl):
                    continue
                z0 = fl * floor_h + 0.25
                z1 = (fl + 1) * floor_h - 0.25
                if bay % 2:
                    z0, z1 = z1, z0
                mb.beam((beam_x, y0, z0), (beam_x, y1, z1), 0.18, 0.24, MAT["timber"])
                mb.beam((beam_x, y0, z1), (beam_x, y1, z0), 0.18, 0.24, MAT["timber"])
    window_x = facade_x + front_sign * 0.15
    # Depth layering: glass recessed behind the wall plane, frame slightly
    # proud, sill stepping out further. Reads as a real opening up close.
    glass_x = facade_x - front_sign * 0.05
    frame_x = facade_x + front_sign * 0.04
    has_shutters = (not stone) and (not narrow) and r.random() < 0.30
    shutter_angle = math.radians(r.uniform(12, 26))
    for fl in range(floors):
        z = fl * floor_h + 1.65
        for bay in range(bays):
            if fl == 0 and bay == bays // 2:
                continue
            if is_diag_panel(bay, fl):
                continue
            y = cy - width / 2 + width * (bay + 0.5) / bays
            ww = min(1.15, width / bays * 0.53)
            wh = 1.28
            window_mat = floor_lights[fl]
            if window_mat is None or light_r.random() > 0.82:
                window_mat = MAT["glass"]
            mb.box((glass_x, y, z), (0.10, ww, wh), window_mat)
            for oy in (-ww / 2 - 0.06, ww / 2 + 0.06):
                mb.box((frame_x, y + oy, z), (0.14, 0.12, wh + 0.24), MAT["timber"])
            mb.box((frame_x, y, z + wh / 2 + 0.06), (0.14, ww + 0.28, 0.12), MAT["timber"])
            mb.box((facade_x + front_sign * 0.07, y, z - wh / 2 - 0.07),
                   (0.24, ww + 0.34, 0.10), MAT["wood"])
            mb.box((glass_x + front_sign * 0.03, y, z), (0.09, 0.06, wh), MAT["timber"])
            mb.box((glass_x + front_sign * 0.03, y, z), (0.09, ww, 0.06), MAT["timber"])
            if has_shutters and fl >= 1:
                sw = ww * 0.52
                off_x = math.sin(shutter_angle) * sw / 2
                off_y = math.cos(shutter_angle) * sw / 2
                for side in (-1, 1):
                    hinge_y = y + side * (ww / 2 + 0.10)
                    mb.box((facade_x + front_sign * (0.06 + off_x),
                            hinge_y + side * off_y, z),
                           (0.05, sw, wh + 0.10), MAT["wood"],
                           rot=-side * front_sign * shutter_angle)

    door_y = cy + r.uniform(-0.16, 0.16) * width
    mb.box((window_x, door_y, 1.1), (0.15, 1.22, 2.2), MAT["wood"])
    for oy in (-0.68, 0.68):
        mb.box((window_x + front_sign * 0.07, door_y + oy, 1.18), (0.13, 0.12, 2.48), MAT["timber"])
    mb.box((window_x + front_sign * 0.07, door_y, 2.4), (0.13, 1.48, 0.14), MAT["timber"])

    if not narrow and r.random() < 0.62:
        awning_z = 2.75
        awning_w = min(width * 0.48, 3.5)
        awning_depth = 1.25
        awning_cx = window_x + front_sign * awning_depth * 0.5
        mb.box((awning_cx, door_y, awning_z), (awning_depth, awning_w, 0.15),
               r.choice((MAT["tile"], MAT["green"], MAT["cream2"])))
        for yy in (door_y - awning_w / 2 + 0.14, door_y + awning_w / 2 - 0.14):
            mb.beam((window_x, yy, awning_z), (window_x + front_sign * awning_depth, yy, awning_z - 0.38),
                    0.09, 0.09, MAT["timber"])

    if not narrow and r.random() < 0.38:
        sy = cy + (0.38 if r.random() < 0.5 else -0.38) * width
        mb.box((window_x + front_sign * 0.78, sy, 2.85), (1.42, 0.14, 0.11), MAT["iron"])
        mb.box((window_x + front_sign * 1.45, sy, 2.38), (0.10, 0.86, 0.78), MAT["wood"])

    chimney_x = cx - front_sign * depth * 0.18
    chimney_y = cy + r.uniform(-0.3, 0.3) * width
    mb.box((chimney_x, chimney_y, wall_h + roof_h * 0.62),
           (chimney_w, chimney_w, roof_h + 1.2), MAT["stone"])
    mb.box((chimney_x, chimney_y, wall_h + roof_h * 1.22 + 0.05),
           (chimney_w + 0.18, chimney_w + 0.18, 0.16), MAT["stone2"])

    # Sparse side/gable decoration pass. It is seeded independently so adding
    # it never changes the established house shapes or facade details.
    decor_r = random.Random(SEED * 31 + style_seed)
    forced_decor = {
        "House_Main_E_North_00": (-1, 0),  # prominent south gable: ivy
        "House_Plaza_E_00": (-1, 2),       # tavern-side gable: trellis
    }
    if not narrow and (name in forced_decor or decor_r.random() < 0.27):
        side = forced_decor[name][0] if name in forced_decor else (-1 if decor_r.random() < 0.5 else 1)
        side_y = cy + side * (width / 2 + 0.07)
        deco_x = cx + decor_r.uniform(-0.16, 0.16) * depth
        variant = forced_decor[name][1] if name in forced_decor else decor_r.randrange(4)
        if variant == 0:
            # Branching climbing stems with irregular flat leaf clusters.
            vine_top = min(wall_h * 0.78, 7.2)
            mb.beam((deco_x, side_y, 0.65),
                    (deco_x - 1.05, side_y, vine_top * 0.72),
                    0.10, 0.10, MAT["wood"])
            mb.beam((deco_x, side_y, 0.75),
                    (deco_x + 1.15, side_y, vine_top),
                    0.09, 0.10, MAT["wood"])
            for k in range(11):
                z = 0.95 + k * min(0.66, wall_h * 0.065)
                x = deco_x + math.sin(k * 1.55) * (0.62 + k * 0.045)
                mb.box((x, side_y + side * 0.025, z),
                       (decor_r.uniform(0.58, 0.96), 0.09,
                        decor_r.uniform(0.38, 0.62)), MAT["leaf"])
                if k in (4, 7, 9):
                    branch_side = -1 if k % 2 else 1
                    mb.box((x + branch_side * 0.72, side_y + side * 0.03, z + 0.25),
                           (decor_r.uniform(0.48, 0.78), 0.09,
                            decor_r.uniform(0.34, 0.55)), MAT["leaf"])
        elif variant == 1:
            # Wall banner with a small gold rail and lower tail.
            banner_z = min(wall_h - 1.6, max(3.1, wall_h * 0.60))
            banner_mat = MAT["rose"] if decor_r.random() < 0.5 else MAT["ochre"]
            mb.box((deco_x, side_y, banner_z), (1.55, 0.10, 2.65), banner_mat)
            mb.box((deco_x, side_y + side * 0.03, banner_z + 1.42),
                   (1.95, 0.13, 0.13), MAT["gold"])
            mb.box((deco_x, side_y + side * 0.04, banner_z),
                   (0.16, 0.13, 1.55), MAT["gold"])
        elif variant == 2:
            # Timber trellis, optionally beginning to green over.
            trellis_z = min(wall_h - 1.6, max(2.8, wall_h * 0.48))
            for ox in (-0.9, -0.3, 0.3, 0.9):
                mb.box((deco_x + ox, side_y, trellis_z),
                       (0.10, 0.10, 3.1), MAT["wood"])
            for oz in (-1.2, -0.4, 0.4, 1.2):
                mb.box((deco_x, side_y, trellis_z + oz),
                       (2.15, 0.10, 0.10), MAT["wood"])
            for ox, oz in ((-0.72, -0.65), (0.45, 0.15), (-0.15, 0.85)):
                mb.box((deco_x + ox, side_y + side * 0.035, trellis_z + oz),
                       (0.48, 0.09, 0.38), MAT["leaf"])
        else:
            # One small framed side window breaks an otherwise blank wall.
            wz = min(wall_h - 1.45, max(2.8, wall_h * 0.58))
            side_glass = (
                MAT["glass_lit_soft"]
                if (DUSK_MODE or SNOW_MODE) and decor_r.random() < 0.7
                else MAT["glass"]
            )
            mb.box((deco_x, side_y, wz), (1.15, 0.10, 1.35), side_glass)
            for ox in (-0.68, 0.68):
                mb.box((deco_x + ox, side_y + side * 0.035, wz),
                       (0.12, 0.12, 1.62), MAT["timber"])
            for oz in (-0.78, 0.78):
                mb.box((deco_x, side_y + side * 0.035, wz + oz),
                       (1.48, 0.12, 0.12), MAT["timber"])
    mb.rotate_about(cx, cy, rot)
    return mb.object(name, COLLECTIONS["Town"])


def fill_row(prefix, x, y0, y1, depth, front_sign, seed_base, narrow=False):
    total = y1 - y0
    widths = []
    used = 0.0
    r = random.Random(seed_base)
    while total - used > 11.5:
        w = r.uniform(6.2, 9.2)
        widths.append(w)
        used += w
    widths.append(total - used)
    cursor = y0
    for i, width in enumerate(widths):
        y_c = cursor + width / 2
        # Follow the street curve: lateral shift plus tangent rotation. The
        # slight cursor under-advance overlaps neighbours a hair so convex-
        # side wedge gaps never open between rotated houses.
        make_house(f"House_{prefix}_{i:02d}", street_x(y_c) + x, y_c, width + 0.25, depth,
                   front_sign, seed_base * 100 + i, stone=(i % 7 in (3, 6)), narrow=narrow,
                   rot=street_rot(y_c))
        cursor += width * 0.995


def build_streets_and_houses():
    ground = MeshBuilder()
    ground.box((0, 0, -0.42), (235, 235, 0.75), MAT["grass"])
    # Far apron out to the mountain ring: without it, downward sight lines
    # from elevated cameras slip past the mist ring into black sub-horizon sky.
    ground.box((0, 0, -1.3), (760, 760, 0.8), MAT["grass"])
    def road_strip(x_offset, half_w, y0, y1, mat, step=4.0):
        # Curve-following surface: top quads at z=0.076 plus side skirts so
        # the raised road edge never shows a hollow underside.
        ys = []
        y = y0
        while y < y1:
            ys.append(y)
            y += step
        ys.append(y1)
        for a, b in zip(ys, ys[1:]):
            xla, xra = street_x(a) + x_offset - half_w, street_x(a) + x_offset + half_w
            xlb, xrb = street_x(b) + x_offset - half_w, street_x(b) + x_offset + half_w
            ground.face([(xla, a, 0.076), (xra, a, 0.076),
                         (xrb, b, 0.076), (xlb, b, 0.076)], mat)
            ground.face([(xla, a, -0.06), (xla, a, 0.076), (xlb, b, 0.076), (xlb, b, -0.06)], mat)
            ground.face([(xra, a, 0.076), (xra, a, -0.06), (xrb, b, -0.06), (xrb, b, 0.076)], mat)

    road_strip(0, 6.25, -92, 84, MAT["road"])
    ground.box((0, 25, 0.025), (61, 30, 0.14), MAT["road2"])
    ground.box((27.5, -18, 0.02), (51, 4.2, 0.12), MAT["road"])
    ground.box((-28.5, 51, 0.02), (49, 4.2, 0.12), MAT["road"])
    road_strip(27, 2.3, -84, -18, MAT["road"])
    road_strip(-27, 1.5, -53, 17, MAT["road"])
    ground.box((46, -8, 0.02), (4.0, 142, 0.12), MAT["road"])
    ground.box((-46, -8, 0.02), (4.0, 142, 0.12), MAT["road"])
    ground.object("Ground_RoadsAndTerrain", COLLECTIONS["Ground"])

    rows = [
        ("Main_W_South", -10.25, -82, 9, 8, 1, 10),
        ("Main_E_South", 10.25, -82, 9, 8, -1, 20),
        ("Main_W_North", -10.25, 41, 72, 8, 1, 30),
        ("Main_E_North", 10.25, 41, 72, 8, -1, 40),
        ("WestLane_Outer", -33.0, -50, 16, 9, 1, 50),
        # The two EastLane rows face each other across a 3.75m alley;
        # narrow=True drops awnings/signs and shrinks jetties so the lane
        # stays walkable and visually clean.
        ("EastLane_Outer", 33.0, -50, 9, 9, -1, 60, True),
        ("EastLane_Inner", 21.05, -50, 9, 8.5, 1, 65, True),
        ("WestPerimeter", -57.0, -72, 69, 9, 1, 67),
        ("EastPerimeter", 57.0, -72, 69, 9, -1, 68),
    ]
    if TEST_MODE:
        rows = rows[:4]
    for args in rows:
        fill_row(*args)

    # Plaza perimeter: continuous facades facing the square.
    fill_row("Plaza_W", -34.5, 9, 42, 9, 1, 70)
    fill_row("Plaza_E", 34.5, 21, 42, 9, -1, 80)


def make_tavern():
    """A prominent three-storey inn anchoring the plaza's southeast corner."""
    mb = MeshBuilder()
    ty = 15.0
    tx = street_x(ty) + 34.5
    rot = street_rot(ty)
    depth, width = 10.2, 12.0
    floor_h, wall_h = 3.2, 9.6
    front_x = tx - depth / 2

    # Stone public floor and jettied plaster accommodation floors.
    mb.box((tx, ty, floor_h / 2), (depth, width, floor_h), MAT["stone2"])
    upper_cx = tx - 0.24
    upper_depth = depth + 0.48
    mb.box((upper_cx, ty, floor_h + (wall_h - floor_h) / 2),
           (upper_depth, width + 0.25, wall_h - floor_h), MAT["cream2"])
    upper_front = upper_cx - upper_depth / 2
    mb.gable_roof(upper_cx, ty, wall_h, depth + 1.65, width + 1.0, 4.0, MAT["tile"])

    # Plaza-facing half-timber grid with broad crossed braces.
    for z in (floor_h + 0.08, floor_h * 2, wall_h - 0.18):
        mb.box((upper_front - 0.10, ty, z), (0.22, width + 0.28, 0.22), MAT["timber"])
    for y in (ty - width / 2 + 0.18, ty - 3.0, ty, ty + 3.0, ty + width / 2 - 0.18):
        mb.box((upper_front - 0.10, y, 6.35), (0.22, 0.20, 6.0), MAT["timber"])
    for y0, y1, reverse in ((ty - 5.7, ty - 3.2, False),
                            (ty - 2.8, ty - 0.25, True),
                            (ty + 0.25, ty + 2.8, False),
                            (ty + 3.2, ty + 5.7, True)):
        z0, z1 = (3.45, 6.15) if not reverse else (6.15, 3.45)
        mb.beam((upper_front - 0.12, y0, z0),
                (upper_front - 0.12, y1, z1), 0.17, 0.22, MAT["timber"])

    tavern_rng = random.Random(SEED + 2201)

    def window_material(index):
        roll = tavern_rng.random()
        if DUSK_MODE and roll < 0.88:
            return MAT["glass_lit"] if index % 3 else MAT["glass_lit_soft"]
        if SNOW_MODE and roll < 0.68:
            return MAT["glass_lit_soft"]
        return MAT["glass"]

    # Upper guest-room windows and warm ground-floor taproom windows.
    for floor, z in ((1, 4.85), (2, 8.0)):
        for i, y in enumerate((ty - 4.25, ty - 2.1, ty, ty + 2.1, ty + 4.25)):
            mat = window_material(floor * 10 + i)
            mb.box((upper_front - 0.16, y, z), (0.12, 1.02, 1.30), mat)
            for oy in (-0.59, 0.59):
                mb.box((upper_front - 0.22, y + oy, z),
                       (0.16, 0.12, 1.52), MAT["timber"])
            for oz in (-0.75, 0.75):
                mb.box((upper_front - 0.22, y, z + oz),
                       (0.16, 1.30, 0.12), MAT["timber"])

    door_y = ty + 0.25
    mb.box((front_x - 0.08, door_y, 1.34), (0.18, 1.55, 2.68), MAT["wood"])
    for y in (ty - 4.15, ty - 2.25, ty + 2.35, ty + 4.25):
        mat = window_material(int(y * 10))
        mb.box((front_x - 0.08, y, 1.62), (0.16, 1.12, 1.45), mat)
        for oy in (-0.65, 0.65):
            mb.box((front_x - 0.16, y + oy, 1.62),
                   (0.15, 0.12, 1.68), MAT["timber"])
        mb.box((front_x - 0.16, y, 0.82), (0.22, 1.42, 0.14), MAT["stone"])

    # Entrance porch with deep roof and two stout timber posts.
    porch_x = front_x - 1.35
    mb.box((porch_x, door_y, 3.05), (2.8, 3.2, 0.18), MAT["green"])
    for y in (door_y - 1.35, door_y + 1.35):
        mb.box((front_x - 2.48, y, 1.55), (0.20, 0.20, 3.0), MAT["timber"])
    mb.box((front_x - 1.30, door_y, 0.16), (2.8, 2.4, 0.26), MAT["stone2"])

    # Perpendicular hanging sign: bracket, hangers, timber board and emblem.
    sign_y = ty + 4.65
    mb.box((front_x - 0.88, sign_y, 4.15), (1.75, 0.13, 0.13), MAT["iron"])
    mb.box((front_x - 1.55, sign_y, 3.78), (0.12, 0.13, 0.78), MAT["iron"])
    mb.box((front_x - 1.55, sign_y, 3.25), (1.25, 0.18, 0.90), MAT["wood"])
    mb.cylinder((front_x - 1.55, sign_y - 0.12, 3.25),
                0.24, 0.10, MAT["gold"], 10, axis="Y")

    # Two chimneys with cap stones make the inn read as a busy kitchen.
    for y in (ty - 3.4, ty + 3.35):
        mb.box((tx + 1.7, y, 11.65), (0.72, 0.72, 4.5), MAT["stone"])
        mb.box((tx + 1.7, y, 13.92), (0.95, 0.95, 0.18), MAT["stone2"])

    # Outdoor drinking area kept tight to the facade and clear of the fountain.
    for y in (ty - 3.0, ty + 3.5):
        table_x = front_x - 2.75
        mb.cylinder((table_x, y, 0.80), 0.72, 0.16, MAT["wood"], 12)
        mb.cylinder((table_x, y, 0.42), 0.12, 0.76, MAT["timber"], 8)
        for oy in (-1.05, 1.05):
            mb.box((table_x, y + oy, 0.48), (1.45, 0.38, 0.16), MAT["wood"])
            mb.box((table_x, y + oy, 0.24), (1.15, 0.16, 0.42), MAT["timber"])
    barrel_x, barrel_y = front_x - 2.65, ty + 0.35
    mb.cylinder((barrel_x, barrel_y, 0.58), 0.54, 1.10, MAT["wood"], 10)
    for z in (0.18, 0.58, 0.98):
        mb.cylinder((barrel_x, barrel_y, z), 0.56, 0.08, MAT["iron"], 10)

    mb.rotate_about(tx, ty, rot)
    mb.object("Landmark_Tavern", COLLECTIONS["Town"])

    if DUSK_MODE:
        c, s = math.cos(rot), math.sin(rot)
        for i, local_y in enumerate((door_y - 1.15, door_y + 1.15)):
            lx, ly = front_x - 1.0, local_y
            dx, dy = lx - tx, ly - ty
            add_point_light(
                f"Prop_TavernPorchLamp_{i}",
                (tx + dx * c - dy * s, ty + dx * s + dy * c, 2.65),
                energy=360.0, radius=0.8,
            )

def add_cobbles():
    mb = MeshBuilder()
    r = random.Random(SEED + 22)
    # Full mode uses hand-scale pavers. Besides improving first-person detail,
    # this keeps the delivered asset in the specified mid-poly budget.
    scale = 0.50 if not TEST_MODE else 2.2
    # The road surface itself is a flat textured strip; geometry is reserved
    # for a sparse scatter of slightly proud accent stones. A full paver grid
    # was ~78% of the whole scene's triangles and made colliders bumpy.
    # Worn patches instead of lone floaters: stones cluster in small repaired
    # areas, sit half-sunk into the road (top ~3cm proud), same-family color.
    def patches(x0, x1, y0, y1, count, follow_curve=False):
        for _ in range(count):
            px, py = r.uniform(x0, x1), r.uniform(y0, y1)
            for _ in range(r.randint(5, 11)):
                ox, oy = px + r.gauss(0, 0.85), py + r.gauss(0, 0.85)
                if not (x0 - 0.4 < ox < x1 + 0.4 and y0 < oy < y1):
                    continue
                shift = street_x(oy) if follow_curve else 0.0
                mb.box((ox + shift, oy, 0.095), (r.uniform(0.45, 0.7) * scale * 1.6,
                                                 r.uniform(0.3, 0.5) * scale * 1.6, 0.05),
                       MAT["road2"] if r.random() < 0.2 else MAT["road"],
                       r.uniform(-0.4, 0.4))
    patches(-5.2, 5.2, -86.0, 80.0, 8 if TEST_MODE else 55, follow_curve=True)
    if not TEST_MODE:
        patches(-29.0, 29.0, 11.0, 39.0, 30)
    mb.object("Ground_Cobblestones", COLLECTIONS["Ground"])


def make_castle():
    mb = MeshBuilder()
    # Layered court, curtain wall, keep and roofed gatehouse. The central
    # volumes are deliberately split around a 8.4m corridor: the approach is
    # a real passage through the gatehouse and keep, not a dark panel pasted
    # onto a solid wall.
    gate_half = 4.2
    for side in (-1, 1):
        mb.box((side * 18.1, 104, 2.2), (27.8, 22, 4.4), MAT["stone"])
        mb.box((side * 8.85, 101, 6.7), (9.3, 21, 9.4), MAT["stone2"])
    mb.box((0, 101, 27.7), (27, 21, 32.6), MAT["stone2"])
    mb.hip_roof(0, 101, 44, 29.5, 23.5, 8.0, MAT["slate"])
    for side in (-1, 1):
        mb.box((side * 16.6, 92, 11), (24.8, 4.2, 18), MAT["stone"])
    mb.box((0, 92, 15.7), (8.4, 4.2, 8.6), MAT["stone"])
    mb.box((-28, 103, 13), (4.2, 25, 22), MAT["stone"])
    mb.box((28, 103, 13), (4.2, 25, 22), MAT["stone"])
    for side in (-1, 1):
        mb.box((side * 6.725, 88.5, 12), (5.05, 9.5, 22), MAT["stone2"])
    mb.box((0, 88.5, 17.2), (8.4, 9.5, 11.6), MAT["stone2"])
    mb.hip_roof(0, 88.5, 23, 11.2, 21.0, 5.4, MAT["green"])

    # Continuous tunnel shell and a slightly raised floor lead the eye through
    # the gate into the rear court.
    tunnel_mid_y = 97.6
    tunnel_depth = 27.6
    mb.box((0, tunnel_mid_y, 0.12), (7.8, tunnel_depth, 0.20), MAT["stone2"])
    for side in (-1, 1):
        mb.box((side * 4.05, tunnel_mid_y, 5.65),
               (0.30, tunnel_depth, 10.9), MAT["stone"])
    mb.box((0, tunnel_mid_y, 11.18), (8.1, tunnel_depth, 0.34), MAT["tunnel"])

    # Stepped stone spandrels turn the rectangular structural opening into a
    # readable arch; a proper extruded voussoir ring hides the small steps.
    arch_y_front, arch_y_back = 83.42, 83.92
    spring_z, inner_r, outer_r = 7.25, 3.55, 4.32
    strips = 14
    strip_w = inner_r * 2 / strips
    for j in range(strips):
        x = -inner_r + (j + 0.5) * strip_w
        arch_z = spring_z + math.sqrt(max(0.0, inner_r * inner_r - x * x))
        h = 11.42 - arch_z
        if h > 0.02:
            mb.box((x, 83.66, arch_z + h / 2), (strip_w + 0.03, 0.50, h), MAT["stone2"])

    for side in (-1, 1):
        for z in (1.25, 3.35, 5.45):
            mb.box((side * 3.88, 83.62, z), (0.72, 0.62, 1.72), MAT["stone2"])

    for j in range(12):
        a0, a1 = j * math.pi / 12, (j + 1) * math.pi / 12
        verts = []
        for y in (arch_y_front, arch_y_back):
            verts.extend([
                (inner_r * math.cos(a0), y, spring_z + inner_r * math.sin(a0)),
                (outer_r * math.cos(a0), y, spring_z + outer_r * math.sin(a0)),
                (outer_r * math.cos(a1), y, spring_z + outer_r * math.sin(a1)),
                (inner_r * math.cos(a1), y, spring_z + inner_r * math.sin(a1)),
            ])
        start = len(mb.vertices)
        mb.vertices.extend(verts)
        for q in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                  (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
            mb.faces.append(tuple(start + i for i in q))
            mb.mat_ids.append(MAT_INDEX[MAT["stone2"].name])

    # A raised portcullis is visible deep inside while leaving a human-scale
    # clear passage beneath it.
    port_y = 105.2
    for x in (-3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0):
        mb.box((x, port_y, 7.3), (0.13, 0.16, 7.2), MAT["gate_iron"])
    for z in (4.2, 6.8, 9.4, 10.75):
        mb.box((0, port_y, z), (6.6, 0.18, 0.15), MAT["gate_iron"])

    # Projecting gallery/corbels and compact guard rooms add a defensive,
    # occupied silhouette without competing with the main towers.
    mb.box((0, 83.25, 18.4), (18.8, 1.25, 0.72), MAT["stone"])
    for x in (-7.4, -3.7, 0, 3.7, 7.4):
        mb.box((x, 83.45, 17.25), (0.72, 1.35, 1.7), MAT["stone2"])
    for side in (-1, 1):
        gx = side * 12.0
        mb.box((gx, 86.9, 3.25), (4.5, 6.0, 6.5), MAT["stone"])
        mb.hip_roof(gx, 86.9, 6.5, 5.0, 6.5, 2.0, MAT["slate"])
        mb.box((gx, 83.84, 1.3), (1.25, 0.16, 2.4), MAT["wood"])
        mb.box((gx, 83.72, 4.35), (0.62, 0.14, 1.25), MAT["glass"])
        if DUSK_MODE:
            mb.box((side * 3.45, 83.26, 4.45), (0.24, 0.18, 0.42), MAT["light"])

    # Towers now use wall-top coordinates; roof eaves sit directly on the drums.
    towers = [(-23, 92, 6.4, 35), (23, 92, 6.4, 35),
              (-20, 108, 7.0, 43), (20, 108, 7.0, 43),
              # Rear-set central tower preserves the landmark silhouette
              # without plugging the newly open gate tunnel.
              (0, 119, 7.8, 61)]
    for i, (x, y, radius, height) in enumerate(towers):
        mb.cylinder((x, y, height / 2), radius, height, MAT["stone2"], 14)
        mb.cylinder((x, y, height + 0.2), radius + 1.15, 0.8, MAT["stone"], 14)
        mb.cone((x, y), height + 0.6, radius + 1.25, 10.5 if i < 4 else 13.0,
                MAT["green"] if i % 2 == 0 else MAT["slate"], 14)
        # Arrow-slit windows on the street-facing tower drums.
        for z in range(10, int(height - 5), 8):
            mb.box((x, y - radius - 0.08, z), (0.8, 0.20, 2.2), MAT["glass"])

    # Crenellations on curtain wall and keep provide a readable defensive crown.
    for x in range(-27, 28, 4):
        mb.box((x, 89.75, 20.8), (2.0, 4.7, 2.2), MAT["stone2"])
    for x in range(-12, 13, 4):
        mb.box((x, 90.3, 45.2), (2.0, 1.0, 2.4), MAT["stone2"])

    for z in (16, 24, 32, 40):
        for x in (-8, -3, 3, 8):
            mb.box((x, 90.38, z), (1.0, 0.24, 2.3), MAT["glass"])
    for x in (-6.0, 6.0):
        mb.box((x, 90.04, 28), (2.4, 0.12, 7.5), MAT["rose"])
        mb.box((x, 89.94, 30.5), (0.3, 0.12, 2.0), MAT["gold"])
    mb.object("Castle_MainComplex", COLLECTIONS["Castle"])
    if DUSK_MODE:
        for side in (-1, 1):
            add_point_light(
                f"Prop_CastleGateTorch_{'E' if side > 0 else 'W'}",
                (side * 3.45, 82.85, 4.45), energy=480.0, radius=0.75,
            )


def make_castle_inner_details():
    """Occupied inner-bailey details without changing the castle silhouette."""
    mb = MeshBuilder()
    # Side courts fit between the front drum towers and perimeter roads. The
    # gate axis remains completely clear for the tunnel sight line.
    for x in (-36.0, 36.0):
        mb.box((x, 90.0, 0.08), (11.0, 14.0, 0.16), MAT["road"])
        # Pale edge stones make the court boundary legible from above.
        for ex in (-5.35, 5.35):
            mb.box((x + ex, 90.0, 0.17), (0.35, 14.0, 0.18), MAT["stone2"])
        for ey in (-6.85, 6.85):
            mb.box((x, 90.0 + ey, 0.17), (11.0, 0.35, 0.18), MAT["stone2"])

    # Broad, shallow entry steps meet the tunnel's raised floor.
    for y, width, height in ((82.25, 9.0, 0.06),
                             (82.65, 8.5, 0.10),
                             (83.05, 8.0, 0.14)):
        mb.box((0, y, height / 2), (width, 0.78, height), MAT["stone2"])

    # Two banner poles frame the entrance while staying outside the tower
    # footprints and clear of the central approach.
    for i, x in enumerate((-34.0, 34.0)):
        mb.cylinder((x, 84.7, 4.6), 0.10, 9.2, MAT["iron"], 8)
        mb.cylinder((x, 84.7, 0.18), 0.42, 0.30, MAT["stone"], 10)
        mb.box((x + (0.75 if i == 0 else -0.75), 84.7, 8.45),
               (1.55, 0.09, 0.09), MAT["gold"])
        direction = 1 if i == 0 else -1
        mb.face(((x, 84.65, 8.40),
                 (x + direction * 1.45, 84.65, 8.40),
                 (x + direction * 1.28, 84.65, 6.15),
                 (x, 84.65, 6.65)), MAT["rose"])

    # Crates, barrels and a weapon rack occupy the edges rather than the
    # maneuvering space at the center of each court.
    for x, y, size in ((-39.0, 86.0, 0.85), (-37.8, 86.4, 0.70),
                       (-39.1, 87.2, 0.62), (31.8, 86.2, 0.82)):
        mb.box((x, y, size / 2 + 0.18), (size, size, size),
               MAT["wood"], rot=0.12 * x)
    for x, y in ((-32.0, 86.2), (-32.0, 87.25), (39.0, 86.5)):
        mb.cylinder((x, y, 0.62), 0.46, 1.12, MAT["wood"], 10)
        mb.cylinder((x, y, 0.62), 0.48, 0.10, MAT["iron"], 10)
    mb.box((-40.2, 92.8, 1.15), (0.22, 3.4, 2.3), MAT["timber"])
    for y in (91.7, 92.8, 93.9):
        mb.beam((-40.0, y, 0.35), (-39.1, y, 2.35), 0.10, 0.12, MAT["iron"])
    mb.object("Castle_InnerCourtyardsAndStores", COLLECTIONS["Castle"])

    # Compact guard post on the east court. Its 6.4m roofline stays far below
    # the adjacent 35m drum tower and therefore does not alter the skyline.
    mb = MeshBuilder()
    gx, gy = 36.0, 92.0
    mb.box((gx, gy, 2.1), (7.0, 5.0, 4.2), MAT["stone"])
    mb.hip_roof(gx, gy, 4.2, 7.7, 5.8, 2.2, MAT["slate"])
    mb.box((gx, gy - 2.54, 1.35), (1.35, 0.18, 2.7), MAT["wood"])
    for x in (gx - 2.25, gx + 2.25):
        mb.box((x, gy - 2.55, 2.15), (0.82, 0.16, 1.15), MAT["glass"])
        mb.box((x, gy - 2.64, 1.48), (1.02, 0.24, 0.16), MAT["stone2"])
    mb.box((gx, gy + 2.55, 1.0), (5.2, 0.18, 1.8), MAT["timber"])
    mb.object("Castle_GuardPost", COLLECTIONS["Castle"])


def make_outskirts():
    """Approach landscape outside the south gate: river with a stone bridge,
    the approach road, fenced crop fields, farmhouses and a windmill."""
    mb = MeshBuilder()
    # Approach road from the gate to the horizon.
    mb.box((0, -128, 0.01), (5.2, 74, 0.1), MAT["road"])
    # River crossing east-west. Water sits just above the grass with low
    # soil banks so the edge reads from first person.
    mb.box((0, -114, -0.02), (230, 9.0, 0.06), MAT["water"])
    for sy in (-5.6, 5.6):
        mb.box((0, -114 + sy, 0.03), (230, 2.2, 0.14), MAT["soil"])
    # Stone bridge: gently arched deck in three segments plus parapets.
    mb.box((0, -114, 0.55), (6.4, 7.2, 0.5), MAT["stone2"])
    for sy in (-6.2, 6.2):
        mb.box((0, -114 + sy, 0.32), (6.4, 5.6, 0.55), MAT["stone2"], rot=0)
    for sx in (-2.9, 2.9):
        mb.box((sx, -114, 1.25), (0.5, 18.0, 1.0), MAT["stone"])
    mb.object("Ground_ApproachRoadRiverBridge", COLLECTIONS["Ground"])

    # Crop fields with furrow rows and fence posts.
    mb = MeshBuilder()
    fr = random.Random(SEED + 131)
    fields = [(-16, -104, 16, 12), (17, -106, 18, 13), (-20, -128, 18, 14),
              (18, -132, 15, 15), (-14, -148, 20, 12)]
    for fx, fy, fw, fd in fields:
        mb.box((fx, fy, 0.03), (fw, fd, 0.14), MAT["soil"])
        crop = MAT["leaf"] if fr.random() < 0.6 else MAT["ochre"]
        rows = int(fd / 1.6)
        for k in range(rows):
            ry = fy - fd / 2 + (k + 0.5) * fd / rows
            mb.box((fx, ry, 0.16), (fw * 0.92, 0.55, 0.18), crop)
        for px in (-fw / 2, fw / 2):
            for py in range(int(-fd / 2), int(fd / 2) + 1, 3):
                mb.box((fx + px, fy + py, 0.55), (0.14, 0.14, 1.1), MAT["timber"])
            mb.box((fx + px, fy, 1.0), (0.09, fd, 0.09), MAT["timber"])
    # Haystacks.
    for hx, hy in ((-27, -110), (28, -122), (-8, -140)):
        mb.cone((hx, hy), 0.1, 1.6, 2.6, MAT["ochre"], 10)
    mb.object("Prop_FarmFields", COLLECTIONS["Props"])

    # Windmill on a low knoll west of the road.
    mb = MeshBuilder()
    wx, wy = -30, -136
    mb.cylinder((wx, wy, 5.5), 3.0, 11.0, MAT["stone2"], 12)
    mb.cone((wx, wy), 11.0, 3.5, 3.4, MAT["green"], 12)
    mb.box((wx, wy - 2.6, 1.5), (1.3, 0.8, 2.6), MAT["wood"])
    hub_y = wy - 3.4
    mb.cylinder((wx, hub_y + 0.3, 9.8), 0.35, 1.0, MAT["timber"], 8, axis="Y")
    for ang in (0.785, 2.356, 3.927, 5.498):
        dx, dz = math.cos(ang), math.sin(ang)
        mb.beam((wx + dx * 0.6, hub_y, 9.8 + dz * 0.6),
                (wx + dx * 6.2, hub_y, 9.8 + dz * 6.2), 0.22, 0.22, MAT["timber"])
        mb.beam((wx + dx * 2.2, hub_y + 0.12, 9.8 + dz * 2.2),
                (wx + dx * 6.0, hub_y + 0.12, 9.8 + dz * 6.0), 0.1, 1.35, MAT["cream2"])
    mb.object("Prop_Windmill", COLLECTIONS["Props"])

    # Farmhouses flanking the road.
    make_house("House_Farm_A", -13, -117.5, 8.5, 7, 1, 9101)
    make_house("House_Farm_B", 14, -145, 9.0, 7.5, -1, 9102)

    make_harbor()


def make_harbor():
    """Small river port SE of the gate: dirt path, quay, piers, boats,
    a warehouse and a loading crane."""
    mb = MeshBuilder()
    # Dirt path from the approach road to the quay.
    mb.box((21, -102.5, 0.015), (38, 3.4, 0.09), MAT["soil"])
    # Stone quay along the north bank.
    mb.box((47, -107.2, 0.35), (30, 4.0, 0.9), MAT["stone2"])
    for bx in range(34, 61, 4):
        mb.cylinder((bx, -105.4, 0.95), 0.16, 0.9, MAT["timber"], 8)
    # Two timber piers reaching into the river.
    for px in (40, 54):
        mb.box((px, -113.5, 0.42), (2.2, 9.5, 0.18), MAT["wood"])
        for py in (-117.6, -113.5, -109.6):
            for sx in (-0.95, 0.95):
                mb.cylinder((px + sx, py, -0.1), 0.14, 1.6, MAT["timber"], 8)
    # Rowboat beside the west pier: flat bottom, flared side planks, thwarts.
    bx, by = 36.6, -114.5
    mb.box((bx, by, 0.06), (1.3, 3.4, 0.16), MAT["wood"])
    for sx in (-1, 1):
        mb.box((bx + sx * 0.75, by, 0.28), (0.14, 3.6, 0.34), MAT["wood"], rot=0)
    for oy in (-1.75, 1.75):
        mb.box((bx, by + oy, 0.28), (1.35, 0.16, 0.34), MAT["wood"], rot=0)
    for oy in (-0.8, 0.5):
        mb.box((bx, by + oy, 0.3), (1.2, 0.3, 0.08), MAT["timber"])
    # Cargo barge with a mast and furled sail at the east pier.
    gx, gy = 57.5, -114.8
    mb.box((gx, gy, 0.22), (3.0, 7.5, 0.5), MAT["wood"])
    for sx in (-1, 1):
        mb.box((gx + sx * 1.55, gy, 0.62), (0.18, 7.8, 0.75), MAT["wood"])
    for oy in (-3.85, 3.85):
        mb.box((gx, gy + oy, 0.62), (3.1, 0.2, 0.75), MAT["wood"])
    mb.cylinder((gx, gy - 1.2, 3.4), 0.16, 6.0, MAT["timber"], 8)
    mb.box((gx, gy - 1.2, 5.6), (3.6, 0.14, 0.14), MAT["timber"])
    mb.box((gx, gy - 1.2, 5.15), (3.2, 0.3, 0.7), MAT["cream2"])
    for k in range(3):
        mb.box((gx - 0.7 + k * 0.7, gy + 1.4, 0.75), (0.62, 0.62, 0.6), MAT["wood"], 0.3 * k)
    # Warehouse on the quay.
    mb.box((46, -101.5, 2.4), (12, 7.5, 4.8), MAT["stone2"])
    mb.gable_roof(46, -101.5, 4.8, 13.4, 8.3, 2.6, MAT["slate"])
    mb.box((41.5, -105.05, 1.5), (2.6, 0.2, 3.0), MAT["wood"])
    mb.box((50.5, -105.05, 1.5), (2.6, 0.2, 3.0), MAT["timber"])
    # Loading crane: post, angled jib, rope and hanging crate.
    mb.cylinder((59, -106.5, 2.6), 0.3, 5.2, MAT["timber"], 8)
    mb.beam((59, -106.5, 4.9), (59, -111.5, 3.4), 0.26, 0.26, MAT["timber"])
    mb.box((59, -111.5, 2.4), (0.05, 0.05, 2.1), MAT["iron"])
    mb.box((59, -111.5, 1.05), (0.75, 0.75, 0.7), MAT["wood"], 0.4)
    # Barrels and crates on the quay.
    hr = random.Random(SEED + 151)
    for _ in range(7):
        ox, oy = hr.uniform(36, 58), hr.uniform(-108.6, -106.2)
        if hr.random() < 0.5:
            mb.cylinder((ox, oy, 1.3), 0.42, 1.0, MAT["wood"], 10)
        else:
            s = hr.uniform(0.55, 0.8)
            mb.box((ox, oy, 0.8 + s / 2), (s, s, s), MAT["wood"], hr.uniform(0, 1.5))
    mb.object("Prop_RiverHarbor", COLLECTIONS["Props"])


def make_landmarks():
    """Secondary landmarks below the castle's dominant 61m keep."""
    mb = MeshBuilder()

    # --- Church: nave along X, bell tower on the street-facing east end.
    cx, cy = -34.0, 64.0
    nave_w, nave_l, wall_h = 10.0, 15.0, 7.0
    # Cream nave with stone trim so the church reads separately from the
    # all-stone castle right behind it.
    mb.box((cx - 1.5, cy, wall_h / 2), (nave_l, nave_w, wall_h), MAT["cream2"])
    mb.gable_roof(cx - 1.5, cy, wall_h, nave_l + 1.4, nave_w + 1.0, 3.6, MAT["slate"])
    # Buttresses and tall windows along both nave sides.
    for i in range(4):
        bx = cx - 7.0 + i * 3.7
        for sy in (-1, 1):
            yb = cy + sy * (nave_w / 2 + 0.35)
            mb.box((bx, yb, 2.6), (0.8, 0.9, 5.2), MAT["stone"])
            mb.box((bx + 1.85, cy + sy * (nave_w / 2 + 0.06), 3.6), (1.1, 0.14, 3.4), MAT["glass"])
            mb.box((bx + 1.85, cy + sy * (nave_w / 2 + 0.10), 3.6), (1.34, 0.10, 3.6), MAT["stone"])
    # Apse (west end).
    mb.cylinder((cx - 9.5, cy, 2.75), 4.2, 5.5, MAT["stone2"], 12)
    mb.cone((cx - 9.5, cy), 5.5, 4.5, 2.6, MAT["slate"], 12)
    # Bell tower (east end): shaft, belfry openings, spire and gold cross.
    tx = cx + 8.6
    mb.box((tx, cy, 12.0), (4.6, 4.6, 24.0), MAT["stone2"])
    for ox, oy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        for off in (-0.62, 0.62):
            mb.box((tx + ox * 2.35 + (0 if ox else off), cy + oy * 2.35 + (0 if oy else off), 21.0),
                   (0.45 if ox else 0.85, 0.45 if oy else 0.85, 2.2), MAT["iron"])
    mb.box((tx, cy, 24.4), (5.4, 5.4, 0.8), MAT["stone"])
    mb.cone((tx, cy), 24.8, 3.6, 6.5, MAT["green"], 8)
    mb.box((tx, cy, 32.2), (0.18, 0.18, 1.9), MAT["gold"])
    mb.box((tx, cy, 32.5), (0.18, 0.95, 0.18), MAT["gold"])
    # Arched entrance porch facing the town.
    mb.box((tx + 2.5, cy, 1.7), (0.7, 2.6, 3.4), MAT["wood"])
    mb.box((tx + 2.62, cy, 3.6), (0.6, 3.2, 0.5), MAT["stone"])
    mb.object("Landmark_Church", COLLECTIONS["Town"])

    # --- Clock tower at the plaza NE corner.
    mb = MeshBuilder()
    qx, qy = 27.5, 46.5
    mb.box((qx, qy, 10.0), (5.0, 5.0, 20.0), MAT["stone2"])
    mb.box((qx, qy, 20.7), (5.8, 5.8, 1.4), MAT["stone"])
    for ox, oy, rot in ((0, -1, 0.0), (-1, 0, math.pi / 2)):
        fx, fy = qx + ox * 2.56, qy + oy * 2.56
        mb.cylinder((fx, fy, 17.5), 1.55, 0.18, MAT["cream2"], 16, axis="X" if ox else "Y")
        mb.cylinder((fx + ox * 0.06, fy + oy * 0.06, 17.5), 0.16, 0.1,
                    MAT["gold"], 8, axis="X" if ox else "Y")
        # Hands: minute up, hour angled.
        mb.box((fx + ox * 0.12, fy + oy * 0.12, 18.05), (0.1, 0.12, 1.1), MAT["iron"])
        mb.box((fx + ox * 0.12, fy + oy * 0.12 + (0.35 if ox else 0), 17.62),
               (0.1, 0.75 if ox else 0.12, 0.12 if ox else 0.75), MAT["iron"])
    mb.hip_roof(qx, qy, 21.4, 5.6, 5.6, 3.2, MAT["green"])
    mb.box((qx, qy, 25.2), (0.16, 0.16, 1.4), MAT["gold"])
    for wx, wy in ((-1.6, -1.6), (1.6, -1.6), (-1.6, 1.6), (1.6, 1.6)):
        mb.box((qx + wx, qy + wy, 21.9), (0.5, 0.5, 1.6), MAT["stone2"])
    for z in (4.0, 9.0, 13.5):
        mb.box((qx, qy - 2.53, z), (1.0, 0.16, 1.8), MAT["glass"])
    mb.box((qx, qy - 2.56, 1.35), (1.4, 0.2, 2.7), MAT["wood"])
    mb.object("Landmark_ClockTower", COLLECTIONS["Town"])

    # --- Wizard tower: squeezed into the east lawn between Plaza_E and the
    # outer road. Its tapered shaft and oversized observatory crown keep the
    # silhouette fanciful without competing with the castle keep.
    mb = MeshBuilder()
    wx, wy = 39.0, 30.5
    shaft_h, base_r, neck_r = 19.4, 2.85, 2.05
    mb.frustum((wx, wy), 0.08, base_r, neck_r, shaft_h, MAT["stone2"], 18)
    # Uneven masonry bands articulate the tall shaft at street-view distance.
    for z, radius in ((0.45, 2.94), (6.4, 2.67), (12.8, 2.38), (19.35, 2.20)):
        mb.cylinder((wx, wy, z), radius, 0.34, MAT["stone"], 18)

    # Spiral windows follow the taper. The thin local X dimension is radial.
    for i, z in enumerate((3.6, 6.2, 8.8, 11.4, 14.0, 16.6)):
        angle = math.radians(25 + i * 67)
        radius = base_r + (neck_r - base_r) * (z / shaft_h) + 0.02
        px, py = wx + math.cos(angle) * radius, wy + math.sin(angle) * radius
        mb.box((px, py, z), (0.18, 0.72, 1.30), MAT["glass"], rot=angle)
        # Small stone sill sits proud of each inset opening.
        mb.box((px + math.cos(angle) * 0.09, py + math.sin(angle) * 0.09, z - 0.74),
               (0.30, 0.96, 0.18), MAT["stone"], rot=angle)

    # West-facing door opens toward the plaza rather than the perimeter road.
    mb.box((wx - base_r - 0.04, wy, 1.45), (0.20, 1.34, 2.75),
           MAT["wood"], rot=math.pi)
    mb.box((wx - base_r - 0.15, wy, 2.95), (0.30, 1.72, 0.30),
           MAT["stone"], rot=math.pi)

    # Corbelled, projecting observatory and conical roof.
    for a in range(0, 360, 30):
        angle = math.radians(a)
        mb.box((wx + 2.55 * math.cos(angle), wy + 2.55 * math.sin(angle), 19.72),
               (1.10, 0.58, 1.05), MAT["stone"], rot=angle)
    mb.cylinder((wx, wy, 20.10), 3.48, 0.72, MAT["stone"], 18)
    mb.cylinder((wx, wy, 22.10), 3.15, 3.85, MAT["ochre"], 18)
    mb.cylinder((wx, wy, 24.10), 3.48, 0.35, MAT["timber"], 18)
    for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        px, py = wx + 3.16 * math.cos(angle), wy + 3.16 * math.sin(angle)
        mb.box((px, py, 22.15), (0.18, 0.92, 1.55), MAT["glass"], rot=angle)
        mb.box((px + math.cos(angle) * 0.08, py + math.sin(angle) * 0.08, 21.27),
               (0.28, 1.12, 0.18), MAT["timber"], rot=angle)
    mb.cone((wx, wy), 24.28, 3.86, 4.65, MAT["green"], 18)

    # Faceted, double-ended emissive crystal on a compact metal setting.
    mb.cylinder((wx, wy, 29.08), 0.55, 0.34, MAT["gold"], 8)
    ring_start = len(mb.vertices)
    crystal_ring_z, crystal_r, crystal_segments = 30.0, 0.48, 6
    for i in range(crystal_segments):
        a = math.tau * i / crystal_segments
        mb.vertices.append((wx + crystal_r * math.cos(a),
                            wy + crystal_r * math.sin(a), crystal_ring_z))
    bottom_i = len(mb.vertices)
    mb.vertices.append((wx, wy, 29.20))
    top_i = len(mb.vertices)
    mb.vertices.append((wx, wy, 31.55))
    for i in range(crystal_segments):
        j = (i + 1) % crystal_segments
        mb.faces.extend(((ring_start + j, ring_start + i, bottom_i),
                         (ring_start + i, ring_start + j, top_i)))
        mb.mat_ids.extend((MAT_INDEX[MAT["crystal"].name],) * 2)

    # A few domestic details keep the landmark grounded rather than isolated.
    for py in (wy - 0.95, wy + 0.95):
        mb.cylinder((wx - 3.22, py, 0.32), 0.34, 0.54, MAT["tile"], 8)
        mb.sphere((wx - 3.22, py, 0.78), 0.42, MAT["leaf"], 8, 4)
    for row in range(2):
        for col in range(3):
            mb.cylinder((wx - 0.52 + col * 0.52, wy - 3.03, 0.32 + row * 0.43),
                        0.21, 1.20, MAT["wood"], 8, axis="Y")
    mb.object("Landmark_WizardTower", COLLECTIONS["Town"])
    add_point_light(
        "Prop_WizardCrystalGlow", (wx, wy, 30.25),
        energy=520.0, color=(0.04, 0.55, 1.0), radius=1.8,
    )


def make_walls():
    mb = MeshBuilder()
    # A tighter ring keeps the interior urban rather than park-like.
    mb.box((-78, 3, 5.5), (4, 190, 11), MAT["stone"])
    mb.box((78, 3, 5.5), (4, 190, 11), MAT["stone"])
    # The castle replaces the central north-wall span. Leaving a wall slab
    # behind it would silently plug the otherwise open gate tunnel.
    for side in (-1, 1):
        mb.box((side * 55, 98, 5.5), (46, 4, 11), MAT["stone"])
    mb.box((-45, -92, 5.5), (66, 4, 11), MAT["stone"])
    mb.box((45, -92, 5.5), (66, 4, 11), MAT["stone"])
    # Crenellations.
    for x in range(-75, 76, 5):
        if abs(x) >= 33:
            mb.box((x, 98, 11.9), (2.7, 4.5, 2.0), MAT["stone2"])
        if abs(x) > 13:
            mb.box((x, -92, 11.9), (2.7, 4.5, 2.0), MAT["stone2"])
    for y in range(-89, 96, 5):
        mb.box((-78, y, 11.9), (4.5, 2.7, 2.0), MAT["stone2"])
        mb.box((78, y, 11.9), (4.5, 2.7, 2.0), MAT["stone2"])
    # South gate towers and arch-like lintel.
    for x in (-11, 11):
        mb.cylinder((x, -92, 9.5), 6, 19, MAT["stone2"], 12)
        mb.cylinder((x, -92, 19.3), 6.8, 1.2, MAT["stone"], 12)
    mb.box((0, -92, 15), (16, 5, 7), MAT["stone2"])
    # Half-raised portcullis: a bar grid in the upper part of the opening so
    # the main street stays visible through the gate from the approach road.
    for bx in range(-4, 5, 1):
        mb.box((bx, -89.4, 10.4), (0.16, 0.16, 4.4), MAT["iron"])
    for bz in (8.6, 10.2, 11.8):
        mb.box((0, -89.4, bz), (9.2, 0.14, 0.14), MAT["iron"])
    mb.object("Wall_OuterRingAndSouthGate", COLLECTIONS["Walls"])


def make_market_and_props():
    mb = MeshBuilder()
    # Legible stepped fountain: plinth, lower basin/water, pedestal, upper bowl.
    mb.cylinder((0, 25, 0.18), 4.8, 0.36, MAT["stone"], 20)
    mb.cylinder((0, 25, 0.48), 4.25, 0.42, MAT["stone2"], 20)
    mb.cylinder((0, 25, 0.73), 3.72, 0.10, MAT["water"], 20)
    mb.cylinder((0, 25, 2.05), 0.72, 2.7, MAT["stone"], 14)
    mb.cylinder((0, 25, 3.08), 2.05, 0.34, MAT["stone2"], 18)
    mb.cylinder((0, 25, 3.28), 1.68, 0.08, MAT["water"], 18)
    mb.cylinder((0, 25, 4.15), 0.36, 1.75, MAT["stone2"], 12)
    mb.cylinder((0, 25, 5.12), 0.82, 0.22, MAT["stone"], 14)
    for a in range(0, 360, 90):
        rad = math.radians(a)
        mb.box((1.35 * math.cos(rad), 25 + 1.35 * math.sin(rad), 2.55),
               (0.09, 0.09, 1.25), MAT["water"], rot=rad)
    # Market stalls.
    stall_positions = [(-19, 17, 0), (-10, 34, math.pi), (13, 16, 0), (21, 33, math.pi)]
    for i, (x, y, rot) in enumerate(stall_positions):
        canopy = (MAT["tile"], MAT["green"], MAT["cream2"], MAT["rose"])[i]
        mb.box((x, y, 1.05), (5.2, 2.6, 0.42), MAT["wood"], rot)
        mb.box((x, y, 3.2), (5.8, 3.2, 0.22), canopy, rot)
        for dx in (-2.35, 2.35):
            for dy in (-1.1, 1.1):
                mb.box((x + dx, y + dy, 1.75), (0.16, 0.16, 3.4), MAT["timber"])
        for j in range(4):
            mb.box((x - 1.8 + j * 1.2, y, 1.48), (0.65, 0.7, 0.35),
                   MAT["ochre"] if j % 2 else MAT["green"])
        if DUSK_MODE:
            mb.box((x, y, 2.72), (0.30, 0.30, 0.42), MAT["light"])
    # Crates, barrels and carts.
    for i in range(20 if not TEST_MODE else 7):
        x = RNG.choice((-1, 1)) * RNG.uniform(8, 28)
        y = RNG.uniform(12, 38)
        if i % 3:
            mb.box((x, y, 0.48), (0.85, 0.85, 0.92), MAT["wood"], RNG.uniform(0, math.pi))
        else:
            mb.cylinder((x, y, 0.58), 0.46, 1.12, MAT["wood"], 10)
            mb.cylinder((x, y, 0.58), 0.48, 0.10, MAT["iron"], 10)
    # Lanterns along the street.
    street_lamp_x = 5.65
    for y in range(-60, 73, 12):
        for side in (-street_lamp_x, street_lamp_x):
            x = street_x(y) + side
            mb.cylinder((x, y, 2.1), 0.09, 4.2, MAT["iron"], 8)
            mb.box((x, y, 4.35), (0.48, 0.48, 0.68), MAT["light"])
            mb.box((x, y, 4.73), (0.68, 0.68, 0.10), MAT["iron"])

    # Street-life clusters hugging the facades: barrels, crates, benches,
    # planters. Placed at wall edges and corners, never mid-street.
    cr = random.Random(SEED + 91)
    for cy_ in range(-70, 71, 9):
        if 5 < cy_ < 44:
            continue  # plaza block manages its own props
        if cr.random() < 0.5:
            continue
        side = cr.choice((-1, 1))
        # Facades sit at ~±5.9m; clusters hug the wall on the street side.
        bx = street_x(cy_) + side * cr.uniform(5.15, 5.5)
        for _ in range(cr.randint(2, 4)):
            ox = bx + cr.uniform(-0.4, 0.2) * side
            oy = cy_ + cr.uniform(-1.4, 1.4)
            roll = cr.random()
            if roll < 0.45:
                mb.cylinder((ox, oy, 0.52), 0.42, 1.0, MAT["wood"], 10)
                mb.cylinder((ox, oy, 0.52), 0.44, 0.09, MAT["iron"], 10)
            elif roll < 0.8:
                s = cr.uniform(0.55, 0.85)
                mb.box((ox, oy, s / 2 + 0.1), (s, s, s), MAT["wood"], cr.uniform(0, 1.5))
            else:
                mb.box((ox, oy, 0.5), (0.45, 1.35, 0.09), MAT["wood"])
                for ly in (-0.55, 0.55):
                    mb.box((ox, oy + ly, 0.25), (0.4, 0.1, 0.42), MAT["timber"])
    # Planters and a handcart give the plaza human-scale accents.
    for px, py in ((-26.5, 12.5), (27, 37.5), (-27, 33)):
        mb.box((px, py, 0.4), (1.0, 1.0, 0.72), MAT["wood"])
        mb.sphere((px, py, 1.15), 0.58, MAT["leaf"], 8, 5)
    # Handcart: bed long in Y, axle along X, wheels either side, twin handles.
    cart_x, cart_y = 14.5, 27.5
    mb.box((cart_x, cart_y, 0.78), (1.35, 2.5, 0.14), MAT["wood"])
    for side_ in (-0.55, 0.55):
        mb.box((cart_x + side_, cart_y, 1.02), (0.10, 2.5, 0.34), MAT["wood"])
    for wx in (-0.82, 0.82):
        mb.cylinder((cart_x + wx, cart_y, 0.52), 0.5, 0.12, MAT["timber"], 12, axis="X")
    mb.cylinder((cart_x, cart_y, 0.52), 0.06, 1.8, MAT["iron"], 8, axis="X")
    for hx in (-0.45, 0.45):
        mb.box((cart_x + hx, cart_y - 1.85, 0.72), (0.08, 1.2, 0.08), MAT["timber"])
    mb.object("Prop_MarketFountainStallsStreetFurniture", COLLECTIONS["Props"])
    if DUSK_MODE:
        for y in range(-60, 73, 12):
            for x in (-street_lamp_x, street_lamp_x):
                add_point_light(
                    f"Prop_StreetLantern_{'E' if x > 0 else 'W'}_{y:+04d}",
                    (x, y, 3.88), energy=430.0, radius=1.15,
                )
        for i, (x, y, _rot) in enumerate(stall_positions):
            add_point_light(
                f"Prop_MarketLamp_{i:02d}", (x, y, 2.34),
                energy=280.0, radius=0.85,
            )


def make_green_space_props():
    """Low-rise food and work areas filling the otherwise uniform forecourt."""
    # Walled kitchen garden in the NW castle forecourt, beyond the final
    # building row and clear of the west perimeter road.
    mb = MeshBuilder()
    gx, gy = -56.0, 81.0
    mb.box((gx, gy, 0.10), (15.5, 8.8, 0.20), MAT["soil"])
    crop_rng = random.Random(SEED + 1801)
    for row in range(6):
        x = gx - 6.25 + row * 2.5
        mb.box((x, gy, 0.24), (1.15, 7.35, 0.28), MAT["soil"])
        for plant in range(6):
            y = gy - 2.95 + plant * 1.18
            height = crop_rng.uniform(0.42, 0.68)
            mb.cylinder((x, y, 0.30 + height / 2), 0.07, height,
                        MAT["wood"], 6)
            mb.sphere((x, y, 0.52 + height), crop_rng.uniform(0.20, 0.29),
                      MAT["leaf"], 7, 4)

    # Fence has an east-side gate facing the open forecourt.
    for x in range(-64, -47, 3):
        for y in (76.35, 85.65):
            mb.box((x, y, 0.72), (0.16, 0.16, 1.42), MAT["timber"])
    for y in (77.0, 79.5, 82.5, 85.0):
        mb.box((-63.85, y, 0.72), (0.16, 0.16, 1.42), MAT["timber"])
        if not 80.0 < y < 82.0:
            mb.box((-48.15, y, 0.72), (0.16, 0.16, 1.42), MAT["timber"])
    for y in (76.35, 85.65):
        for z in (0.48, 0.98):
            mb.box((gx, y, z), (15.8, 0.12, 0.12), MAT["wood"])
    for x in (-63.85, -48.15):
        for z in (0.48, 0.98):
            # Leave a 2m opening on the garden's east edge.
            if x > -50:
                for yy in (78.4, 83.8):
                    mb.box((x, yy, z), (0.12, 3.0, 0.12), MAT["wood"])
            else:
                mb.box((x, gy, z), (0.12, 8.7, 0.12), MAT["wood"])
    mb.object("Ground_CastleForecourtKitchenGarden", COLLECTIONS["Ground"])

    # East forecourt orchard, work shed, firewood and material stacks.
    mb = MeshBuilder()
    orchard_rng = random.Random(SEED + 1802)
    orchard = ((44.0, 76.0), (48.0, 85.0), (58.0, 85.5),
               (67.0, 76.5), (68.0, 86.0))
    for i, (x, y) in enumerate(orchard):
        trunk_h = orchard_rng.uniform(2.5, 3.2)
        mb.cylinder((x, y, trunk_h / 2), orchard_rng.uniform(0.28, 0.38),
                    trunk_h, MAT["wood"], 8)
        mb.sphere((x, y, trunk_h + 1.25), orchard_rng.uniform(1.55, 1.85),
                  MAT["leaf"], 9, 5)
        mb.sphere((x - 0.75, y + 0.35, trunk_h + 0.75), 1.05,
                  MAT["leaf"], 8, 4)
        for fruit in range(3):
            a = math.tau * (fruit / 3.0) + i * 0.37
            mb.sphere((x + 1.25 * math.cos(a), y + 1.25 * math.sin(a),
                       trunk_h + 1.10 + 0.28 * fruit),
                      0.14, MAT["ochre"], 6, 3)

    sx, sy = 57.0, 73.0
    mb.box((sx, sy, 1.65), (7.0, 5.0, 3.3), MAT["ochre"])
    mb.gable_roof(sx, sy, 3.3, 7.8, 5.8, 2.2, MAT["tile"])
    for x in (sx - 3.15, sx + 3.15):
        mb.box((x, sy - 2.54, 1.65), (0.30, 0.20, 3.3), MAT["timber"])
    mb.box((sx, sy - 2.55, 1.25), (1.35, 0.18, 2.5), MAT["wood"])
    mb.box((sx - 2.1, sy - 2.55, 1.75), (1.05, 0.16, 1.05), MAT["glass"])
    # Logs and covered material stacks at the shed's west side.
    for row in range(3):
        for col in range(4):
            mb.cylinder((51.2 + col * 0.48, 73.0, 0.28 + row * 0.45),
                        0.20, 1.65, MAT["wood"], 8, axis="Y")
    for x, y, size in ((64.0, 71.5, 1.0), (65.1, 72.0, 0.8), (64.4, 73.0, 0.9)):
        mb.box((x, y, size / 2 + 0.06), (size, size, size), MAT["wood"], 0.18)
    # Short boundary fence loosely ties shed and orchard into one work yard.
    for x in range(42, 71, 4):
        mb.box((x, 89.0, 0.72), (0.16, 0.16, 1.42), MAT["timber"])
    for z in (0.48, 0.98):
        mb.box((56.0, 89.0, z), (28.0, 0.12, 0.12), MAT["wood"])
    mb.object("Prop_CastleForecourtOrchardAndYard", COLLECTIONS["Props"])


def make_trees_and_mountains():
    mb = MeshBuilder()
    positions = [(-48, 22), (48, 22), (-47, 64), (48, 64), (-78, -48), (74, -35)]
    tr = random.Random(SEED + 77)
    for i, (x, y) in enumerate(positions):
        mb.cylinder((x, y, 2.8), 0.58, 5.6, MAT["wood"], 9)
        # Overlapping spheres read as a foliage crown instead of stacked drums.
        for ox, oy, oz, rad in ((0, 0, 6.6, 2.6), (-1.7, 0.5, 5.6, 1.9),
                                (1.5, -0.6, 5.8, 2.0), (0.3, 1.4, 5.5, 1.7),
                                (-0.4, -1.5, 5.9, 1.7)):
            mb.sphere((x + ox + tr.uniform(-0.3, 0.3), y + oy + tr.uniform(-0.3, 0.3), oz),
                      rad * tr.uniform(0.9, 1.1), MAT["leaf"], segments=10, rings=6)
    mb.object("Environment_Trees", COLLECTIONS["Environment"])

    mountains = MeshBuilder()
    mr = random.Random(SEED + 501)
    # Interior mist ring masks the physically dark below-horizon half of the
    # analytic sky while retaining its bright vertical gradient above.
    ring_start = len(mountains.vertices)
    ring_segments = 32
    # Top must stay above every camera position, otherwise sight lines clear
    # the ring and hit the pitch-black below-horizon half of the Nishita sky.
    for z in (-12.0, 170.0):
        for j in range(ring_segments):
            aa = math.tau * j / ring_segments
            mountains.vertices.append((340 * math.cos(aa), 340 * math.sin(aa), z))
    for j in range(ring_segments):
        nj = (j + 1) % ring_segments
        mountains.faces.append((ring_start + j, ring_start + ring_segments + j,
                                ring_start + ring_segments + nj, ring_start + nj))
        mountains.mat_ids.append(MAT_INDEX[MAT["horizon"].name])
    # Continuous jagged ridge rings: a strip whose crest height varies smoothly
    # with sharp peaks. Reads as a mountain RANGE, not scattered pyramids.
    def ridge_ring(radius, base_z, min_h, max_h, mat, seed, segments=120, snow=None):
        rr = random.Random(seed)
        phases = [(rr.uniform(0, math.tau), rr.uniform(0, math.tau)) for _ in range(3)]
        start = len(mountains.vertices)
        crest = []
        for j in range(segments):
            a = math.tau * j / segments
            h = 0.0
            for k, (p1, p2) in enumerate(phases):
                f = 3 + k * 4
                h += (math.sin(a * f + p1) * 0.5 + math.sin(a * (f + 1) + p2) * 0.35) / (k + 1)
            h = min_h + (max_h - min_h) * (0.5 + h * 0.30)
            crest.append(max(min_h * 0.6, h))
        for j in range(segments):
            a = math.tau * j / segments
            rj = radius + rr.uniform(-6, 6)
            mountains.vertices.append((rj * math.cos(a), rj * math.sin(a), base_z))
        for j in range(segments):
            a = math.tau * j / segments
            rj = radius + rr.uniform(-4, 4)
            mountains.vertices.append((rj * math.cos(a), rj * math.sin(a), base_z + crest[j]))
        for j in range(segments):
            nj = (j + 1) % segments
            mountains.faces.append((start + j, start + nj,
                                    start + segments + nj, start + segments + j))
            mountains.mat_ids.append(MAT_INDEX[mat.name])
        if snow is not None:
            # Snowcap strip: hangs below each crest vertex only where the
            # ridge is tall enough, so caps appear on peaks, not saddles.
            snow_start = len(mountains.vertices)
            threshold = min_h + (max_h - min_h) * 0.55
            for j in range(segments):
                x, y, z = mountains.vertices[start + segments + j]
                mountains.vertices.append((x, y, z))
            for j in range(segments):
                x, y, z = mountains.vertices[start + segments + j]
                drop = max(4.5, (crest[j] - threshold) * 0.75)
                mountains.vertices.append((x * 0.997, y * 0.997, z - drop))
            for j in range(segments):
                nj = (j + 1) % segments
                if crest[j] > threshold or crest[nj] > threshold:
                    mountains.faces.append((snow_start + segments + j, snow_start + segments + nj,
                                            snow_start + nj, snow_start + j))
                    mountains.mat_ids.append(MAT_INDEX[snow.name])
    ridge_ring(318, -8, 30, 78, MAT["horizon"], SEED + 502, snow=MAT["snow"])
    ridge_ring(288, -8, 18, 46, MAT["mountain"], SEED + 503)
    mountains.object("Environment_DistantMountains", COLLECTIONS["Environment"])


def apply_snow_cover():
    """Replace only upward-facing exterior polygons with packed snow."""
    if not SNOW_MODE:
        return
    snow_mat = MAT["snow_surface"]
    path_snow_mat = MAT["snow_path"]
    path_materials = {MAT["road"].name, MAT["road2"].name}
    roof_materials = {MAT["tile"].name, MAT["slate"].name, MAT["green"].name}
    excluded_materials = {
        MAT["glass"].name,
        MAT["glass_lit"].name,
        MAT["glass_lit_soft"].name,
        MAT["water"].name,
        MAT["light"].name,
        MAT["crystal"].name,
        MAT["gold"].name,
    }
    covered_faces = 0
    covered_meshes = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name == "Environment_DistantMountains":
            continue
        mesh = obj.data
        snow_slots = {
            snow_mat.name: mesh.materials.find(snow_mat.name),
            path_snow_mat.name: mesh.materials.find(path_snow_mat.name),
        }
        mesh_faces = 0
        for poly in mesh.polygons:
            original = mesh.materials[poly.material_index]
            upward = (
                poly.normal.z > 0.38
                or (original is not None
                    and original.name in roof_materials
                    and abs(poly.normal.z) > 0.38)
            )
            if (upward
                    and original is not None
                    and original.name not in excluded_materials):
                target_mat = path_snow_mat if original.name in path_materials else snow_mat
                target_slot = snow_slots[target_mat.name]
                if target_slot < 0:
                    target_slot = len(mesh.materials)
                    mesh.materials.append(target_mat)
                    snow_slots[target_mat.name] = target_slot
                poly.material_index = target_slot
                mesh_faces += 1
        if mesh_faces:
            covered_meshes += 1
            covered_faces += mesh_faces
    log(f"Snow cover: meshes={covered_meshes}, upward_faces={covered_faces}")


def setup_scene():
    scene = bpy.context.scene
    engine_items = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engine_items else "BLENDER_EEVEE"
    )
    scene.render.resolution_x = 640 if TEST_MODE else 1920
    scene.render.resolution_y = 360 if TEST_MODE else 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.20 if SNOW_MODE else (1.15 if DUSK_MODE else 0.0)
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg = nodes.get("Background")
    sky = nodes.new("ShaderNodeTexSky")
    sky_items = sky.bl_rna.properties["sky_type"].enum_items.keys()
    sky.sky_type = "NISHITA" if "NISHITA" in sky_items else "SINGLE_SCATTERING"
    if SNOW_MODE:
        sky.sun_elevation = math.radians(7.0)
        sky.sun_rotation = math.radians(225)
        sky.altitude = 0.08
        sky.air_density = 1.65
    else:
        sky.sun_elevation = math.radians(-2.0 if DUSK_MODE else 18)
        sky.sun_rotation = math.radians(248 if DUSK_MODE else 215)
        sky.altitude = 0.12 if DUSK_MODE else 0.18
        sky.air_density = 1.45 if DUSK_MODE else 1.25
    if hasattr(sky, "dust_density"):
        sky.dust_density = 1.2 if SNOW_MODE else (3.8 if DUSK_MODE else 2.2)
    if hasattr(sky, "ground_albedo"):
        sky.ground_albedo = 0.82 if SNOW_MODE else (0.18 if DUSK_MODE else 0.45)
    bg.inputs["Strength"].default_value = 0.48 if SNOW_MODE else (0.26 if DUSK_MODE else 0.38)
    links.new(sky.outputs["Color"], bg.inputs["Color"])

    sun_name = (
        "Environment_WinterSun" if SNOW_MODE
        else ("Environment_Moonlight" if DUSK_MODE else "Environment_Sun")
    )
    sun_data = bpy.data.lights.new(sun_name, "SUN")
    if SNOW_MODE:
        sun_data.energy = 2.15
        sun_data.color = (0.66, 0.78, 1.0)
        sun_data.angle = math.radians(12)
    else:
        sun_data.energy = 1.30 if DUSK_MODE else 3.0
        sun_data.color = (0.28, 0.42, 1.0) if DUSK_MODE else (1.0, 0.70, 0.45)
        sun_data.angle = math.radians(11 if DUSK_MODE else 7)
    sun = bpy.data.objects.new(sun_name, sun_data)
    COLLECTIONS["Environment"].objects.link(sun)
    sun.rotation_euler = (math.radians(43), math.radians(-22), math.radians(-32))

    cam_data = bpy.data.cameras.new("Environment_RenderCamera")
    cam = bpy.data.objects.new("Environment_RenderCamera", cam_data)
    COLLECTIONS["Environment"].objects.link(cam)
    scene.camera = cam
    cam_data.lens = 35
    cam_data.sensor_width = 36
    return cam


def point_camera(cam, location, target, lens):
    cam.location = location
    cam.data.lens = lens
    direction = Vector(target) - Vector(location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_views(cam):
    # First-person cameras stand on the curved street and aim along it.
    ms_loc = (street_x(-72), -72, 1.6)
    ms_tgt = (street_x(-10) * 0.5, 30, 11)
    alley_loc = (street_x(-42) + 26.9, -42, 1.6)
    alley_tgt = (street_x(-4) + 26.9, -2, 4.2)
    gate_loc = (street_x(34) + 1.2, 34, 1.6)
    if SNOW_MODE:
        views = {
            "snow_overview_quarter": ((94, -104, 62), (0, 15, 10), 52),
            "snow_main_street": (ms_loc, (ms_tgt[0], 55, 10), 32),
            "snow_plaza": ((-14, 12, 1.6), (3, 27, 2.6), 34),
        }
    elif DUSK_MODE:
        views = {
            "dusk_main_street": (ms_loc, (ms_tgt[0], 65, 12), 32),
            "dusk_plaza": ((-14, 12, 1.6), (3, 27, 2.6), 34),
            "dusk_overview_quarter": ((94, -104, 62), (0, 15, 10), 52),
        }
    else:
        views = {
            "overview": ((122, -140, 112), (0, 18, 13), 47),
            "overview_quarter": ((94, -104, 62), (0, 15, 10), 52),
            "main_street_fp": (ms_loc, ms_tgt, 32),
            "plaza_fp": ((-14, 12, 1.6), (3, 27, 2.6), 34),
            "castle_gate_fp": (gate_loc, (0.5, 89, 9.5), 32),
            "alley_fp": (alley_loc, alley_tgt, 35),
            "church_fp": ((-56, 42, 1.6), (-25, 64, 13), 27),
            "wizard_fp": ((36.5, 13, 1.6), (40.5, 30, 15), 30),
            "approach_fp": ((-7, -161, 2.2), (1, -88, 15), 30),
            "harbor_fp": ((28, -117, 1.8), (52, -107, 3.5), 30),
            "tavern_fp": ((2, 22, 1.8), (29.0, 15.0, 6.2), 36),
        }
    if TEST_MODE and not SNOW_MODE:
        if DUSK_MODE:
            views = {
                "dusk_main_street": views["dusk_main_street"],
                "dusk_plaza": views["dusk_plaza"],
            }
        else:
            views = {
                "main_street_fp": views["main_street_fp"],
                "castle_gate_fp": views["castle_gate_fp"],
                "overview": views["overview"],
                "tavern_fp": views["tavern_fp"],
            }
    for name, (loc, target, lens) in views.items():
        log(f"Rendering {name}")
        point_camera(cam, loc, target, lens)
        bpy.context.scene.render.filepath = str(RENDER_DIR / f"{name}.png")
        bpy.ops.render.render(write_still=True)


def render_turntable(cam):
    """Render a deterministic one-orbit H.264 overview animation."""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 24 if TEST_MODE else 216
    scene.render.fps = 24
    scene.render.fps_base = 1.0
    scene.render.resolution_x = 640 if TEST_MODE else 1280
    scene.render.resolution_y = 360 if TEST_MODE else 720
    scene.render.resolution_percentage = 100
    # Blender 5.1 filters file_format by media_type; VIDEO must be selected
    # before FFMPEG becomes a valid enum item.
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(RENDER_DIR / "turntable.mp4")

    center = Vector((0.0, 10.0, 14.0))
    radius = 180.0
    height = 92.0
    frame_count = scene.frame_end - scene.frame_start + 1
    for frame in range(scene.frame_start, scene.frame_end + 1):
        angle = math.tau * (frame - scene.frame_start) / frame_count
        cam.location = (radius * math.cos(angle),
                        10.0 + radius * math.sin(angle), height)
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = 48
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Every rendered integer frame has an explicit transform key, avoiding
    # dependence on Blender 5.1's layered Action/F-Curve interpolation API.
    scene.frame_set(scene.frame_start)
    log(
        f"Rendering turntable: {frame_count} frames, {scene.render.fps}fps, "
        f"{scene.render.resolution_x}x{scene.render.resolution_y}"
    )
    bpy.ops.render.render(animation=True)


def render_walkthrough(cam):
    """Eye-height dolly along the approach road and the curved main street:
    bridge -> south gate -> main street -> plaza -> castle gate."""
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 24 if TEST_MODE else 600
    scene.render.fps = 24
    scene.render.fps_base = 1.0
    scene.render.resolution_x = 640 if TEST_MODE else 1280
    scene.render.resolution_y = 360 if TEST_MODE else 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(RENDER_DIR / "walkthrough.mp4")

    def path_x(y):
        # The approach road south of the gate is straight on x=0; the street
        # curve takes over exactly at the gate (street_x(-92) == 0).
        return street_x(y) if y > STREET_Y0 else 0.0

    y_start, y_end = -122.0, 78.0
    frame_count = scene.frame_end - scene.frame_start + 1
    for frame in range(scene.frame_start, scene.frame_end + 1):
        t = (frame - scene.frame_start) / (frame_count - 1)
        y = y_start + (y_end - y_start) * t
        cam.location = (path_x(y), y, 2.5 if y < -108 else 1.75)
        look_y = min(y + 20.0, 86.0)
        target = Vector((path_x(look_y), look_y, 4.5))
        direction = target - Vector(cam.location)
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = 30
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    scene.frame_set(scene.frame_start)
    log(f"Rendering walkthrough: {frame_count} frames")
    bpy.ops.render.render(animation=True)


def export_scene():
    log("Exporting GLB")
    bpy.ops.export_scene.gltf(
        filepath=str(EXPORT_DIR / "town.glb"),
        export_format="GLB",
        export_apply=True,
        export_cameras=False,
        export_lights=False,
    )
    log("Exporting FBX")
    bpy.ops.export_scene.fbx(
        filepath=str(EXPORT_DIR / "town.fbx"),
        use_selection=False,
        object_types={"MESH"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        bake_space_transform=False,
        add_leaf_bones=False,
        use_mesh_modifiers=True,
        path_mode="COPY",
        embed_textures=False,
    )
    for texture_path in TEXTURES.values():
        shutil.copy2(texture_path, EXPORT_DIR / Path(texture_path).name)


def validate_and_report():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    polys = sum(len(o.data.polygons) for o in meshes)
    tris = 0
    for obj in meshes:
        obj.data.calc_loop_triangles()
        tris += len(obj.data.loop_triangles)
    log(f"Validation: objects={len(bpy.context.scene.objects)}, meshes={len(meshes)}, "
        f"polygons={polys}, triangles={tris}")
    log(f"Collections: {', '.join(c.name for c in bpy.data.collections)}")
    log(f"Elapsed: {time.time() - START_TIME:.1f}s")


def main():
    log(
        f"Starting seed={SEED}, test_mode={TEST_MODE}, dusk_mode={DUSK_MODE}, "
        f"turntable_mode={TURNTABLE_MODE}, walkthrough_mode={WALKTHROUGH_MODE}, "
        f"snow_mode={SNOW_MODE}, "
        f"Blender={bpy.app.version_string}"
    )
    build_streets_and_houses()
    make_tavern()
    add_cobbles()
    make_castle()
    make_castle_inner_details()
    make_landmarks()
    make_outskirts()
    make_walls()
    make_market_and_props()
    make_green_space_props()
    make_trees_and_mountains()
    apply_snow_cover()
    cam = setup_scene()
    validate_and_report()
    if SNOW_MODE:
        render_views(cam)
    elif TURNTABLE_MODE:
        render_turntable(cam)
    elif WALKTHROUGH_MODE:
        render_walkthrough(cam)
    else:
        render_views(cam)
    if (not SNOW_MODE and not TURNTABLE_MODE and not WALKTHROUGH_MODE and not DUSK_MODE
            and (not TEST_MODE or os.environ.get("TOWN_TEST_EXPORT", "0") == "1")):
        export_scene()
    if SNOW_MODE:
        blend_name = "town_snow_test.blend" if TEST_MODE else "town_snow.blend"
    elif TURNTABLE_MODE:
        blend_name = "town_turntable_test.blend" if TEST_MODE else "town_turntable.blend"
    elif DUSK_MODE:
        blend_name = "town_dusk_test.blend" if TEST_MODE else "town_dusk.blend"
    else:
        blend_name = "town_test.blend" if TEST_MODE else "town.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(EXPORT_DIR / blend_name))
    validate_and_report()
    log("SUCCESS")


if __name__ == "__main__":
    main()
