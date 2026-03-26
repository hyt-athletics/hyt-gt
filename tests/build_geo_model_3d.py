"""精细金属矿床三维地质模型构建脚本。

构建 1000m x 1000m x 1000m 区域的合成地质模型，适用于金属矿深部勘探：
  - 起伏地形覆盖层（风化壳，30~80m 厚，非均质电阻率）
  - 风化基岩过渡带（20~30m 厚，ρ=800 Ohm·m）
  - 新鲜基岩层（花岗岩基底）
  - 板状/脉状金属矿体（带蚀变晕渐变带）
  - 三口钻孔轨迹（两口垂直孔 + 一口 80 度斜孔）

精细网格：dx=dy=5m, dz=2.5m，矿体内部 3~6 个格子分辨率。

运行方式：
    .venv/bin/python tests/build_geo_model_3d.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from core.data_store import DataStore
from core.mesh_io import MeshWithModel, save_mesh

# ── 配置 ──────────────────────────────────────────────────────────────
WORKSPACE = Path.home() / "geophys-workspace"
BH_NAME = "GEO-MODEL-3D"
OUT_DIR = _ROOT / "tests" / "data" / "geo_model_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN = {"x": 1000.0, "y": 1000.0, "z": 1000.0}
DX, DY, DZ = 5.0, 5.0, 2.5

PROPS = {
    "air":        {"resistivity": 1e14,   "conductivity": 1e-14,  "chargeability": 0.0},
    "overburden": {"resistivity": 150.0,  "conductivity": 1/150,  "chargeability": 0.02},
    "weathered":  {"resistivity": 800.0,  "conductivity": 1/800,  "chargeability": 0.015},
    "bedrock":    {"resistivity": 5000.0, "conductivity": 1/5000, "chargeability": 0.01},
    "ore_plate":  {"resistivity": 5.0,    "conductivity": 1/5,    "chargeability": 0.15},
    "ore_vein":   {"resistivity": 2.0,    "conductivity": 1/2,    "chargeability": 0.25},
}

print("=" * 70)
print("精细金属矿床三维地质模型构建")
print(f"域: {DOMAIN['x']:.0f}x{DOMAIN['y']:.0f}x{DOMAIN['z']:.0f} m")
print(f"网格: dx={DX}, dy={DY}, dz={DZ} m")
print("=" * 70)

# ── 1. 构建张量网格 ───────────────────────────────────────────────────
from discretize import TensorMesh

nx = int(DOMAIN["x"] / DX)
ny = int(DOMAIN["y"] / DY)
nz = int(DOMAIN["z"] / DZ)

n_pad = 5
pad_factor = 1.3

hx_core = np.ones(nx) * DX
hy_core = np.ones(ny) * DY
hz_core = np.ones(nz) * DZ

hx_pad = DX * pad_factor ** np.arange(1, n_pad + 1)
hy_pad = DY * pad_factor ** np.arange(1, n_pad + 1)
hz_pad = DZ * pad_factor ** np.arange(1, n_pad + 1)

hx = np.concatenate([hx_pad[::-1], hx_core, hx_pad])
hy = np.concatenate([hy_pad[::-1], hy_core, hy_pad])
hz = np.concatenate([hz_core, hz_pad])

# origin: core domain at [0,1000] in x/y, top of mesh at z=0
origin = [-hx_pad.sum(), -hy_pad.sum(), -hz.sum()]
mesh = TensorMesh([hx, hy, hz], origin=origin)
print(f"\n[1] 网格构建完成: {mesh.nC:,} cells, shape {mesh.shape_cells}")
print(f"    x: [{mesh.nodes_x.min():.0f}, {mesh.nodes_x.max():.0f}] m")
print(f"    y: [{mesh.nodes_y.min():.0f}, {mesh.nodes_y.max():.0f}] m")
print(f"    z: [{mesh.nodes_z.min():.0f}, {mesh.nodes_z.max():.0f}] m")

# ── 2. 地形面 + 地层界面 ──────────────────────────────────────────────
cc = mesh.cell_centers
x_cc, y_cc, z_cc = cc[:, 0], cc[:, 1], cc[:, 2]

topo_z = (
    25.0 * np.sin(2 * np.pi * x_cc / 800.0) * np.cos(2 * np.pi * y_cc / 600.0)
    + 10.0 * np.sin(2 * np.pi * x_cc / 300.0 + 0.5)
)

# 覆盖层底界（西薄东厚）
overburden_thickness = 30.0 + 50.0 * (x_cc - x_cc.min()) / (x_cc.max() - x_cc.min())
overburden_bottom = topo_z - overburden_thickness

# 风化基岩过渡带（覆盖层底部往下 20~30m）
weathered_thickness = 20.0 + 10.0 * np.sin(2 * np.pi * y_cc / 500.0)
weathered_bottom = overburden_bottom - weathered_thickness

print(f"[2] 地形: [{topo_z.min():.1f}, {topo_z.max():.1f}] m")
print(f"    覆盖层: [{overburden_thickness.min():.0f}, {overburden_thickness.max():.0f}] m")
print(f"    风化带: [{weathered_thickness.min():.0f}, {weathered_thickness.max():.0f}] m")

# ── 3. 矿体几何定义 ──────────────────────────────────────────────────
ore_plate_center = np.array([300.0, 400.0, -200.0])
ore_plate_strike_len = 300.0
ore_plate_dip_len = 200.0
ore_plate_thickness = 15.0
ore_plate_dip_deg = 60.0
ore_plate_azimuth_deg = 45.0

ore_vein_center = np.array([650.0, 600.0, -350.0])
ore_vein_strike_len = 200.0
ore_vein_dip_len = 400.0
ore_vein_thickness = 8.0
ore_vein_dip_deg = 80.0
ore_vein_azimuth_deg = 135.0

HALO_WIDTH = 10.0  # 蚀变晕宽度（米）


def _in_dipping_plate(
    x, y, z, center, strike_len, dip_len, thickness, dip_deg, azimuth_deg
):
    """判断点是否在倾斜板状体内。"""
    az = np.radians(azimuth_deg)
    dip = np.radians(dip_deg)

    sx = np.sin(az)
    sy = np.cos(az)
    dx_dip = np.cos(az) * np.cos(dip)
    dy_dip = -np.sin(az) * np.cos(dip)
    dz_dip = -np.sin(dip)
    nx_ = np.cos(az) * np.sin(dip)
    ny_ = -np.sin(az) * np.sin(dip)
    nz_ = np.cos(dip)

    rx = x - center[0]
    ry = y - center[1]
    rz = z - center[2]

    u = rx * sx + ry * sy
    v = rx * dx_dip + ry * dy_dip + rz * dz_dip
    w = rx * nx_ + ry * ny_ + rz * nz_

    return u, v, w, strike_len, dip_len, thickness


def _classify_ore(u, v, w, strike_len, dip_len, thickness):
    """返回 (is_core, is_halo) 布尔掩码。"""
    in_core = (
        (np.abs(u) <= strike_len / 2)
        & (np.abs(v) <= dip_len / 2)
        & (np.abs(w) <= thickness / 2)
    )
    in_halo = (
        (np.abs(u) <= strike_len / 2 + HALO_WIDTH)
        & (np.abs(v) <= dip_len / 2 + HALO_WIDTH)
        & (np.abs(w) <= thickness / 2 + HALO_WIDTH)
        & (~in_core)
    )
    return in_core, in_halo


plate_uvw = _in_dipping_plate(
    x_cc, y_cc, z_cc, ore_plate_center,
    ore_plate_strike_len, ore_plate_dip_len, ore_plate_thickness,
    ore_plate_dip_deg, ore_plate_azimuth_deg,
)
is_ore_plate, is_halo_plate = _classify_ore(*plate_uvw)

vein_uvw = _in_dipping_plate(
    x_cc, y_cc, z_cc, ore_vein_center,
    ore_vein_strike_len, ore_vein_dip_len, ore_vein_thickness,
    ore_vein_dip_deg, ore_vein_azimuth_deg,
)
is_ore_vein, is_halo_vein = _classify_ore(*vein_uvw)

is_air = z_cc > topo_z
is_overburden = (~is_air) & (z_cc > overburden_bottom)
is_weathered = (~is_air) & (~is_overburden) & (z_cc > weathered_bottom)
is_bedrock = (
    (~is_air) & (~is_overburden) & (~is_weathered)
    & (~is_ore_plate) & (~is_ore_vein)
    & (~is_halo_plate) & (~is_halo_vein)
)

print(f"[3] 岩性分布:")
print(f"    空气层:   {is_air.sum():>10,} cells")
print(f"    覆盖层:   {is_overburden.sum():>10,} cells")
print(f"    风化带:   {is_weathered.sum():>10,} cells")
print(f"    基岩层:   {is_bedrock.sum():>10,} cells")
print(f"    板状矿体: {is_ore_plate.sum():>10,} cells")
print(f"    脉状矿体: {is_ore_vein.sum():>10,} cells")
print(f"    蚀变晕:   {(is_halo_plate | is_halo_vein).sum():>10,} cells")

# ── 4. 赋物性值 ───────────────────────────────────────────────────────
resistivity = np.full(mesh.nC, PROPS["bedrock"]["resistivity"])
conductivity = np.full(mesh.nC, PROPS["bedrock"]["conductivity"])
chargeability = np.full(mesh.nC, PROPS["bedrock"]["chargeability"])
lithology = np.zeros(mesh.nC, dtype=int)  # 0=bedrock

# Uniform layers: (mask, property_key, lithology_code)
_LAYERS = [
    (is_air,       "air",       -1),
    (is_weathered, "weathered",  4),
    (is_ore_plate, "ore_plate",  2),
    (is_ore_vein,  "ore_vein",   3),
]
for mask, key, lith_code in _LAYERS:
    resistivity[mask] = PROPS[key]["resistivity"]
    conductivity[mask] = PROPS[key]["conductivity"]
    chargeability[mask] = PROPS[key]["chargeability"]
    lithology[mask] = lith_code

# Overburden with ±20% random perturbation for heterogeneity
rng = np.random.default_rng(42)
n_over = is_overburden.sum()
perturbation = 1.0 + 0.2 * (2 * rng.random(n_over) - 1)
resistivity[is_overburden] = PROPS["overburden"]["resistivity"] * perturbation
conductivity[is_overburden] = 1.0 / resistivity[is_overburden]
chargeability[is_overburden] = PROPS["overburden"]["chargeability"]
lithology[is_overburden] = 1

# 蚀变晕：矿体边缘渐变（电阻率在矿体值和基岩值之间对数线性插值）


def _assign_halo(mask, uvw_result, ore_props):
    """蚀变晕电阻率渐变赋值。"""
    u, v, w, sl, dl, th = uvw_result
    # 到矿体核心边界的最小法向距离（归一化到 HALO_WIDTH）
    du = np.maximum(np.abs(u[mask]) - sl / 2, 0)
    dv = np.maximum(np.abs(v[mask]) - dl / 2, 0)
    dw = np.maximum(np.abs(w[mask]) - th / 2, 0)
    dist = np.sqrt(du**2 + dv**2 + dw**2)
    frac = np.clip(dist / HALO_WIDTH, 0, 1)
    # 对数线性插值：近矿体 → 矿体值，远矿体 → 基岩值
    log_rho = (
        np.log10(ore_props["resistivity"]) * (1 - frac)
        + np.log10(PROPS["bedrock"]["resistivity"]) * frac
    )
    resistivity[mask] = 10**log_rho
    conductivity[mask] = 1.0 / resistivity[mask]
    chargeability[mask] = ore_props["chargeability"] * (1 - frac) + PROPS["bedrock"]["chargeability"] * frac
    lithology[mask] = 5  # 蚀变晕


_assign_halo(is_halo_plate, plate_uvw, PROPS["ore_plate"])
_assign_halo(is_halo_vein, vein_uvw, PROPS["ore_vein"])

print(f"[4] 物性赋值完成:")
print(f"    电阻率: [{resistivity[~is_air].min():.1f}, {resistivity[~is_air].max():.0f}] Ohm.m")
print(f"    极化率: [{chargeability.min():.3f}, {chargeability.max():.3f}]")

# ── 5. 定义三口钻孔 ───────────────────────────────────────────────────
def _find_topo_z(x0, y0):
    """在 cell centers 中查找最接近 (x0, y0) 的地形高程。"""
    idx = np.argmin((x_cc - x0)**2 + (y_cc - y0)**2)
    return topo_z[idx]


def _make_vertical_bh(x, y, max_depth, n_pts):
    collar = np.array([x, y, _find_topo_z(x, y)])
    depth = np.linspace(0, max_depth, n_pts)
    traj = np.column_stack([
        np.full_like(depth, collar[0]),
        np.full_like(depth, collar[1]),
        collar[2] - depth,
    ])
    return collar, traj


bh01_collar, bh01_traj = _make_vertical_bh(300.0, 400.0, 600, 200)
bh02_collar, bh02_traj = _make_vertical_bh(650.0, 600.0, 800, 300)

# BH-03: 斜孔（方位 N45E，倾角 80 度从垂直量起），从 (100, 200) 出发
# 偏离垂直 10 度：水平偏移 = depth * sin(10) ≈ 87m @ 500m
bh03_collar = np.array([100.0, 200.0, _find_topo_z(100, 200)])
bh03_depth = np.linspace(0, 500, 200)
az3 = np.radians(45.0)
dev_angle = np.radians(10.0)  # 偏离垂直的角度 = 90 - 80
bh03_traj = np.column_stack([
    bh03_collar[0] + bh03_depth * np.sin(az3) * np.sin(dev_angle),
    bh03_collar[1] + bh03_depth * np.cos(az3) * np.sin(dev_angle),
    bh03_collar[2] - bh03_depth * np.cos(dev_angle),
])

boreholes = {
    "BH-01": {"collar": bh01_collar, "trajectory": bh01_traj, "desc": "vertical-plate-ore"},
    "BH-02": {"collar": bh02_collar, "trajectory": bh02_traj, "desc": "vertical-vein-ore"},
    "BH-03": {"collar": bh03_collar, "trajectory": bh03_traj, "desc": "inclined-80deg-plate"},
}

for name, bh in boreholes.items():
    traj_path = OUT_DIR / f"{name}_trajectory.csv"
    np.savetxt(traj_path, bh["trajectory"], delimiter=",", header="x,y,z", comments="")
    end = bh["trajectory"][-1]
    h_offset = np.sqrt((end[0] - bh["collar"][0])**2 + (end[1] - bh["collar"][1])**2)
    print(f"[5] {name} ({bh['desc']}): collar ({bh['collar'][0]:.0f}, {bh['collar'][1]:.0f}, {bh['collar'][2]:.1f}), "
          f"end z={end[2]:.0f}m, h_offset={h_offset:.0f}m")

# ── 6. 导出模型 ───────────────────────────────────────────────────────
models = {
    "resistivity": resistivity,
    "conductivity": conductivity,
    "chargeability": chargeability,
    "lithology": lithology.astype(float),
}
mesh_data = MeshWithModel(mesh=mesh, models=models)
vtu_path = save_mesh(mesh_data, OUT_DIR, "geo_model_3d")
print(f"[6] VTU export: {vtu_path}")

np.savez_compressed(
    OUT_DIR / "geo_model_3d.npz",
    resistivity=resistivity,
    conductivity=conductivity,
    chargeability=chargeability,
    lithology=lithology,
    mesh_origin=np.array(mesh.origin),
    mesh_hx=mesh.h[0],
    mesh_hy=mesh.h[1],
    mesh_hz=mesh.h[2],
)
print(f"[6] NPZ export: {OUT_DIR / 'geo_model_3d.npz'}")

meta = {
    "domain": DOMAIN,
    "cell_size": {"dx": DX, "dy": DY, "dz": DZ},
    "n_cells": mesh.nC,
    "shape": list(mesh.shape_cells),
    "origin": list(mesh.origin),
    "properties": PROPS,
    "lithology_codes": {
        "-1": "air", "0": "bedrock", "1": "overburden",
        "2": "ore_plate", "3": "ore_vein", "4": "weathered", "5": "halo",
    },
    "boreholes": {
        name: {
            "collar": bh["collar"].tolist(),
            "end": bh["trajectory"][-1].tolist(),
            "desc": bh["desc"],
            "n_points": len(bh["trajectory"]),
        }
        for name, bh in boreholes.items()
    },
    "ore_bodies": {
        "plate": {
            "center": ore_plate_center.tolist(),
            "strike_length": ore_plate_strike_len,
            "dip_length": ore_plate_dip_len,
            "thickness": ore_plate_thickness,
            "dip_deg": ore_plate_dip_deg,
            "azimuth_deg": ore_plate_azimuth_deg,
            "resistivity": PROPS["ore_plate"]["resistivity"],
            "halo_width": HALO_WIDTH,
        },
        "vein": {
            "center": ore_vein_center.tolist(),
            "strike_length": ore_vein_strike_len,
            "dip_length": ore_vein_dip_len,
            "thickness": ore_vein_thickness,
            "dip_deg": ore_vein_dip_deg,
            "azimuth_deg": ore_vein_azimuth_deg,
            "resistivity": PROPS["ore_vein"]["resistivity"],
            "halo_width": HALO_WIDTH,
        },
    },
}
(OUT_DIR / "geo_model_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ── 7. 可视化 ─────────────────────────────────────────────────────────
import pyvista
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(shape=(2, 2), window_size=(1800, 1400), off_screen=True)
plotter.set_background("#f7f9fc")

vtk_obj = mesh.to_vtk(models=models)
active = vtk_obj["lithology"] >= 0
sub = vtk_obj.extract_cells(np.where(active)[0])

# (0,0) resistivity log-scale
plotter.subplot(0, 0)
plotter.add_text("Resistivity (Ohm.m)", font_size=10)
plotter.add_mesh(
    sub, scalars="resistivity", cmap="jet_r",
    log_scale=True, show_edges=False, opacity=0.9,
    scalar_bar_args={"title": "rho (Ohm.m)", "fmt": "%.0f"},
)
plotter.add_axes()
plotter.camera.azimuth = 225
plotter.camera.elevation = 30

# (0,1) chargeability
plotter.subplot(0, 1)
plotter.add_text("Chargeability", font_size=10)
plotter.add_mesh(
    sub, scalars="chargeability", cmap="hot_r",
    show_edges=False, opacity=0.9,
    scalar_bar_args={"title": "eta", "fmt": "%.2f"},
)
plotter.add_axes()
plotter.camera.azimuth = 225
plotter.camera.elevation = 30

# (1,0) lithology + boreholes
plotter.subplot(1, 0)
plotter.add_text("Lithology + Boreholes", font_size=10)

ore_cells = np.where((sub["lithology"] == 2) | (sub["lithology"] == 3))[0]
if len(ore_cells) > 0:
    ore_mesh = sub.extract_cells(ore_cells)
    plotter.add_mesh(ore_mesh, scalars="lithology", cmap="Set1", show_edges=True, opacity=1.0)

halo_cells = np.where(sub["lithology"] == 5)[0]
if len(halo_cells) > 0:
    halo_mesh = sub.extract_cells(halo_cells)
    plotter.add_mesh(halo_mesh, color="#ffcc66", opacity=0.3, show_edges=False)

over_cells = np.where(sub["lithology"] == 1)[0]
if len(over_cells) > 0:
    over_mesh = sub.extract_cells(over_cells)
    plotter.add_mesh(over_mesh, color="#d4a574", opacity=0.15, show_edges=False)

colors = {"BH-01": "blue", "BH-02": "green", "BH-03": "red"}
for name, bh in boreholes.items():
    traj = bh["trajectory"]
    line = pyvista.Spline(traj, n_points=max(len(traj), 100))
    tube = line.tube(radius=3.0)
    plotter.add_mesh(tube, color=colors[name], label=name)
    plotter.add_point_labels(
        [traj[0]], [name], font_size=12, point_color=colors[name],
        text_color=colors[name], bold=True, shape_opacity=0.5,
    )
plotter.add_legend()
plotter.add_axes()
plotter.camera.azimuth = 225
plotter.camera.elevation = 30

# (1,1) Y=400 cross-section
plotter.subplot(1, 1)
plotter.add_text("Y=400m Section - Resistivity", font_size=10)
sliced = vtk_obj.slice(normal="y", origin=(0, 400, 0))
active_slice = sliced["lithology"] >= 0
slice_cells = np.where(active_slice)[0]
if len(slice_cells) > 0:
    sliced_sub = sliced.extract_cells(slice_cells)
    plotter.add_mesh(
        sliced_sub, scalars="resistivity", cmap="jet_r",
        log_scale=True,
        scalar_bar_args={"title": "rho (Ohm.m)", "fmt": "%.0f"},
    )
for name, bh in boreholes.items():
    traj = bh["trajectory"]
    line = pyvista.Spline(traj, n_points=max(len(traj), 100))
    plotter.add_mesh(line, color=colors[name], line_width=3)
plotter.add_axes()
plotter.view_xz()

png_path = OUT_DIR / "geo_model_3d_overview.png"
plotter.screenshot(str(png_path))
plotter.close()
print(f"[7] Screenshot: {png_path}")

# ── 8. 写入工作区 ─────────────────────────────────────────────────────
store = DataStore(WORKSPACE)
store.init_workspace()
store.create_borehole(BH_NAME)

for method in ("ert", "ip", "em", "tem"):
    raw_dir, _ = store.ensure_method_dirs(BH_NAME, method)
    for f in OUT_DIR.glob("*.npz"):
        shutil.copy(f, raw_dir / f.name)
    for f in OUT_DIR.glob("*_trajectory.csv"):
        shutil.copy(f, raw_dir / f.name)

print(f"[8] Model copied to workspace: ~/geophys-workspace/boreholes/{BH_NAME}/")

print("\n" + "=" * 70)
print("Done! Output files:")
for f in sorted(OUT_DIR.iterdir()):
    size = f.stat().st_size
    unit = "KB" if size > 1024 else "B"
    val = size / 1024 if size > 1024 else size
    print(f"  {f.name:40s} {val:8.1f} {unit}")
print("=" * 70)
