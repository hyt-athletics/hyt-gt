"""精细模型三维可视化 — EMIGMA / Voxler / Petrel 综合图板。

6 视图布局：
  (0,0) 3D 全景：矿体 + 蚀变晕 + 钻孔 + 地形 + 坐标网格
  (0,1) 嵌套等值面：rho=5/50/500 三级半透明 + 地层界面
  (1,0) Fence 剖面 1：BH-01 → BH-03（穿越板状矿体）
  (1,1) Fence 剖面 2：BH-03 → BH-02（穿越脉状矿体）
  (2,0) EMIGMA 式多道测井沿孔展示
  (2,1) Z=-200m 俯视切片 + 矿体投影

运行：.venv/bin/python tests/demo_3d_viewer.py
"""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pyvista
pyvista.OFF_SCREEN = True

MODEL_DIR = _ROOT / "tests" / "data" / "geo_model_3d"
OUT_PNG = MODEL_DIR / "demo_3d_emigma_style.png"

# ── 加载模型 ──────────────────────────────────────────────────────────
meta = json.loads((MODEL_DIR / "geo_model_meta.json").read_text(encoding="utf-8"))
md = np.load(MODEL_DIR / "geo_model_3d.npz")

from discretize import TensorMesh
mesh = TensorMesh([md["mesh_hx"], md["mesh_hy"], md["mesh_hz"]], origin=md["mesh_origin"])

resistivity = md["resistivity"]
chargeability = md["chargeability"]
lithology = md["lithology"]

vtk_full = mesh.to_vtk(models={
    "resistivity": resistivity,
    "chargeability": chargeability,
    "lithology": lithology.astype(float),
})

full_lith = np.asarray(vtk_full["lithology"])
sub = vtk_full.extract_cells(np.where(full_lith >= 0)[0])

sub_lith = np.asarray(sub["lithology"])
ore_idx = np.where((sub_lith == 2) | (sub_lith == 3))[0]
halo_idx = np.where(sub_lith == 5)[0]
over_idx = np.where(sub_lith == 1)[0]
ore = sub.extract_cells(ore_idx) if len(ore_idx) > 0 else None
halo = sub.extract_cells(halo_idx) if len(halo_idx) > 0 else None
over = sub.extract_cells(over_idx) if len(over_idx) > 0 else None

# 钻孔
boreholes = {}
for name in ("BH-01", "BH-02", "BH-03"):
    boreholes[name] = np.loadtxt(str(MODEL_DIR / f"{name}_trajectory.csv"), delimiter=",", skiprows=1)
BH_COLORS = {"BH-01": "#1565C0", "BH-02": "#2E7D32", "BH-03": "#C62828"}

bh01_collar = boreholes["BH-01"][0]
bh02_collar = boreholes["BH-02"][0]
bh03_end = boreholes["BH-03"][-1]

# 地形面
cc = mesh.cell_centers
x_u, y_u = np.unique(cc[:, 0]), np.unique(cc[:, 1])
xg, yg = np.meshgrid(x_u, y_u)
zg = 25.0 * np.sin(2*np.pi*xg/800) * np.cos(2*np.pi*yg/600) + 10.0 * np.sin(2*np.pi*xg/300 + 0.5)
topo = pyvista.StructuredGrid(xg, yg, zg)

FORMATIONS = [
    (0, 60, "Overburden", "#C8A870"),
    (60, 180, "Weathered", "#A0A0A0"),
    (180, 600, "Fresh Bedrock", "#707070"),
]


def _make_borehole_logs(traj):
    collar_z = traj[0, 2]
    depths = collar_z - traj[:, 2]
    n = len(traj)
    rng = np.random.default_rng(42)
    gr = np.full(n, 30.0) + rng.standard_normal(n) * 5
    gr[depths < 60] = 100.0 + rng.standard_normal(np.sum(depths < 60)) * 15
    cond = np.full(n, 0.0002) + np.abs(rng.standard_normal(n)) * 0.00005
    cond[depths < 60] = 0.007 + np.abs(rng.standard_normal(np.sum(depths < 60))) * 0.002
    for i, pt in enumerate(traj):
        d_plate = np.sqrt((pt[0]-300)**2 + (pt[1]-400)**2 + (pt[2]+200)**2)
        d_vein = np.sqrt((pt[0]-650)**2 + (pt[1]-600)**2 + (pt[2]+350)**2)
        if d_plate < 80:
            gr[i] = 150 + rng.standard_normal() * 20
            cond[i] = 0.2 * (1 - d_plate/80) + 0.001
        elif d_vein < 60:
            gr[i] = 180 + rng.standard_normal() * 25
            cond[i] = 0.5 * (1 - d_vein/60) + 0.001
    return {"GR": gr, "COND": cond * 1000}


def setup_lights(p):
    p.remove_all_lights()
    p.add_light(pyvista.Light(position=(1, 1, 1), intensity=0.8, light_type="scenelight"))
    p.add_light(pyvista.Light(position=(-1, 0.5, 0.5), intensity=0.3, light_type="scenelight"))
    p.add_light(pyvista.Light(position=(0, -1, -0.5), intensity=0.15, light_type="scenelight"))


def add_boreholes_simple(p, with_labels=True):
    for name, traj in boreholes.items():
        line = pyvista.Spline(traj, n_points=100)
        tube = line.tube(radius=3.0)
        p.add_mesh(tube, color=BH_COLORS[name])
        if with_labels:
            p.add_point_labels([traj[0]], [name], font_size=11, text_color=BH_COLORS[name],
                               bold=True, shape_opacity=0.4, point_size=0)


print("Generating 6-view EMIGMA/Voxler/Petrel composite...")

# ══════════════════════════════════════════════════════════════════════
plotter = pyvista.Plotter(
    shape=(3, 2), window_size=(2000, 2400), off_screen=True,
    border=True, border_color="#cccccc", border_width=1,
)
plotter.set_background("#f0f2f5")
bounds = sub.bounds

# ── (0,0) 3D 全景 ─────────────────────────────────────────────────────
plotter.subplot(0, 0)
setup_lights(plotter)
plotter.add_text("3D Panorama: Ore + Halo + Boreholes",
                 font_size=10, color="#1a1a2e", position="upper_left")

if ore is not None:
    plotter.add_mesh(ore, scalars="resistivity", cmap="jet_r", log_scale=True,
                     show_edges=True, edge_color="#888888", line_width=0.3,
                     opacity=1.0, scalar_bar_args={"title": "rho (Ohm.m)", "fmt": "%.1f",
                                                   "n_labels": 5, "position_x": 0.02})
if halo is not None:
    plotter.add_mesh(halo, color="#ffcc66", opacity=0.25, show_edges=False)
if over is not None:
    plotter.add_mesh(over, color="#C8A870", opacity=0.06, show_edges=False)
plotter.add_mesh(topo, cmap="terrain", opacity=0.3, show_edges=False, lighting=True)

# Formation-segmented boreholes with depth marks
for name, traj in boreholes.items():
    color = BH_COLORS[name]
    collar_z = traj[0, 2]
    depths = collar_z - traj[:, 2]
    total = depths[-1]
    for top, bot, _fm, fm_color in FORMATIONS:
        if top >= total:
            continue
        idx = np.where((depths >= top) & (depths <= min(bot, total)))[0]
        if len(idx) < 2:
            continue
        seg = traj[idx[0]:idx[-1]+1]
        line = pyvista.Spline(seg, n_points=max(len(seg), 30))
        tube = line.tube(radius=4.0)
        plotter.add_mesh(tube, color=fm_color, lighting=True)
    plotter.add_point_labels([traj[0]], [name], font_size=13, text_color=color,
                             bold=True, shape_opacity=0.5, point_size=0)
    for d in np.arange(100, total, 100):
        idx = np.argmin(np.abs(depths - d))
        plotter.add_point_labels([traj[idx]], [f"{d:.0f}m"], font_size=7,
                                 text_color="#444444", shape_opacity=0.2, always_visible=True)

plotter.show_bounds(mesh=sub, grid="back", location="outer", font_size=7,
                    color="#666666", xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)")
arrow = pyvista.Arrow(start=(bounds[1]+40, bounds[2], bounds[5]), direction=(0, 1, 0), scale=50)
plotter.add_mesh(arrow, color="#333333")
plotter.add_point_labels([(bounds[1]+40, bounds[2]+60, bounds[5])], ["N"],
                         font_size=14, text_color="#333333", bold=True, shape_opacity=0.0)
plotter.camera.azimuth = 225
plotter.camera.elevation = 25

# ── (0,1) 嵌套等值面 ──────────────────────────────────────────────────
plotter.subplot(0, 1)
setup_lights(plotter)
plotter.add_text("Nested Isosurfaces: rho=5/50/500 Ohm.m",
                 font_size=10, color="#1a1a2e", position="upper_left")

iso_specs = [
    (5, "#FF2222", 0.8, "Ore core"),
    (50, "#FF8800", 0.35, "Alteration"),
    (500, "#4488FF", 0.12, "OB base"),
]
legend_labels = []
for iso_val, iso_color, iso_opacity, iso_label in iso_specs:
    try:
        contour = sub.contour(isosurfaces=[iso_val], scalars="resistivity")
        if contour.n_points > 0:
            plotter.add_mesh(contour, color=iso_color, opacity=iso_opacity, smooth_shading=True)
            legend_labels.append((f"rho={iso_val}: {iso_label}", iso_color))
    except Exception:
        pass

# Formation boundary surface from lithology
try:
    lith_contour = sub.contour(isosurfaces=[0.5], scalars="lithology")
    if lith_contour.n_points > 0:
        plotter.add_mesh(lith_contour, color="#886644", opacity=0.15, smooth_shading=True)
        legend_labels.append(("Bedrock surface", "#886644"))
except Exception:
    pass

add_boreholes_simple(plotter)
plotter.add_mesh(topo, cmap="terrain", opacity=0.2, show_edges=False)
if legend_labels:
    plotter.add_legend(labels=legend_labels, size=(0.2, 0.15))
plotter.show_bounds(mesh=sub, grid="back", location="outer", font_size=7, color="#666666")
plotter.camera.azimuth = 240
plotter.camera.elevation = 30

def _add_fence_section(p, row_col, p0_xy, p1_xy, title):
    """Add an oblique vertical slice along two XY points to the plotter."""
    p.subplot(*row_col)
    setup_lights(p)
    p.add_text(title, font_size=10, color="#1a1a2e", position="upper_left")

    direction = np.array(p1_xy) - np.array(p0_xy)
    normal = np.array([-direction[1], direction[0], 0.0])
    normal /= np.linalg.norm(normal)
    midpoint = np.array([(p0_xy[0]+p1_xy[0])/2, (p0_xy[1]+p1_xy[1])/2, -400])

    sl = vtk_full.slice(normal=normal, origin=midpoint)
    act = np.asarray(sl["lithology"]) >= 0
    if np.any(act):
        s = sl.extract_cells(np.where(act)[0])
        p.add_mesh(s, scalars="resistivity", cmap="jet_r", log_scale=True,
                   scalar_bar_args={"title": "rho (Ohm.m)", "fmt": "%.1f",
                                    "n_labels": 5, "position_x": 0.02})

    add_boreholes_simple(p)
    p.show_bounds(mesh=sub, grid="back", location="outer", font_size=7, color="#666666")
    p.camera.azimuth = 210
    p.camera.elevation = 20


_add_fence_section(
    plotter, (1, 0),
    (bh01_collar[0], bh01_collar[1]), (bh03_end[0], bh03_end[1]),
    "Fence Section 1: BH-01 -> BH-03 (Plate Ore)",
)
_add_fence_section(
    plotter, (1, 1),
    (bh03_end[0], bh03_end[1]), (bh02_collar[0], bh02_collar[1]),
    "Fence Section 2: BH-03 -> BH-02 (Vein Ore)",
)

# ── (2,0) EMIGMA 式多道测井 ───────────────────────────────────────────
plotter.subplot(2, 0)
setup_lights(plotter)
plotter.add_text("Multi-track Borehole Logs (GR + COND)",
                 font_size=10, color="#1a1a2e", position="upper_left")

if ore is not None:
    plotter.add_mesh(ore, style="wireframe", color="#FFCC00", line_width=0.5, opacity=0.3)
plotter.add_mesh(topo, cmap="terrain", opacity=0.2, show_edges=False)

for name, traj in boreholes.items():
    color = BH_COLORS[name]
    logs = _make_borehole_logs(traj)
    n_pts = 200

    main_line = pyvista.Spline(traj, n_points=n_pts)
    main_tube = main_line.tube(radius=2.0)
    plotter.add_mesh(main_tube, color="#444444")

    gr_traj = traj.copy()
    gr_traj[:, 0] += 15
    gr_line = pyvista.Spline(gr_traj, n_points=n_pts)
    gr_interp = np.interp(np.linspace(0, 1, n_pts), np.linspace(0, 1, len(logs["GR"])), logs["GR"])
    gr_line["GR"] = gr_interp
    gr_tube = gr_line.tube(radius=6.0)
    plotter.add_mesh(gr_tube, scalars="GR", cmap="YlOrRd",
                     scalar_bar_args={"title": "GR (API)", "fmt": "%.0f", "position_x": 0.02})

    cond_traj = traj.copy()
    cond_traj[:, 0] -= 15
    cond_line = pyvista.Spline(cond_traj, n_points=n_pts)
    cond_interp = np.interp(np.linspace(0, 1, n_pts), np.linspace(0, 1, len(logs["COND"])), logs["COND"])
    cond_line["COND"] = cond_interp
    cond_tube = cond_line.tube(radius=6.0)
    plotter.add_mesh(cond_tube, scalars="COND", cmap="viridis",
                     scalar_bar_args={"title": "COND (mS/m)", "fmt": "%.1f", "position_x": 0.15})

    plotter.add_point_labels([traj[0]], [f"{name}\nGR|COND"], font_size=10,
                             text_color=color, bold=True, shape_opacity=0.5, point_size=0)

plotter.show_bounds(mesh=sub, grid="back", location="outer", font_size=7, color="#666666")
plotter.camera.azimuth = 225
plotter.camera.elevation = 25

# ── (2,1) Z=-200m 水平切片俯视 ────────────────────────────────────────
plotter.subplot(2, 1)
setup_lights(plotter)
plotter.add_text("Plan View @ Z=-200m + Ore Projection",
                 font_size=10, color="#1a1a2e", position="upper_left")

hz_slice = vtk_full.slice(normal="z", origin=(0, 0, -200))
act_hz = np.asarray(hz_slice["lithology"]) >= 0
if np.any(act_hz):
    hz_sub = hz_slice.extract_cells(np.where(act_hz)[0])
    plotter.add_mesh(hz_sub, scalars="resistivity", cmap="jet_r", log_scale=True,
                     scalar_bar_args={"title": "rho @ Z=-200m", "fmt": "%.1f",
                                      "n_labels": 5, "position_x": 0.02})

for name, traj in boreholes.items():
    plotter.add_mesh(pyvista.PolyData([traj[0]]), color=BH_COLORS[name],
                     point_size=12, render_points_as_spheres=True)
    plotter.add_point_labels([traj[0]], [name], font_size=12, text_color=BH_COLORS[name],
                             bold=True, shape_opacity=0.0, point_size=0)

arrow2 = pyvista.Arrow(start=(bounds[1]+30, bounds[2], -200), direction=(0, 1, 0), scale=40)
plotter.add_mesh(arrow2, color="#333333")
plotter.add_point_labels([(bounds[1]+30, bounds[2]+50, -200)], ["N"],
                         font_size=14, text_color="#333333", bold=True, shape_opacity=0.0)

s0 = np.array([bounds[0], bounds[2]-40, -200])
plotter.add_mesh(pyvista.Line(s0, s0+[200, 0, 0]), color="black", line_width=3)
for f, lb in [(0, "0"), (1, "200m")]:
    plotter.add_point_labels([s0+[200*f, 0, 0]], [lb], font_size=9,
                             text_color="black", shape_opacity=0.0)

plotter.show_bounds(mesh=sub, grid="back", location="outer", font_size=7, color="#666666")
plotter.camera.parallel_projection = True
plotter.view_xy()

# ── 保存 ──────────────────────────────────────────────────────────────
plotter.screenshot(str(OUT_PNG))
plotter.close()
print(f"Done: {OUT_PNG} ({OUT_PNG.stat().st_size/1024:.0f} KB)")
