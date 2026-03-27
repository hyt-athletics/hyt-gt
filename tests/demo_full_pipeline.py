"""井中物探全流程验证演示。

将 1D TEM 正反演、测井滤波、3D 地质模型、2.5D 剖面可视化
整合为一个完整的端到端工作流，用于：
  - 验证整个平台管线的可靠性
  - 为后续汇报提供可演示的成果物
  - 给算法开发者提供集成参考模板

工作流程：
  ┌─────────────────────────────────────────────────────────────┐
  │ Phase 1: 三维地质模型加载 + 钻孔几何                        │
  │   → 1000×1000×1000m 域，3 口井，板状/脉状矿体              │
  ├─────────────────────────────────────────────────────────────┤
  │ Phase 2: TEM 正演（empymod，合成 1D 层状模型）              │
  │   → 生成 dB/dt 时间序列 + 5% 高斯噪声                     │
  ├─────────────────────────────────────────────────────────────┤
  │ Phase 3: TEM 一维 Occam 反演                               │
  │   → 恢复电阻率-深度模型                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ Phase 4: 测井曲线多算法滤波                                 │
  │   → SG / 小波 / Kalman 三滤波对比                          │
  ├─────────────────────────────────────────────────────────────┤
  │ Phase 5: 综合可视化（2D + 2.5D + 3D）                      │
  │   → 9 子图板式输出，涵盖全部可视化能力                     │
  └─────────────────────────────────────────────────────────────┘

运行：.venv/bin/python tests/demo_full_pipeline.py

输出：tests/data/demo_full_pipeline.png（汇报用综合图板）
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from core.data_store import DataStore
from core.font_utils import setup_matplotlib_cjk
from algorithms.registry import AlgorithmRegistry
from algorithms.runner import AlgorithmRunner

setup_matplotlib_cjk()

# ── 配置 ──────────────────────────────────────────────────────────────
WORKSPACE = Path.home() / "geophys-workspace"
ALGO_DIR = _ROOT / "algorithms"
MODEL_DIR = _ROOT / "tests" / "data" / "geo_model_3d"
OUT_PNG = _ROOT / "tests" / "data" / "demo_full_pipeline.png"
OUT_DIR = _ROOT / "tests" / "data"

BH_TEM = "DEMO-TEM-FULL"
BH_LOG = "DEMO-LOG-FULL"
LOOP_R = 50.0

store = DataStore(WORKSPACE)
store.init_workspace()
registry = AlgorithmRegistry(ALGO_DIR)
registry.scan()
runner = AlgorithmRunner(registry, WORKSPACE)

print("=" * 72)
print("  井中物探全流程验证演示")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════════
# Phase 1: 加载三维地质模型
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 1] 加载三维地质模型...")
meta = json.loads((MODEL_DIR / "geo_model_meta.json").read_text(encoding="utf-8"))
model_data = np.load(MODEL_DIR / "geo_model_3d.npz")

boreholes = {}
for name in ("BH-01", "BH-02", "BH-03"):
    traj = np.loadtxt(str(MODEL_DIR / f"{name}_trajectory.csv"), delimiter=",", skiprows=1)
    boreholes[name] = traj

print(f"  域: {meta['domain']['x']}×{meta['domain']['y']}×{meta['domain']['z']} m")
print(f"  网格: {meta['n_cells']} cells")
print(f"  钻孔: {', '.join(boreholes.keys())}")
print(f"  矿体: 板状(ρ=5Ω·m, 300×200×15m) + 脉状(ρ=2Ω·m, 200×400×8m)")

# ══════════════════════════════════════════════════════════════════════
# Phase 2: TEM 正演
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 2] TEM 正演（empymod 1D 层状模型）...")

store.create_borehole(BH_TEM)
raw_dir, _ = store.ensure_method_dirs(BH_TEM, "tem")

TRUE_RHO = np.array([2e14, 200.0, 10.0, 1000.0])
TRUE_DEPTHS = np.array([50.0, 150.0])

np.savetxt(raw_dir / "resistivity.csv", TRUE_RHO, delimiter=",")
np.savetxt(raw_dir / "depths.csv", TRUE_DEPTHS, delimiter=",")

fwd_params = {"loop_radius": str(LOOP_R), "t_min": "-5.0", "t_max": "-2.0", "n_times": "30"}
saved_fwd = runner.run("empymod_tem_forward", BH_TEM, fwd_params)

emf_clean = np.loadtxt(str(saved_fwd["emf"]), delimiter=",")
times = np.loadtxt(str(saved_fwd["times"]), delimiter=",")
rho_app = np.loadtxt(str(saved_fwd["rho_app"]), delimiter=",")
print(f"  正演完成: {len(times)} 时道, EMF范围 [{emf_clean.min():.2e}, {emf_clean.max():.2e}]")

# 添加噪声
rng = np.random.default_rng(42)
emf_noisy = np.abs(emf_clean * (1.0 + 0.05 * rng.standard_normal(len(emf_clean))))
np.savetxt(raw_dir / "emf.csv", emf_noisy.reshape(1, -1), delimiter=",")
np.savetxt(raw_dir / "times.csv", times, delimiter=",")
print(f"  添加 5% 高斯噪声 → emf_noisy")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: TEM 一维 Occam 反演
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 3] TEM 一维 Occam 反演...")
inv_params = {"n_layers": "20", "lambda_": "0.001", "depth_max": "500.0", "loop_radius": str(LOOP_R)}
saved_inv = runner.run("tem_1d_inversion", BH_TEM, inv_params)

rho_inv = np.loadtxt(str(saved_inv["resistivity"]), delimiter=",")
tops_inv = np.loadtxt(str(saved_inv["layer_tops"]), delimiter=",")
misfit = float(Path(saved_inv["misfit"]).read_text(encoding="utf-8"))
print(f"  反演完成: {len(rho_inv)} 层, RMS = {misfit:.4f}")

# ══════════════════════════════════════════════════════════════════════
# Phase 4: 测井曲线多算法滤波
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 4] 测井曲线三滤波对比...")
LAS_FILE = _ROOT / "tests" / "data" / "6038187_v1.2_short.las"

store.create_borehole(BH_LOG)
from core.importer import DataImporter
importer = DataImporter(WORKSPACE)
channels = importer.import_file(LAS_FILE, BH_LOG, method="logging")

import shutil
raw_log = WORKSPACE / "boreholes" / BH_LOG / "logging" / "raw"
gamn = raw_log / "GAMN.csv"
gr_dst = raw_log / "GR.csv"
if gamn.exists() and not gr_dst.exists():
    shutil.copy(gamn, gr_dst)

import lasio
las = lasio.read(str(LAS_FILE))
depth = las["DEPT"]
gr_raw = las["GAMN"]

filter_results = {}
for algo, params in [
    ("log_preproc_savgol", {"window_length": "11", "polyorder": "3"}),
    ("log_preproc_wavelet", {"wavelet": "db4", "level": "4"}),
    ("log_preproc_kalman", {"process_noise": "0.01", "measurement_noise": "1.0"}),
]:
    saved = runner.run(algo, BH_LOG, params)
    filt = np.loadtxt(str(saved["curve_filtered"]), delimiter=",")
    snr = float(Path(saved["snr_improvement"]).read_text(encoding="utf-8"))
    filter_results[algo] = (filt, snr)
    print(f"  {algo}: SNR↑{snr:.1f} dB")

# ══════════════════════════════════════════════════════════════════════
# Phase 5: 综合可视化图板
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 5] 生成综合可视化图板...")

fig = plt.figure(figsize=(28, 20), dpi=150)
fig.patch.set_facecolor("#f7f9fc")
fig.suptitle(
    "井中物探数据处理与反演子系统 — 全流程验证演示\n"
    "平台集成验证：TEM 正反演 · 测井滤波 · 三维地质建模 · 多维度可视化",
    fontsize=16, fontweight="bold", y=0.98, color="#1a1a2e",
)

gs = gridspec.GridSpec(
    3, 4, figure=fig,
    left=0.04, right=0.97, top=0.92, bottom=0.04,
    wspace=0.30, hspace=0.35,
)

# ── (0,0) TEM 正演：dB/dt 曲线 ───────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
ax00.loglog(times, emf_clean, "g-", lw=2.0, label="无噪声正演")
ax00.loglog(times, emf_noisy, "r.", ms=4, alpha=0.7, label="含噪观测(5%)")
ax00.set_xlabel("时间 (s)")
ax00.set_ylabel("|dB/dt| (V/A·m²)")
ax00.set_title("① TEM 正演响应", fontweight="bold")
ax00.legend(fontsize=8)
ax00.grid(which="both", ls=":", lw=0.4, color="#ccc")

# ── (0,1) TEM 反演：模型对比 ──────────────────────────────────────────
ax01 = fig.add_subplot(gs[0, 1])


def _step_plot(ax, rhos, tops, bottom=500.0, **kwargs):
    xs, ys = [], []
    for i, rho in enumerate(rhos):
        d0 = tops[i] if i < len(tops) else bottom
        d1 = tops[i + 1] if i + 1 < len(tops) else bottom
        xs += [rho, rho]
        ys += [d0, d1]
    ax.plot(xs, ys, **kwargs)


true_rhos = TRUE_RHO[1:]
true_tops = np.concatenate([[0.0], TRUE_DEPTHS])
_step_plot(ax01, true_rhos, true_tops, color="tab:red", lw=2.5, label="真实模型")
_step_plot(ax01, rho_inv, tops_inv, color="tab:blue", lw=1.6, ls="--",
           label=f"反演结果 RMS={misfit:.3f}")
ax01.set_xscale("log")
ax01.invert_yaxis()
ax01.set_xlabel("电阻率 (Ω·m)")
ax01.set_ylabel("深度 (m)")
ax01.set_title("② TEM 反演模型对比", fontweight="bold")
ax01.axhspan(TRUE_DEPTHS[0], TRUE_DEPTHS[1], alpha=0.12, color="tab:orange")
ax01.text(8, (TRUE_DEPTHS[0] + TRUE_DEPTHS[1]) / 2, "矿化层\nρ=10Ω·m",
          fontsize=7, color="tab:orange", va="center")
ax01.legend(fontsize=8)
ax01.grid(which="both", ls=":", lw=0.4, color="#ccc")

# ── (0,2) 视电阻率曲线 ───────────────────────────────────────────────
ax02 = fig.add_subplot(gs[0, 2])
valid = np.isfinite(rho_app) & (rho_app > 0)
ax02.loglog(times[valid], rho_app[valid], "s-", color="tab:purple", ms=3, lw=1.5, label="ρₐ(t)")
ax02.axhline(10, color="tab:orange", ls="--", lw=1, label="矿化层 10Ω·m")
ax02.axhline(200, color="tab:green", ls="--", lw=1, label="覆盖层 200Ω·m")
ax02.set_xlabel("时间 (s)")
ax02.set_ylabel("视电阻率 (Ω·m)")
ax02.set_title("③ 视电阻率曲线", fontweight="bold")
ax02.legend(fontsize=8)
ax02.grid(which="both", ls=":", lw=0.4, color="#ccc")

# ── (0,3) 反演拟合残差 ───────────────────────────────────────────────
ax03 = fig.add_subplot(gs[0, 3])
ax03.text(0.5, 0.65, f"RMS Misfit\n{misfit:.4f}", transform=ax03.transAxes,
          ha="center", va="center", fontsize=28, fontweight="bold", color="#2196F3")
ax03.text(0.5, 0.35, f"反演层数: {len(rho_inv)}\n深度范围: 0-500m\n正则化 λ=0.001",
          transform=ax03.transAxes, ha="center", va="center", fontsize=10, color="#666")
ax03.text(0.5, 0.10, f"回线半径: {LOOP_R}m | empymod + L-BFGS-B",
          transform=ax03.transAxes, ha="center", va="center", fontsize=8, color="#999")
ax03.set_xlim(0, 1)
ax03.set_ylim(0, 1)
ax03.set_axis_off()
ax03.set_title("④ 反演质量指标", fontweight="bold")

# ── (1,0) 测井曲线三滤波叠显 ─────────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.plot(gr_raw, depth, color="#ccc", lw=0.8, label="原始 GR", zorder=1)
FCOLORS = {
    "log_preproc_savgol": ("SG", "tab:green"),
    "log_preproc_wavelet": ("小波", "tab:orange"),
    "log_preproc_kalman": ("Kalman", "tab:blue"),
}
for algo, (label, color) in FCOLORS.items():
    filt, snr = filter_results[algo]
    ax10.plot(filt, depth, color=color, lw=1.4, label=f"{label} SNR↑{snr:.1f}dB", zorder=2)
ax10.invert_yaxis()
ax10.set_xlabel("GR (GAPI)")
ax10.set_ylabel("深度 (m)")
ax10.set_title("⑤ 测井曲线三滤波对比", fontweight="bold")
ax10.legend(fontsize=7, loc="lower right")
ax10.grid(ls=":", lw=0.4, color="#ccc")

# ── (1,1) 滤波残差 ───────────────────────────────────────────────────
ax11 = fig.add_subplot(gs[1, 1])
for algo, (label, color) in FCOLORS.items():
    filt, _ = filter_results[algo]
    ax11.plot(gr_raw - filt, depth, color=color, lw=1.0, alpha=0.85, label=label)
ax11.axvline(0, color="black", lw=0.8, ls="--")
ax11.invert_yaxis()
ax11.set_xlabel("滤波残差 (GAPI)")
ax11.set_ylabel("深度 (m)")
ax11.set_title("⑥ 滤波残差对比", fontweight="bold")
ax11.legend(fontsize=8)
ax11.grid(ls=":", lw=0.4, color="#ccc")

# SNR 柱状图（嵌入）
ax_inset = ax11.inset_axes([0.55, 0.02, 0.42, 0.22])
labels_s = [FCOLORS[a][0] for a in filter_results]
snrs = [filter_results[a][1] for a in filter_results]
colors_s = [FCOLORS[a][1] for a in filter_results]
bars = ax_inset.bar(range(3), snrs, color=colors_s, alpha=0.8, edgecolor="white")
for bar, v in zip(bars, snrs):
    ax_inset.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")
ax_inset.set_xticks(range(3))
ax_inset.set_xticklabels(["SG", "小波", "Kalman"], fontsize=7)
ax_inset.set_ylabel("SNR↑(dB)", fontsize=7)
ax_inset.set_ylim(bottom=0)

# ── (1,2-3) 3D 地质模型 — 电阻率剖面 ────────────────────────────────
import pyvista
pyvista.OFF_SCREEN = True
from discretize import TensorMesh

mesh = TensorMesh(
    [model_data["mesh_hx"], model_data["mesh_hy"], model_data["mesh_hz"]],
    origin=model_data["mesh_origin"],
)
resistivity = model_data["resistivity"]
lithology = model_data["lithology"]
vtk_obj = mesh.to_vtk(models={"resistivity": resistivity, "lithology": lithology.astype(float)})

# Y=400 剖面
slice_y = vtk_obj.slice(normal="y", origin=(0, 400, 0))
active_sy = np.asarray(slice_y["lithology"]) >= 0
if np.any(active_sy):
    sy = slice_y.extract_cells(np.where(active_sy)[0])
    pts = np.asarray(sy.cell_centers().points)
    rho_vals = np.asarray(sy["resistivity"])

    ax12 = fig.add_subplot(gs[1, 2:])
    scatter = ax12.scatter(
        pts[:, 0], pts[:, 2], c=np.log10(rho_vals), cmap="jet_r",
        s=8, edgecolors="none", vmin=0, vmax=4,
    )
    cb = fig.colorbar(scatter, ax=ax12, label="log₁₀(ρ) [Ω·m]", shrink=0.8)

    # 叠加钻孔轨迹
    for name, traj in boreholes.items():
        ax12.plot(traj[:, 0], traj[:, 2], lw=1.5, label=name)

    ax12.set_xlabel("X (m)")
    ax12.set_ylabel("Z (m)")
    ax12.set_title("⑦ Y=400m 剖面 — 电阻率 + 钻孔轨迹（2.5D 视图）", fontweight="bold")
    ax12.legend(fontsize=8, loc="lower right")
    ax12.set_aspect("equal")

# ── (2,0-1) 3D 渲染：矿体 + 钻孔 + 地形 ─────────────────────────────
ax20 = fig.add_subplot(gs[2, 0:2])

plotter = pyvista.Plotter(off_screen=True, window_size=(900, 600))
plotter.set_background("#f7f9fc")

# 矿体
active_cells = np.where(np.asarray(vtk_obj["lithology"]) >= 0)[0]
sub = vtk_obj.extract_cells(active_cells)
ore_cells = np.where((np.asarray(sub["lithology"]) == 2) | (np.asarray(sub["lithology"]) == 3))[0]
if len(ore_cells) > 0:
    ore = sub.extract_cells(ore_cells)
    plotter.add_mesh(ore, scalars="resistivity", cmap="jet_r", log_scale=True, show_edges=True, opacity=1.0)

# 覆盖层
over_cells = np.where(np.asarray(sub["lithology"]) == 1)[0]
if len(over_cells) > 0:
    over = sub.extract_cells(over_cells)
    plotter.add_mesh(over, color="#d4a574", opacity=0.1)

# 地形面
cc = mesh.cell_centers
x_u = np.unique(cc[:, 0])
y_u = np.unique(cc[:, 1])
xg, yg = np.meshgrid(x_u, y_u)
zg = 25.0 * np.sin(2 * np.pi * xg / 800.0) * np.cos(2 * np.pi * yg / 600.0) + 10.0 * np.sin(2 * np.pi * xg / 300.0 + 0.5)
topo_grid = pyvista.StructuredGrid(xg, yg, zg)
plotter.add_mesh(topo_grid, cmap="terrain", opacity=0.3)

# 钻孔
BH_COLORS = {"BH-01": "#2196F3", "BH-02": "#4CAF50", "BH-03": "#F44336"}
for name, traj in boreholes.items():
    line = pyvista.Spline(traj, n_points=150)
    tube = line.tube(radius=5.0)
    plotter.add_mesh(tube, color=BH_COLORS[name], label=name)
    plotter.add_point_labels([traj[0]], [name], font_size=14, text_color="black",
                             bold=True, shape_opacity=0.5)

plotter.add_legend(size=(0.12, 0.12))
plotter.add_axes()
plotter.camera.azimuth = 225
plotter.camera.elevation = 25
img_3d = plotter.screenshot(return_img=True)
plotter.close()

ax20.imshow(img_3d)
ax20.set_axis_off()
ax20.set_title("⑧ 三维地质模型（矿体 + 钻孔 + 地形）", fontweight="bold")

# ── (2,2-3) 3D 渲染：双剖面交叉 ──────────────────────────────────────
ax21 = fig.add_subplot(gs[2, 2:])

plotter2 = pyvista.Plotter(off_screen=True, window_size=(900, 600))
plotter2.set_background("#f7f9fc")

sy_full = vtk_obj.slice(normal="y", origin=(0, 400, 0))
sx_full = vtk_obj.slice(normal="x", origin=(300, 0, 0))

for sl in [sy_full, sx_full]:
    active_s = np.asarray(sl["lithology"]) >= 0
    if np.any(active_s):
        s = sl.extract_cells(np.where(active_s)[0])
        plotter2.add_mesh(s, scalars="resistivity", cmap="jet_r", log_scale=True)

for name, traj in boreholes.items():
    line = pyvista.Spline(traj, n_points=100)
    plotter2.add_mesh(line, color=BH_COLORS[name], line_width=4)

plotter2.add_axes()
plotter2.camera.azimuth = 210
plotter2.camera.elevation = 20
img_slice = plotter2.screenshot(return_img=True)
plotter2.close()

ax21.imshow(img_slice)
ax21.set_axis_off()
ax21.set_title("⑨ 正交剖面交叉（Y=400m + X=300m）+ 钻孔穿越", fontweight="bold")

# ── 页脚 ──────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.005,
    "井中探测数据处理与反演子系统 v0.3 | "
    "正演: empymod (Werthmuller 2017) | 反演: Occam L-BFGS-B | "
    "网格: discretize TensorMesh | 可视化: matplotlib + PyVista",
    ha="center", fontsize=8, color="#999",
)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[完成] 综合图板: {OUT_PNG}")
print(f"       大小: {OUT_PNG.stat().st_size / 1024:.0f} KB")
print("=" * 72)
