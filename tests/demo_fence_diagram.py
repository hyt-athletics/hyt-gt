"""Petrel 式 Fence Diagram — 沿钻孔连线的 2.5D 电阻率剖面。

剖面路径：BH-01 → BH-03 终孔 → BH-02
  - 剖面1：BH-01(300,400) → BH-03 终孔点 — 穿越板状矿体
  - 剖面2：BH-03 终孔点 → BH-02(650,600) — 穿越脉状矿体

每条剖面渲染：
  1. 电阻率伪彩色（log 色标，jet_r）
  2. 地层界线叠加（覆盖层/风化带/新鲜基岩 虚线）
  3. 钻孔轨迹投影（孔名 + 深度刻度）
  4. 矿体轮廓（红色实线勾边）

运行方式：
    .venv/bin/python tests/demo_fence_diagram.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from scipy.interpolate import griddata

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

DATA_DIR = _ROOT / "tests" / "data" / "geo_model_3d"
OUT_PNG = DATA_DIR / "demo_fence_diagram.png"

# ── 加载模型 ──────────────────────────────────────────────────────────
npz = np.load(DATA_DIR / "geo_model_3d.npz")
resistivity = npz["resistivity"]
lithology = npz["lithology"]
hx, hy, hz = npz["mesh_hx"], npz["mesh_hy"], npz["mesh_hz"]
origin = npz["mesh_origin"]

from discretize import TensorMesh

mesh = TensorMesh([hx, hy, hz], origin=origin)
cc = mesh.cell_centers
x_cc, y_cc, z_cc = cc[:, 0], cc[:, 1], cc[:, 2]

meta = json.loads((DATA_DIR / "geo_model_meta.json").read_text(encoding="utf-8"))

# 钻孔轨迹
bh01_traj = np.loadtxt(DATA_DIR / "BH-01_trajectory.csv", delimiter=",", skiprows=1)
bh02_traj = np.loadtxt(DATA_DIR / "BH-02_trajectory.csv", delimiter=",", skiprows=1)
bh03_traj = np.loadtxt(DATA_DIR / "BH-03_trajectory.csv", delimiter=",", skiprows=1)

bh01_collar = bh01_traj[0]
bh02_collar = bh02_traj[0]
bh03_end = bh03_traj[-1]

print(f"BH-01 collar: ({bh01_collar[0]:.0f}, {bh01_collar[1]:.0f})")
print(f"BH-03 end:    ({bh03_end[0]:.0f}, {bh03_end[1]:.0f})")
print(f"BH-02 collar: ({bh02_collar[0]:.0f}, {bh02_collar[1]:.0f})")


# ── 沿任意垂直剖面提取数据 ────────────────────────────────────────────
def extract_fence_section(p0_xy, p1_xy, swath_half=15.0, z_range=(-800, 50)):
    """从网格中沿水平线段 p0→p1 提取垂直剖面数据。

    Args:
        p0_xy: 起点 (x, y)
        p1_xy: 终点 (x, y)
        swath_half: 剖面半宽（垂直于剖面方向的采样宽度），米
        z_range: 深度范围

    Returns:
        profile_dist, z, rho, litho — 投影到剖面上的坐标和属性
    """
    dx = p1_xy[0] - p0_xy[0]
    dy = p1_xy[1] - p0_xy[1]
    length = np.sqrt(dx**2 + dy**2)
    # 剖面方向单位向量
    ux, uy = dx / length, dy / length
    # 法线方向
    nx, ny = -uy, ux

    # 每个 cell center 投影到剖面
    rx = x_cc - p0_xy[0]
    ry = y_cc - p0_xy[1]
    along = rx * ux + ry * uy     # 沿剖面距离
    across = rx * nx + ry * ny    # 垂直剖面距离

    # 选取在 swath 内、深度范围内、非空气的 cells
    mask = (
        (np.abs(across) <= swath_half)
        & (along >= -10)
        & (along <= length + 10)
        & (z_cc >= z_range[0])
        & (z_cc <= z_range[1])
        & (lithology >= 0)
    )

    return along[mask], z_cc[mask], resistivity[mask], lithology[mask], length


def project_borehole_to_section(traj, p0_xy, p1_xy):
    """将钻孔轨迹投影到剖面坐标系。"""
    dx = p1_xy[0] - p0_xy[0]
    dy = p1_xy[1] - p0_xy[1]
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / length, dy / length

    rx = traj[:, 0] - p0_xy[0]
    ry = traj[:, 1] - p0_xy[1]
    along = rx * ux + ry * uy
    return along, traj[:, 2]


# ── 绘制单条剖面 ──────────────────────────────────────────────────────
def plot_fence_section(
    ax, p0_xy, p1_xy, title, boreholes_on_section, azimuth_label
):
    """绘制单条 fence 剖面。"""
    dist, z, rho, litho, length = extract_fence_section(p0_xy, p1_xy)

    if len(dist) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    # 规则化网格插值（用于平滑的伪彩色图）
    n_dist = 300
    n_z = 200
    di = np.linspace(0, length, n_dist)
    zi = np.linspace(z.min(), z.max(), n_z)
    DI, ZI = np.meshgrid(di, zi)

    # log10 插值避免极端值干扰
    log_rho = np.log10(np.clip(rho, 1.0, 1e5))
    mean_log_rho = np.mean(log_rho)
    RHO_I = griddata((dist, z), log_rho, (DI, ZI), method="linear", fill_value=mean_log_rho)
    LITHO_I = griddata((dist, z), litho.astype(float), (DI, ZI), method="nearest")

    # 电阻率伪彩色（log 色标）
    im = ax.pcolormesh(
        DI, ZI, 10**RHO_I,
        cmap="jet_r", norm=LogNorm(vmin=1, vmax=6000),
        shading="gouraud", rasterized=True,
    )

    # Lithology contours: formation boundaries, ore outlines, alteration halo
    _CONTOUR_SPECS = [
        # (levels, colors, linewidths, linestyles)
        ([0.5, 1.5, 3.5], ["black", "black", "gray"], [1.0, 0.8, 0.6], ["--", "-.", ":"]),
        ([1.5, 2.5], ["red", "darkred"], [2.0, 2.0], ["-", "-"]),
        ([4.5], ["orange"], [1.5], ["--"]),
    ]
    for levels, colors, lws, lstyles in _CONTOUR_SPECS:
        try:
            ax.contour(DI, ZI, LITHO_I, levels=levels,
                       colors=colors, linewidths=lws, linestyles=lstyles)
        except ValueError:
            pass

    # 钻孔轨迹投影
    bh_colors = {"BH-01": "#0066cc", "BH-02": "#00aa44", "BH-03": "#cc0000"}
    for bh_name, traj in boreholes_on_section:
        bd, bz = project_borehole_to_section(traj, p0_xy, p1_xy)
        ax.plot(bd, bz, color=bh_colors.get(bh_name, "white"), linewidth=2.5, zorder=5)
        ax.plot(bd[0], bz[0], "v", color=bh_colors[bh_name], markersize=8, zorder=6)
        ax.annotate(
            bh_name, (bd[0], bz[0]),
            textcoords="offset points", xytext=(5, 8),
            fontsize=9, fontweight="bold", color=bh_colors[bh_name],
            zorder=7,
        )
        # 深度刻度（每 100m）
        depths = np.arange(0, len(traj) * 3, 100)
        for d in depths:
            idx = int(d / 3)
            if idx < len(bd):
                ax.plot(bd[idx], bz[idx], "_", color=bh_colors[bh_name],
                        markersize=6, markeredgewidth=1.5, zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Profile distance (m)  [{azimuth_label}]", fontsize=9)
    ax.set_ylabel("Elevation (m)", fontsize=9)
    ax.set_xlim(0, length)
    # 垂直夸张 2:1，适合地质剖面展示
    ax.set_aspect(0.5)

    return im


# ── 主图 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 14), constrained_layout=True)

# 剖面 1：BH-01 → BH-03 终孔
p0_1 = (bh01_collar[0], bh01_collar[1])
p1_1 = (bh03_end[0], bh03_end[1])
az1 = np.degrees(np.arctan2(p1_1[0] - p0_1[0], p1_1[1] - p0_1[1]))
im1 = plot_fence_section(
    axes[0], p0_1, p1_1,
    "Fence Section 1: BH-01 → BH-03 (Plate Ore)",
    [("BH-01", bh01_traj), ("BH-03", bh03_traj)],
    f"Azimuth N{az1:.0f}E",
)

# 剖面 2：BH-03 终孔 → BH-02
p0_2 = (bh03_end[0], bh03_end[1])
p1_2 = (bh02_collar[0], bh02_collar[1])
az2 = np.degrees(np.arctan2(p1_2[0] - p0_2[0], p1_2[1] - p0_2[1]))
im2 = plot_fence_section(
    axes[1], p0_2, p1_2,
    "Fence Section 2: BH-03 → BH-02 (Vein Ore)",
    [("BH-02", bh02_traj), ("BH-03", bh03_traj)],
    f"Azimuth N{az2:.0f}E",
)

# 共享色标
if im1 is not None:
    cbar = fig.colorbar(im1, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Resistivity (Ohm.m)", fontsize=10)

# 图例
legend_elements = [
    Patch(facecolor="none", edgecolor="black", linestyle="--", label="Overburden base"),
    Patch(facecolor="none", edgecolor="black", linestyle="-.", label="Weathered zone base"),
    Patch(facecolor="none", edgecolor="red", linewidth=2, label="Ore body outline"),
    Patch(facecolor="none", edgecolor="orange", linestyle="--", label="Alteration halo"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
           frameon=True, fancybox=True)

fig.suptitle(
    "Petrel-style Fence Diagram — Borehole Connecting Sections",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Fence diagram saved: {OUT_PNG}")
print(f"  Section 1: ({p0_1[0]:.0f},{p0_1[1]:.0f}) → ({p1_1[0]:.0f},{p1_1[1]:.0f}), azimuth N{az1:.0f}E")
print(f"  Section 2: ({p0_2[0]:.0f},{p0_2[1]:.0f}) → ({p1_2[0]:.0f},{p1_2[1]:.0f}), azimuth N{az2:.0f}E")
