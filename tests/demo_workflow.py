"""端到端演示：LAS 导入 → 多曲线滤波 → matplotlib 可视化。

运行方式：
    uv run python tests/demo_workflow.py
"""
import math
import shutil
import sys
from pathlib import Path

import lasio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from core.data_store import DataStore
from core.font_utils import setup_matplotlib_cjk

setup_matplotlib_cjk()

# ── 0. 工作区与钻孔 ──────────────────────────────────────────────────
WORKSPACE = Path.home() / "geophys-workspace"
BH_NAME   = "DEMO-6038187"
LAS_FILE  = _ROOT / "tests" / "data" / "6038187_v1.2_short.las"

store = DataStore(WORKSPACE)
store.init_workspace()
store.create_borehole(BH_NAME)

# ── 1. 导入 LAS ──────────────────────────────────────────────────────
from core.importer import DataImporter
importer = DataImporter(WORKSPACE)
imported = importer.import_file(LAS_FILE, BH_NAME, method="logging")
print(f"[1] 导入曲线：{imported}")

# ── 2. 读取数据 ───────────────────────────────────────────────────────
las   = lasio.read(str(LAS_FILE))
depth = las["DEPT"]

CURVES = {
    "GAMN":  ("自然伽马 (GAPI)",    "tab:green"),
    "NEUT":  ("中子计数率 (CPS)",   "tab:purple"),
    "CALI":  ("井径 (MM)",          "tab:brown"),
    "DNEAR": ("近探测密度 (g/cm³)", "tab:orange"),
    "COND":  ("地层电导率 (mS/m)",  "tab:red"),
}

# ── 3. Savitzky-Golay 滤波 ───────────────────────────────────────────
WINDOW, POLY = 11, 3
filtered = {}
snr_db   = {}
for name in CURVES:
    raw = las[name]
    flt = savgol_filter(raw, WINDOW, POLY)
    filtered[name] = flt
    resid_var  = float(((raw - flt) ** 2).mean())
    orig_var   = float((raw ** 2).mean())
    snr_db[name] = 10.0 * math.log10(max(orig_var / resid_var, 1e-9)) if resid_var > 1e-12 else 0.0

print("[2] 滤波完成：" + "  ".join(f"{k}→{v:.1f}dB" for k, v in snr_db.items()))

# ── 4. 可视化 ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12), dpi=130)
fig.patch.set_facecolor("#f7f9fc")

# 标题
fig.suptitle(
    f"井中探测数据处理与可视化演示\n钻孔：{BH_NAME}   深度：{depth[0]:.1f}–{depth[-1]:.1f} m   "
    f"滤波：Savitzky-Golay (window={WINDOW}, poly={POLY})",
    fontsize=13, fontweight="bold", y=0.98, color="#1a1a2e",
)

# ─ 上半部分：5 个测井道（原始 + 滤波叠显） ────────────────────────
n_tracks = len(CURVES)
gs_top = gridspec.GridSpec(1, n_tracks, figure=fig,
                           left=0.04, right=0.98, top=0.88, bottom=0.46,
                           wspace=0.06)

curve_names = list(CURVES.keys())
for col, name in enumerate(curve_names):
    ax = fig.add_subplot(gs_top[0, col])
    ax.set_facecolor("#fefefe")

    label, color = CURVES[name]
    raw = las[name]
    flt = filtered[name]

    # 原始曲线（浅色）
    ax.plot(raw, depth, color=color, alpha=0.35, linewidth=0.9, label="原始")
    # 滤波曲线（深色）
    ax.plot(flt, depth, color=color, linewidth=1.8, label="滤波")

    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="major", linestyle="--", linewidth=0.4, color="#cccccc")
    ax.grid(which="minor", linestyle=":",  linewidth=0.2, color="#e0e0e0")

    ax.set_title(label, fontsize=7.5, pad=3)
    ax.tick_params(axis="both", labelsize=7)

    if col == 0:
        ax.set_ylabel("深度 (m)", fontsize=8)
    else:
        ax.set_yticklabels([])

    # SNR 标注
    ax.text(0.97, 0.03, f"SNR↑{snr_db[name]:.1f}dB",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))

    # 图例（仅第一道显示）
    if col == 0:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.8)

# ─ 下半部分：残差对比 + SNR 柱状图 ──────────────────────────────────
gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.07, right=0.98, top=0.40, bottom=0.07,
                           wspace=0.25)

# 左：残差（GAMN）
ax_res = fig.add_subplot(gs_bot[0, 0])
ax_res.set_facecolor("#fefefe")
gamn_raw = las["GAMN"]
gamn_flt = filtered["GAMN"]
residual = gamn_raw - gamn_flt
ax_res.fill_betweenx(depth, residual, 0,
                     where=(residual > 0), color="tab:red",   alpha=0.5, label="正残差")
ax_res.fill_betweenx(depth, residual, 0,
                     where=(residual < 0), color="tab:blue",  alpha=0.5, label="负残差")
ax_res.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax_res.invert_yaxis()
ax_res.set_xlabel("滤波残差 (GAPI)", fontsize=9)
ax_res.set_ylabel("深度 (m)", fontsize=9)
ax_res.set_title("GAMN 滤波残差分析", fontsize=10, fontweight="bold")
ax_res.legend(fontsize=8)
ax_res.grid(linestyle=":", linewidth=0.4, color="#cccccc")
ax_res.tick_params(labelsize=8)

# 右：SNR 改善量柱状图
ax_snr = fig.add_subplot(gs_bot[0, 1])
ax_snr.set_facecolor("#fefefe")
bar_names  = list(snr_db.keys())
bar_values = [snr_db[k] for k in bar_names]
bar_colors = [CURVES[k][1] for k in bar_names]
bars = ax_snr.bar(bar_names, bar_values, color=bar_colors, alpha=0.75,
                  edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, bar_values):
    ax_snr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_snr.axhline(0, color="black", linewidth=0.8)
ax_snr.set_ylabel("信噪比改善量 (dB)", fontsize=9)
ax_snr.set_title("各曲线 SNR 改善量对比", fontsize=10, fontweight="bold")
ax_snr.tick_params(labelsize=9)
ax_snr.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc")
ax_snr.set_ylim(bottom=0)

# 页脚
fig.text(0.5, 0.005,
         "数据来源：lasio 开源测试数据集 (MIT License) | "
         "算法：scipy.signal.savgol_filter | "
         "封装框架：井中探测数据处理与反演子系统",
         ha="center", fontsize=7, color="#888888")

out = _ROOT / "tests" / "data" / "demo_workflow.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[3] 可视化保存到：{out}")
