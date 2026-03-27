"""测井多算法滤波对比工作流测试。

真实数据：6038187_v1.2_short.las（包含 GR/NEUT/CALI/DNEAR/COND 五条曲线）

工作流：
  1. 导入 LAS → 生成 raw/GR.csv（以及其他曲线 CSV）
  2. 分别对 GR 曲线运行三种滤波算法：
       ├── Savitzky-Golay（window=11, polyorder=3）
       ├── 小波阈值去噪（wavelet=db4, level=4）
       └── Kalman-RTS 平滑（Q=0.01, R=1.0）
  3. 生成对比可视化图表，保存到 tests/data/logging_workflow.png

运行方式：
    uv run python tests/test_logging_workflow.py
"""
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
from core.importer import DataImporter
from algorithms.registry import AlgorithmRegistry
from algorithms.runner import AlgorithmRunner

setup_matplotlib_cjk()

# ── 配置 ──────────────────────────────────────────────────────────────
WORKSPACE = Path.home() / "geophys-workspace"
BH_NAME   = "LOG-MULTIFILTER"
LAS_FILE  = _ROOT / "tests" / "data" / "6038187_v1.2_short.las"
ALGO_DIR  = _ROOT / "algorithms"
OUT_PNG   = _ROOT / "tests" / "data" / "logging_workflow.png"

# ── 1. 初始化工作区 ───────────────────────────────────────────────────
store = DataStore(WORKSPACE)
store.init_workspace()
store.create_borehole(BH_NAME)

# ── 2. 导入 LAS ───────────────────────────────────────────────────────
importer = DataImporter(WORKSPACE)
channels = importer.import_file(LAS_FILE, BH_NAME, method="logging")
print(f"[1] 导入曲线：{channels}")

# ── 3. 读取原始 GR 曲线 ───────────────────────────────────────────────
import shutil
import lasio
las  = lasio.read(str(LAS_FILE))
depth = las["DEPT"]
gr_raw = las["GAMN"]   # GR 曲线（GAPI）

# 算法 manifest 默认读取 GR.csv，将导入的 GAMN.csv 复制为 GR.csv
raw_dir = WORKSPACE / "boreholes" / BH_NAME / "logging" / "raw"
gamn_src = raw_dir / "GAMN.csv"
gr_dst   = raw_dir / "GR.csv"
if gamn_src.exists() and not gr_dst.exists():
    shutil.copy(gamn_src, gr_dst)
    print(f"[1b] 复制 GAMN.csv → GR.csv")

# ── 4. 运行三种滤波算法 ───────────────────────────────────────────────
registry = AlgorithmRegistry(ALGO_DIR)
registry.scan()
runner = AlgorithmRunner(registry, WORKSPACE)

results = {}
for algo, params in [
    ("log_preproc_savgol",  {"window_length": "11", "polyorder": "3"}),
    ("log_preproc_wavelet", {"wavelet": "db4", "level": "4"}),
    ("log_preproc_kalman",  {"process_noise": "0.01", "measurement_noise": "1.0"}),
]:
    saved = runner.run(algo, BH_NAME, params)
    filt  = np.loadtxt(str(saved["curve_filtered"]), delimiter=",")
    snr   = float(Path(saved["snr_improvement"]).read_text(encoding="utf-8"))
    results[algo] = (filt, snr)
    print(f"[2] {algo}: SNR↑{snr:.1f} dB，输出→{saved['curve_filtered'].name}")

# ── 5. 可视化对比 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), dpi=130)
fig.patch.set_facecolor("#f7f9fc")
fig.suptitle(
    f"测井曲线三滤波算法对比（钻孔：{BH_NAME}）\n"
    "深绿=SG  深橙=小波  深蓝=Kalman  浅灰=原始",
    fontsize=12, fontweight="bold", y=0.98,
)

COLORS = {
    "log_preproc_savgol":  ("Savitzky-Golay", "tab:green"),
    "log_preproc_wavelet": ("小波阈值",        "tab:orange"),
    "log_preproc_kalman":  ("Kalman-RTS",      "tab:blue"),
}

gs = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.98,
                       top=0.88, bottom=0.06, wspace=0.28)

# 左：三算法叠显（测井道图）
ax_log = fig.add_subplot(gs[0, 0])
ax_log.plot(gr_raw, depth, color="#cccccc", linewidth=0.8, label="原始 GR", zorder=1)
for algo, (label, color) in COLORS.items():
    filt, snr = results[algo]
    ax_log.plot(filt, depth, color=color, linewidth=1.6,
                label=f"{label}  SNR↑{snr:.1f}dB", zorder=2)
ax_log.invert_yaxis()
ax_log.set_xlabel("GR (GAPI)", fontsize=9)
ax_log.set_ylabel("深度 (m)", fontsize=9)
ax_log.set_title("自然伽马曲线三算法滤波叠显", fontsize=10, fontweight="bold")
ax_log.legend(fontsize=8, loc="upper right")
ax_log.grid(linestyle=":", linewidth=0.4, color="#cccccc")
ax_log.tick_params(labelsize=8)

# 右：残差对比（对每种算法显示 raw−filtered）
ax_res = fig.add_subplot(gs[0, 1])
for algo, (label, color) in COLORS.items():
    filt, _ = results[algo]
    res = gr_raw - filt
    ax_res.plot(res, depth, color=color, linewidth=1.0, alpha=0.85, label=label)
ax_res.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax_res.invert_yaxis()
ax_res.set_xlabel("滤波残差 (GAPI)", fontsize=9)
ax_res.set_ylabel("深度 (m)", fontsize=9)
ax_res.set_title("各算法滤波残差（反映高频噪声去除量）", fontsize=10, fontweight="bold")
ax_res.legend(fontsize=8)
ax_res.grid(linestyle=":", linewidth=0.4, color="#cccccc")
ax_res.tick_params(labelsize=8)

# SNR 柱状图（右下角 inset）
ax_inset = ax_res.inset_axes([0.55, 0.02, 0.42, 0.25])
labels_s = [COLORS[a][0] for a in results]
snrs = [results[a][1] for a in results]
colors_s = [COLORS[a][1] for a in results]
bars = ax_inset.bar(range(3), snrs, color=colors_s, alpha=0.8, edgecolor="white")
for bar, v in zip(bars, snrs):
    ax_inset.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")
ax_inset.set_xticks(range(3))
ax_inset.set_xticklabels(["SG", "小波", "Kalman"], fontsize=7)
ax_inset.set_ylabel("SNR↑(dB)", fontsize=7)
ax_inset.set_title("SNR 改善对比", fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.set_ylim(bottom=0)

fig.text(0.5, 0.005,
         "真实数据来源：lasio 开源测试数据集（MIT License） | "
         "算法：scipy.signal + PyWavelets + 自实现 Kalman-RTS",
         ha="center", fontsize=7, color="#888888")

plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[3] 可视化保存→{OUT_PNG}")
