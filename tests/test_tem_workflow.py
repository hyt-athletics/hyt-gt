"""TEM 正演 + 一维 Occam 反演工作流测试。

工作流：
  1. 构建三层合成地球模型（低阻-高阻-低阻，以模拟矿化层）
  2. empymod 正演 → 获得合成 dB/dt 时间序列
  3. 添加 5% 高斯噪声模拟实测噪声
  4. TEM 一维 Occam 反演 → 恢复电阻率-深度模型
  5. 可视化对比：真实模型 vs 反演模型，视电阻率曲线

运行方式：
    uv run python tests/test_tem_workflow.py
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
from algorithms.registry import AlgorithmRegistry
from algorithms.runner import AlgorithmRunner

setup_matplotlib_cjk()

# ── 配置 ──────────────────────────────────────────────────────────────
WORKSPACE  = Path.home() / "geophys-workspace"
BH_NAME    = "TEM-SYNTH-001"
ALGO_DIR   = _ROOT / "algorithms"
OUT_PNG    = _ROOT / "tests" / "data" / "tem_workflow.png"
LOOP_R     = 50.0    # 回线半径 (m)

# ── 1. 合成地球模型（三层 + 空气层） ──────────────────────────────────
# 层定义（从地面往下）：
#   层1（地表覆盖层）：ρ=200 Ω·m，厚 50 m
#   层2（导电矿化层）：ρ=10  Ω·m，厚 100 m
#   层3（高阻基底）  ：ρ=1000 Ω·m，半空间
TRUE_RESISTIVITY = np.array([2e14, 200.0, 10.0, 1000.0])  # 含空气层
TRUE_DEPTHS      = np.array([50.0, 150.0])                  # 层界面深度 (m)

print("[0] 合成模型：", dict(zip(["覆盖层", "矿化层", "基底"],
                               TRUE_RESISTIVITY[1:].tolist())))

# ── 2. 初始化工作区 ───────────────────────────────────────────────────
store = DataStore(WORKSPACE)
store.init_workspace()
store.create_borehole(BH_NAME)

raw_dir, _ = store.ensure_method_dirs(BH_NAME, "tem")

# 保存合成模型到 raw/ 目录（供算法读取）
np.savetxt(raw_dir / "resistivity.csv", TRUE_RESISTIVITY, delimiter=",")
np.savetxt(raw_dir / "depths.csv",      TRUE_DEPTHS,      delimiter=",")
print(f"[1] 合成模型保存→{raw_dir}")

# ── 3. 正演：计算合成 TEM 响应 ────────────────────────────────────────
registry = AlgorithmRegistry(ALGO_DIR)
registry.scan()
runner = AlgorithmRunner(registry, WORKSPACE)

fwd_params = {
    "loop_radius": str(LOOP_R),
    "t_min": "-5.0",
    "t_max": "-2.0",
    "n_times": "30",
}
saved_fwd = runner.run("empymod_tem_forward", BH_NAME, fwd_params)
emf_clean = np.loadtxt(str(saved_fwd["emf"]),   delimiter=",")
times     = np.loadtxt(str(saved_fwd["times"]),  delimiter=",")
rho_app   = np.loadtxt(str(saved_fwd["rho_app"]), delimiter=",")
print(f"[2] 正演完成：{len(times)} 个时道，EMF 范围 {emf_clean.min():.2e}–{emf_clean.max():.2e}")

# ── 4. 添加噪声（模拟野外观测） ───────────────────────────────────────
rng = np.random.default_rng(42)
noise_level = 0.05   # 5% 相对高斯噪声
emf_noisy = emf_clean * (1.0 + noise_level * rng.standard_normal(len(emf_clean)))
emf_noisy = np.abs(emf_noisy)   # TEM 观测值为正

# 覆盖 raw/ 中的 emf 文件，模拟含噪声的实测数据
np.savetxt(raw_dir / "emf.csv",   emf_noisy.reshape(1, -1), delimiter=",")
np.savetxt(raw_dir / "times.csv", times,                    delimiter=",")

# ── 5. 反演 ───────────────────────────────────────────────────────────
inv_params = {
    "n_layers":  "20",
    "lambda_":   "0.001",
    "depth_max": "500.0",
    "loop_radius": str(LOOP_R),
}
saved_inv = runner.run("tem_1d_inversion", BH_NAME, inv_params)
rho_inv  = np.loadtxt(str(saved_inv["resistivity"]), delimiter=",")
tops_inv = np.loadtxt(str(saved_inv["layer_tops"]),  delimiter=",")
misfit   = float(Path(saved_inv["misfit"]).read_text(encoding="utf-8"))
print(f"[3] 反演完成：{len(rho_inv)} 层，RMS 拟合误差 = {misfit:.4f}")

# ── 6. 可视化 ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9), dpi=130)
fig.patch.set_facecolor("#f7f9fc")
fig.suptitle(
    f"TEM 一维正演-反演工作流测试（回线半径={LOOP_R}m，5% 噪声）\n"
    f"合成三层模型：200 / 10 / 1000 Ω·m   反演层数：{len(rho_inv)}",
    fontsize=12, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.98,
                       top=0.88, bottom=0.08, wspace=0.30)

# 左：真实模型 vs 反演模型（阶梯图）
ax_mod = fig.add_subplot(gs[0, 0])

def _step_xy(rhos, tops, bottom=500.0):
    """将 (rho, top_depths) 转换为阶梯图的 x/y 数组。"""
    xs, ys = [], []
    for i, rho in enumerate(rhos):
        d0 = tops[i] if i < len(tops) else bottom
        d1 = tops[i + 1] if i + 1 < len(tops) else bottom
        xs += [rho, rho]
        ys += [d0, d1]
    return np.array(xs), np.array(ys)

true_rhos = TRUE_RESISTIVITY[1:]   # [200, 10, 1000]
true_tops = np.concatenate([[0.0], TRUE_DEPTHS])  # [0, 50, 150]
tx, ty = _step_xy(true_rhos, true_tops)
ax_mod.plot(tx, ty, color="tab:red", linewidth=2.5, label="真实模型", zorder=3)

inv_tops = np.concatenate([[0.0], tops_inv[1:]])
ix, iy = _step_xy(rho_inv, tops_inv)
ax_mod.plot(ix, iy, color="tab:blue", linewidth=1.6, linestyle="--",
            label=f"反演结果 RMS={misfit:.3f}", zorder=2)
ax_mod.set_xscale("log")
ax_mod.set_xlabel("电阻率 (Ω·m)", fontsize=9)
ax_mod.set_ylabel("深度 (m)", fontsize=9)
ax_mod.invert_yaxis()
ax_mod.set_title("电阻率模型对比", fontsize=10, fontweight="bold")
ax_mod.legend(fontsize=8)
ax_mod.grid(which="both", linestyle=":", linewidth=0.4, color="#cccccc")
ax_mod.tick_params(labelsize=8)
# 标注矿化层
ax_mod.axhspan(TRUE_DEPTHS[0], TRUE_DEPTHS[1],
               alpha=0.12, color="tab:orange", label="_矿化层")
ax_mod.text(15, (TRUE_DEPTHS[0] + TRUE_DEPTHS[1]) / 2,
            "矿化层", fontsize=8, color="tab:orange", va="center")

# 中：正演 dB/dt 曲线（含噪 vs 无噪）
ax_emf = fig.add_subplot(gs[0, 1])
ax_emf.loglog(times, emf_clean, color="tab:green", linewidth=2.0,
              label="无噪声正演")
ax_emf.loglog(times, emf_noisy, color="tab:red", linewidth=0.8,
              alpha=0.7, marker="o", markersize=3, label="含噪观测（5%）")
ax_emf.set_xlabel("时间 (s)", fontsize=9)
ax_emf.set_ylabel("|dB/dt| (V/A·m²)", fontsize=9)
ax_emf.set_title("TEM 感应 dB/dt 响应", fontsize=10, fontweight="bold")
ax_emf.legend(fontsize=8)
ax_emf.grid(which="both", linestyle=":", linewidth=0.4, color="#cccccc")
ax_emf.tick_params(labelsize=8)

# 右：视电阻率-时间曲线
ax_rho = fig.add_subplot(gs[0, 2])
valid = np.isfinite(rho_app) & (rho_app > 0)
ax_rho.loglog(times[valid], rho_app[valid],
              color="tab:purple", linewidth=2.0, marker="s",
              markersize=4, label="视电阻率 ρₐ(t)")
ax_rho.axhline(10.0, color="tab:orange", linestyle="--", linewidth=1.0,
               label="矿化层 ρ=10 Ω·m")
ax_rho.axhline(200.0, color="tab:green", linestyle="--", linewidth=1.0,
               label="覆盖层 ρ=200 Ω·m")
ax_rho.set_xlabel("时间 (s)", fontsize=9)
ax_rho.set_ylabel("视电阻率 (Ω·m)", fontsize=9)
ax_rho.set_title("视电阻率曲线", fontsize=10, fontweight="bold")
ax_rho.legend(fontsize=8)
ax_rho.grid(which="both", linestyle=":", linewidth=0.4, color="#cccccc")
ax_rho.tick_params(labelsize=8)

fig.text(0.5, 0.005,
         "正演引擎：empymod (Werthmuller 2017, Geophysics) | "
         "反演：Occam 平滑约束 (L-BFGS-B)",
         ha="center", fontsize=7, color="#888888")

plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[4] 可视化保存→{OUT_PNG}")
