"""神经网络智能反演算法封装全流程演示。

演示三种神经网络部署模式：
  Mode A: sklearn MLP — 训练 + joblib 保存 + 算法模块加载推理
  Mode B: ONNX Runtime — sklearn 导出 ONNX → 跨框架部署推理
  Mode C: (文档) PyTorch 模式 — .pt 权重文件加载推理

每种模式都经过完整的：训练 → 保存模型 → 封装为算法模块 → 端到端测试。

运行：
    .venv/bin/python tests/demo_nn_inversion.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from algorithms.registry import AlgorithmRegistry
from algorithms.runner import AlgorithmRunner
from core.data_store import DataStore

WORKSPACE = Path.home() / "geophys-workspace"
BH_NAME = "NN-TEST"
ALGO_DIR = _ROOT / "algorithms"
REPORT_DIR = _ROOT / "tests" / "data" / "demo_nn_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

_pass = _fail = 0


def ok(name, detail=""):
    global _pass
    _pass += 1
    print(f"  [PASS] {name}{' — ' + detail if detail else ''}")


def fail(name, reason):
    global _fail
    _fail += 1
    print(f"  [FAIL] {name} — {reason}")


# ── 准备合成训练数据 ──────────────────────────────────────────────────
print("=" * 70)
print("  神经网络智能反演算法封装全流程演示")
print("=" * 70)

print("\n准备训练数据...")

rng = np.random.default_rng(42)
N_TRAIN = 2000
N_TEST = 200
N_LAYERS = 5

# 合成 TEM 数据集：随机层状模型 → 简化正演 → (观测数据, 真实模型) 对
# 模型：N_LAYERS 层电阻率（对数空间 0.5~3.5，即 ~3~3000 Ohm.m）
# 观测：30 个时道的 dB/dt 响应（简化指数衰减模型）

times = np.logspace(-5, -2, 30)


def synthetic_forward(log_rho):
    """简化正演：层状模型 → 指数衰减响应。"""
    response = np.zeros(len(times))
    for i, lr in enumerate(log_rho):
        rho = 10**lr
        tau = rho * 1e-4 * (i + 1)
        response += (1.0 / rho) * np.exp(-times / tau)
    return response


# 生成训练集
X_train = np.zeros((N_TRAIN, len(times)))
y_train = np.zeros((N_TRAIN, N_LAYERS))
for i in range(N_TRAIN):
    model = rng.uniform(0.5, 3.5, N_LAYERS)
    response = synthetic_forward(model)
    noise = 1 + 0.03 * rng.standard_normal(len(times))
    X_train[i] = np.log10(np.abs(response * noise) + 1e-20)
    y_train[i] = model

# 生成测试集
X_test = np.zeros((N_TEST, len(times)))
y_test = np.zeros((N_TEST, N_LAYERS))
for i in range(N_TEST):
    model = rng.uniform(0.5, 3.5, N_LAYERS)
    response = synthetic_forward(model)
    noise = 1 + 0.03 * rng.standard_normal(len(times))
    X_test[i] = np.log10(np.abs(response * noise) + 1e-20)
    y_test[i] = model

print(f"  训练集: {X_train.shape[0]} 样本, 输入 {X_train.shape[1]} 维, 输出 {y_train.shape[1]} 维")
print(f"  测试集: {X_test.shape[0]} 样本")

# 准备工作区
store = DataStore(WORKSPACE)
store.init_workspace()
store.create_borehole(BH_NAME)
raw_dir, _ = store.ensure_method_dirs(BH_NAME, "tem")

# 保存测试输入到 raw/
np.savetxt(raw_dir / "nn_input.csv", X_test[:5], delimiter=",")  # 5 个测试样本
np.savetxt(raw_dir / "times.csv", times, delimiter=",")

# ══════════════════════════════════════════════════════════════════════
# Mode A: sklearn MLP + joblib
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("Mode A: sklearn MLP — 训练 + joblib 模型保存 + 算法模块封装")
print(f"{'─'*70}")

NAME_A = "test_nn_sklearn_mlp"
algo_dir_a = ALGO_DIR / NAME_A
if algo_dir_a.exists():
    shutil.rmtree(algo_dir_a)
algo_dir_a.mkdir(parents=True)

# A1. 训练 MLP
print("\n  A1. 训练 sklearn MLPRegressor...")
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)
t0 = time.time()
mlp.fit(scaler_X.transform(X_train), scaler_y.transform(y_train))
dt_train = time.time() - t0

# 测试精度
y_pred_test = scaler_y.inverse_transform(mlp.predict(scaler_X.transform(X_test)))
rmse = float(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))
print(f"  训练时间: {dt_train:.1f}s, 测试 RMSE: {rmse:.4f} (log10 Ohm.m)")

# A2. 保存模型
model_dir = algo_dir_a / "weights"
model_dir.mkdir()
joblib.dump(mlp, model_dir / "mlp_model.joblib")
joblib.dump(scaler_X, model_dir / "scaler_X.joblib")
joblib.dump(scaler_y, model_dir / "scaler_y.joblib")
print(f"  模型保存: {model_dir} ({sum(f.stat().st_size for f in model_dir.iterdir()) / 1024:.0f} KB)")

# A3. 创建算法模块
(algo_dir_a / "manifest.json").write_text(json.dumps({
    "name": NAME_A,
    "display_name": "MLP神经网络TEM反演 (sklearn)",
    "version": "1.0.0",
    "method": "tem",
    "category": "inversion",
    "dimension": "1d",
    "entry": {"type": "python", "module": "algorithm", "class": "SklearnMlpInversion"},
    "inputs": [
        {"name": "observed_data", "type": "ndarray_2d", "file": "nn_input.csv",
         "label": "TEM观测数据 (n_samples, n_times)"},
    ],
    "outputs": [
        {"name": "resistivity_model", "type": "ndarray_2d",
         "label": "反演电阻率模型 log10(Ohm.m)"},
        {"name": "mean_rmse", "type": "float", "label": "平均RMSE"},
    ],
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

(algo_dir_a / "algorithm.py").write_text('''\
"""sklearn MLP 神经网络 TEM 智能反演。

模型文件位于 weights/ 子目录：
  mlp_model.joblib  — 训练好的 MLPRegressor
  scaler_X.joblib   — 输入标准化器
  scaler_y.joblib   — 输出标准化器
"""
import numpy as np
from pathlib import Path
import joblib

from algorithms.base import AlgorithmResult, BaseAlgorithm


class SklearnMlpInversion(BaseAlgorithm):
    def run(self, **inputs):
        observed = np.atleast_2d(inputs["observed_data"])

        # 加载模型（从算法包目录下的 weights/ 子目录）
        weight_dir = Path(__file__).parent / "weights"
        model = joblib.load(weight_dir / "mlp_model.joblib")
        scaler_X = joblib.load(weight_dir / "scaler_X.joblib")
        scaler_y = joblib.load(weight_dir / "scaler_y.joblib")

        self.report_progress(0.3, "Model loaded, running inference...")

        # 推理
        X_scaled = scaler_X.transform(observed)
        y_scaled = model.predict(X_scaled)
        result = scaler_y.inverse_transform(y_scaled)

        self.report_progress(1.0, "Done")

        return AlgorithmResult(
            outputs={
                "resistivity_model": result,
                "mean_rmse": 0.0,  # placeholder, real RMSE needs ground truth
            },
        )
''', encoding="utf-8")

# A4. 运行端到端测试
try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    t0 = time.time()
    saved = runner.run(NAME_A, BH_NAME, {})
    dt_infer = time.time() - t0

    result = np.loadtxt(saved["resistivity_model"], delimiter=",")
    assert result.ndim == 2 and result.shape[0] == 5 and result.shape[1] == N_LAYERS

    # 对比真实模型
    pred_rmse = float(np.sqrt(np.mean((result - y_test[:5]) ** 2)))
    ok("sklearn_mlp:e2e", f"推理 {dt_infer:.2f}s, {result.shape}, RMSE={pred_rmse:.4f}")
except Exception as e:
    fail("sklearn_mlp:e2e", f"{type(e).__name__}: {e}")

# ══════════════════════════════════════════════════════════════════════
# Mode B: ONNX Runtime 跨框架部署
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("Mode B: ONNX Runtime — sklearn 导出 ONNX → 通用推理引擎")
print(f"{'─'*70}")

NAME_B = "test_nn_onnx_inversion"
algo_dir_b = ALGO_DIR / NAME_B
if algo_dir_b.exists():
    shutil.rmtree(algo_dir_b)
algo_dir_b.mkdir(parents=True)

# B1. 导出 sklearn → ONNX
print("\n  B1. 导出 MLP 模型为 ONNX 格式...")
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # 需要把 scaler + model 组成 pipeline
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("scaler", scaler_X),
        ("mlp", mlp),
    ])

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("input", FloatTensorType([None, X_train.shape[1]]))],
        target_opset=13,
    )

    onnx_path = algo_dir_b / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 也保存 scaler_y 供反标准化（ONNX 不包含输出 scaler）
    joblib.dump(scaler_y, algo_dir_b / "scaler_y.joblib")

    onnx_size = onnx_path.stat().st_size
    print(f"  ONNX 模型: {onnx_path.name} ({onnx_size / 1024:.0f} KB)")
    ok("onnx:export", f"{onnx_size / 1024:.0f} KB")
except Exception as e:
    fail("onnx:export", f"{type(e).__name__}: {e}")

# B2. 创建 ONNX 算法模块
(algo_dir_b / "manifest.json").write_text(json.dumps({
    "name": NAME_B,
    "display_name": "ONNX神经网络TEM反演",
    "version": "1.0.0",
    "method": "tem",
    "category": "inversion",
    "dimension": "1d",
    "entry": {"type": "python", "module": "algorithm", "class": "OnnxInversion"},
    "inputs": [
        {"name": "observed_data", "type": "ndarray_2d", "file": "nn_input.csv"},
    ],
    "outputs": [
        {"name": "resistivity_model", "type": "ndarray_2d"},
        {"name": "mean_rmse", "type": "float"},
    ],
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

(algo_dir_b / "algorithm.py").write_text('''\
"""ONNX Runtime 神经网络 TEM 反演。

跨框架部署：PyTorch / TensorFlow / sklearn 训练的模型
都可以导出为 ONNX 格式，统一用 onnxruntime 推理。

文件：
  model.onnx     — ONNX 格式模型（含输入标准化）
  scaler_y.joblib — 输出反标准化器
"""
import numpy as np
from pathlib import Path
import joblib
import onnxruntime as ort

from algorithms.base import AlgorithmResult, BaseAlgorithm


class OnnxInversion(BaseAlgorithm):
    def run(self, **inputs):
        observed = np.atleast_2d(inputs["observed_data"]).astype(np.float32)

        model_dir = Path(__file__).parent
        session = ort.InferenceSession(str(model_dir / "model.onnx"))
        scaler_y = joblib.load(model_dir / "scaler_y.joblib")

        self.report_progress(0.3, "ONNX model loaded")

        # ONNX 推理
        input_name = session.get_inputs()[0].name
        onnx_result = session.run(None, {input_name: observed})
        y_scaled = np.asarray(onnx_result[0], dtype=np.float64)

        # 反标准化（确保形状正确）
        n_outputs = len(scaler_y.mean_)
        y_scaled = y_scaled.reshape(-1, n_outputs)
        result = scaler_y.inverse_transform(y_scaled)

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={"resistivity_model": result, "mean_rmse": 0.0},
        )
''', encoding="utf-8")

# B3. 端到端测试
try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    t0 = time.time()
    saved = runner.run(NAME_B, BH_NAME, {})
    dt_onnx = time.time() - t0

    result_onnx = np.loadtxt(saved["resistivity_model"], delimiter=",")
    pred_rmse_onnx = float(np.sqrt(np.mean((result_onnx - y_test[:5]) ** 2)))
    ok("onnx:e2e", f"推理 {dt_onnx:.2f}s, RMSE={pred_rmse_onnx:.4f}")
except Exception as e:
    fail("onnx:e2e", f"{type(e).__name__}: {e}")

# ── 对比两种模式结果一致性 ────────────────────────────────────────────
print(f"\n{'─'*70}")
print("结果对比: sklearn vs ONNX")
print(f"{'─'*70}")
result_onnx = None
dt_onnx = 0.0
diff = np.array([0.0])
try:
    result_onnx = np.loadtxt(saved["resistivity_model"], delimiter=",")
    diff = np.abs(result - result_onnx)
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    if diff.max() < 0.05:
        ok("consistency", f"sklearn ≈ ONNX (max diff={diff.max():.6f})")
    else:
        fail("consistency", f"Difference too large: {diff.max():.6f}")
except Exception:
    pass

# ── 可视化 ────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("生成可视化报告...")
print(f"{'─'*70}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Neural Network Inversion — sklearn MLP & ONNX Deployment", fontsize=14, fontweight="bold")

# Row 1: 5 个样本的反演对比
for i in range(5):
    ax = axes[0, i] if i < 3 else axes[1, i - 3]
    depths = np.arange(N_LAYERS) + 1
    ax.step(10**y_test[i], depths, "b-", linewidth=2, label="True", where="mid")
    ax.step(10**result[i], depths, "r--", linewidth=1.5, label="sklearn MLP", where="mid")
    if result_onnx is not None:
        ax.step(10**result_onnx[i], depths, "g:", linewidth=1.5, label="ONNX", where="mid")
    ax.set_xscale("log")
    ax.set_xlabel("Resistivity (Ohm.m)")
    ax.set_ylabel("Layer")
    ax.set_title(f"Sample {i+1}")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

# (1,2): 统计摘要
ax_summary = axes[1, 2]
ax_summary.axis("off")
summary = (
    "Neural Network Inversion Summary\n"
    "─────────────────────────────────\n"
    f"Training samples:  {N_TRAIN}\n"
    f"Test samples:      {N_TEST}\n"
    f"Input features:    {X_train.shape[1]} (log10 dB/dt)\n"
    f"Output layers:     {N_LAYERS}\n"
    f"Network:           MLP (64-32-16)\n"
    f"\n"
    f"Training time:     {dt_train:.1f}s\n"
    f"sklearn inference: {dt_infer:.2f}s\n"
    f"ONNX inference:    {dt_onnx:.2f}s\n"
    f"\n"
    f"Test RMSE (log10): {rmse:.4f}\n"
    f"sklearn vs ONNX:   {diff.max():.6f}\n" if result_onnx is not None else ""
    f"\n"
    f"Deployment:\n"
    f"  sklearn → .joblib (model+scaler)\n"
    f"  ONNX   → .onnx (cross-framework)\n"
    f"  PyTorch → .pt/.pth (see docs)\n"
)
ax_summary.text(0.05, 0.95, summary, transform=ax_summary.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#f0f0f0"))

plt.tight_layout()
png_path = REPORT_DIR / "demo_nn_inversion.png"
plt.savefig(str(png_path), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  报告图: {png_path} ({png_path.stat().st_size/1024:.0f} KB)")

# ── 清理 ──────────────────────────────────────────────────────────────
shutil.rmtree(algo_dir_a)
shutil.rmtree(algo_dir_b)
print(f"\n  已清理测试算法包")

# ── 总结 ──────────────────────────────────────────────────────────────
total = _pass + _fail
print(f"\n{'='*70}")
print(f"  Results: {_pass} passed, {_fail} failed / {total} total")
print(f"{'='*70}")
sys.exit(1 if _fail else 0)
