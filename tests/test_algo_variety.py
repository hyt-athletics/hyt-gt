"""Multi-variety algorithm integration test.

Tests 6 different algorithm packaging patterns to verify the platform
can correctly wrap, call, and save results from diverse algorithm types.

Usage:
    .venv/bin/python tests/test_algo_variety.py
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
BH_NAME = "TEST-WELL"
ALGO_DIR = _ROOT / "algorithms"

_pass = _fail = 0


def ok(name, detail=""):
    global _pass
    _pass += 1
    print(f"  [PASS] {name}{' — ' + detail if detail else ''}")


def fail(name, reason):
    global _fail
    _fail += 1
    print(f"  [FAIL] {name} — {reason}")


def _create_algo(name, manifest_dict, code):
    d = ALGO_DIR / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (d / "algorithm.py").write_text(code, encoding="utf-8")
    return d


def _cleanup(name):
    d = ALGO_DIR / name
    if d.exists():
        shutil.rmtree(d)


# Ensure test data exists
raw_log = WORKSPACE / "boreholes" / BH_NAME / "logging" / "raw"
if not (raw_log / "GR.csv").exists():
    print("ERROR: Run build_test_datasets.py first")
    sys.exit(1)

# Create a second curve for multi-input test
if not (raw_log / "COND.csv").exists():
    rng = np.random.default_rng(99)
    cond = 0.01 + 0.005 * np.sin(np.linspace(0, 20, 500)) + rng.standard_normal(500) * 0.001
    np.savetxt(raw_log / "COND.csv", cond, delimiter=",")

print("=" * 70)
print("  Algorithm Variety Test — 6 Packaging Patterns")
print("=" * 70)

# ══ Type A: Pure math, no dependencies ════════════════════════════════
print(f"\n{'─'*70}")
print("A: Pure math — Hilbert envelope (numpy only)")
print(f"{'─'*70}")

NAME_A = "test_var_envelope"
try:
    _create_algo(NAME_A, {
        "name": NAME_A, "display_name": "包络检测", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "Envelope"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
        ],
        "outputs": [
            {"name": "envelope", "type": "ndarray_1d", "label": "包络线"},
            {"name": "peak_value", "type": "float", "label": "峰值"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class Envelope(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        # Hilbert transform via FFT
        n = len(curve)
        fft = np.fft.fft(curve)
        h = np.zeros(n)
        h[0] = 1
        h[1:(n+1)//2] = 2
        if n % 2 == 0:
            h[n//2] = 1
        analytic = np.fft.ifft(fft * h)
        envelope = np.abs(analytic)
        return AlgorithmResult(
            outputs={"envelope": envelope, "peak_value": float(np.max(envelope))},
        )
''')
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    t0 = time.time()
    saved = runner.run(NAME_A, BH_NAME, {})
    dt = time.time() - t0
    env = np.loadtxt(saved["envelope"], delimiter=",")
    peak = float(Path(saved["peak_value"]).read_text(encoding="utf-8"))
    assert env.ndim == 1 and len(env) > 0
    assert peak > 0
    ok("envelope", f"{dt:.2f}s, peak={peak:.1f}, len={len(env)}")
except Exception as e:
    fail("envelope", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_A)

# ══ Type B: Multi-file input ══════════════════════════════════════════
print(f"\n{'─'*70}")
print("B: Multi-file input — Crossplot (GR + COND)")
print(f"{'─'*70}")

NAME_B = "test_var_crossplot"
try:
    _create_algo(NAME_B, {
        "name": NAME_B, "display_name": "双曲线交会", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "Crossplot"},
        "inputs": [
            {"name": "curve_a", "type": "ndarray_1d", "file": "GR.csv", "label": "GR曲线"},
            {"name": "curve_b", "type": "ndarray_1d", "file": "COND.csv", "label": "COND曲线"},
            {"name": "cutoff", "type": "float", "default": 50.0, "required": False},
        ],
        "outputs": [
            {"name": "ratio", "type": "ndarray_1d"},
            {"name": "correlation", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class Crossplot(BaseAlgorithm):
    def run(self, **inputs):
        a = inputs["curve_a"]
        b = inputs["curve_b"]
        cutoff = float(inputs.get("cutoff", 50.0))
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        safe_b = np.where(np.abs(b) > 1e-12, b, 1e-12)
        ratio = a / safe_b
        mask = a > cutoff
        if mask.sum() > 2:
            corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
        else:
            corr = 0.0
        return AlgorithmResult(outputs={"ratio": ratio, "correlation": corr})
''')
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    saved = runner.run(NAME_B, BH_NAME, {"cutoff": "40"})
    ratio = np.loadtxt(saved["ratio"], delimiter=",")
    corr = float(Path(saved["correlation"]).read_text(encoding="utf-8"))
    assert ratio.ndim == 1
    ok("crossplot", f"ratio len={len(ratio)}, corr={corr:.3f}")
except Exception as e:
    fail("crossplot", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_B)

# ══ Type C: 2D output ═════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("C: 2D output — Spectrogram (1D → 2D)")
print(f"{'─'*70}")

NAME_C = "test_var_spectrogram"
try:
    _create_algo(NAME_C, {
        "name": NAME_C, "display_name": "频谱分析", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "Spectrogram"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
            {"name": "window_size", "type": "int", "default": 64, "required": False},
        ],
        "outputs": [
            {"name": "spectrogram", "type": "ndarray_2d", "label": "时频图"},
            {"name": "frequencies", "type": "ndarray_1d", "label": "频率轴"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class Spectrogram(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        ws = int(inputs.get("window_size", 64))
        n = len(curve)
        n_windows = max(1, (n - ws) // (ws // 2) + 1)
        spec = np.zeros((n_windows, ws // 2))
        for i in range(n_windows):
            start = i * (ws // 2)
            segment = curve[start:start+ws]
            if len(segment) < ws:
                segment = np.pad(segment, (0, ws - len(segment)))
            fft_mag = np.abs(np.fft.rfft(segment * np.hanning(ws)))[1:]
            spec[i, :len(fft_mag)] = fft_mag[:ws//2]
        freqs = np.fft.rfftfreq(ws, d=1.0)[1:ws//2+1]
        return AlgorithmResult(outputs={"spectrogram": spec, "frequencies": freqs})
''')
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    saved = runner.run(NAME_C, BH_NAME, {"window_size": "32"})
    spec = np.loadtxt(saved["spectrogram"], delimiter=",")
    freqs = np.loadtxt(saved["frequencies"], delimiter=",")
    assert spec.ndim == 2, f"Expected 2D, got {spec.ndim}D"
    assert freqs.ndim == 1
    ok("spectrogram", f"shape={spec.shape}, freqs={len(freqs)}")
except Exception as e:
    fail("spectrogram", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_C)

# ══ Type D: Progress reporting ════════════════════════════════════════
print(f"\n{'─'*70}")
print("D: Progress reporting — Monte Carlo simulation")
print(f"{'─'*70}")

NAME_D = "test_var_montecarlo"
try:
    _create_algo(NAME_D, {
        "name": NAME_D, "display_name": "蒙特卡洛模拟", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "MonteCarlo"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
            {"name": "n_iter", "type": "int", "default": 50, "required": False},
        ],
        "outputs": [
            {"name": "mean_curve", "type": "ndarray_1d"},
            {"name": "std_value", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class MonteCarlo(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        n_iter = int(inputs.get("n_iter", 50))
        rng = np.random.default_rng(0)
        accumulator = np.zeros_like(curve)
        for i in range(n_iter):
            self.report_progress((i + 1) / n_iter, f"Iteration {i+1}/{n_iter}")
            noise = rng.standard_normal(len(curve)) * 0.1
            accumulator += curve + noise
        mean_curve = accumulator / n_iter
        std_val = float(np.std(mean_curve - curve))
        return AlgorithmResult(outputs={"mean_curve": mean_curve, "std_value": std_val})
''')
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    progress_log = []
    saved = runner.run(NAME_D, BH_NAME, {"n_iter": "20"},
                       progress_cb=lambda f, m: progress_log.append((f, m)))
    assert len(progress_log) == 20, f"Expected 20 callbacks, got {len(progress_log)}"
    assert progress_log[-1][0] == 1.0
    ok("montecarlo", f"{len(progress_log)} progress callbacks, std={float(Path(saved['std_value']).read_text(encoding='utf-8')):.4f}")
except Exception as e:
    fail("montecarlo", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_D)

# ══ Type E: String output ═════════════════════════════════════════════
print(f"\n{'─'*70}")
print("E: String output — Lithology classification")
print(f"{'─'*70}")

NAME_E = "test_var_lithology"
try:
    _create_algo(NAME_E, {
        "name": NAME_E, "display_name": "岩性识别", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "Lithology"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
        ],
        "outputs": [
            {"name": "classification", "type": "str", "label": "岩性分类结果"},
            {"name": "confidence", "type": "float", "label": "置信度"},
        ],
    }, '''\
import json
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class Lithology(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        mean_gr = float(np.mean(curve))
        if mean_gr > 80:
            lith = "shale"
            conf = 0.9
        elif mean_gr > 40:
            lith = "sandy_shale"
            conf = 0.7
        else:
            lith = "sandstone"
            conf = 0.85
        result_json = json.dumps({
            "lithology": lith,
            "mean_gr": round(mean_gr, 1),
            "n_samples": len(curve),
        })
        return AlgorithmResult(
            outputs={"classification": result_json, "confidence": conf},
        )
''')
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    saved = runner.run(NAME_E, BH_NAME, {})
    cls_path = Path(saved["classification"])
    assert cls_path.suffix == ".txt", f"Expected .txt, got {cls_path.suffix}"
    cls_text = cls_path.read_text(encoding="utf-8").strip()
    cls_data = json.loads(cls_text)
    assert "lithology" in cls_data
    conf = float(Path(saved["confidence"]).read_text(encoding="utf-8"))
    ok("lithology", f"class={cls_data['lithology']}, conf={conf:.1f}, GR={cls_data['mean_gr']}")
except Exception as e:
    fail("lithology", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_E)

# ══ Type F: 3D output (NPZ) ══════════════════════════════════════════
print(f"\n{'─'*70}")
print("F: 3D output — Model interpolation (ndarray_3d)")
print(f"{'─'*70}")

NAME_F = "test_var_interp3d"
try:
    # Create 2D scatter input data
    raw_log = WORKSPACE / "boreholes" / BH_NAME / "logging" / "raw"
    scatter = np.column_stack([
        np.random.default_rng(42).uniform(0, 100, 50),  # x
        np.random.default_rng(43).uniform(0, 100, 50),  # y
        np.random.default_rng(44).uniform(0, 50, 50),   # z
        np.random.default_rng(45).uniform(10, 1000, 50), # value
    ])
    np.savetxt(raw_log / "scatter_data.csv", scatter, delimiter=",")

    _create_algo(NAME_F, {
        "name": NAME_F, "display_name": "三维模型插值", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "3d",
        "entry": {"type": "python", "module": "algorithm", "class": "Interp3D"},
        "inputs": [
            {"name": "scatter_data", "type": "ndarray_2d", "file": "scatter_data.csv",
             "label": "散点数据 (x,y,z,value)"},
            {"name": "nx", "type": "int", "default": 10, "required": False},
            {"name": "ny", "type": "int", "default": 10, "required": False},
            {"name": "nz", "type": "int", "default": 8, "required": False},
        ],
        "outputs": [
            {"name": "model_3d", "type": "ndarray_3d", "label": "三维模型"},
            {"name": "mean_value", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class Interp3D(BaseAlgorithm):
    def run(self, **inputs):
        data = inputs["scatter_data"]
        nx = int(inputs.get("nx", 10))
        ny = int(inputs.get("ny", 10))
        nz = int(inputs.get("nz", 8))
        # Simple nearest-neighbor 3D interpolation
        x, y, z, val = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)
        zi = np.linspace(z.min(), z.max(), nz)
        model = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    dist = (x - xi[i])**2 + (y - yi[j])**2 + (z - zi[k])**2
                    model[i, j, k] = val[np.argmin(dist)]
            self.report_progress((i + 1) / nx, f"Layer {i+1}/{nx}")
        return AlgorithmResult(
            outputs={"model_3d": model, "mean_value": float(np.mean(model))},
        )
''')
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    saved = runner.run(NAME_F, BH_NAME, {"nx": "8", "ny": "8", "nz": "6"})

    # 3D outputs are saved as NPZ
    npz_path = Path(saved["model_3d"])
    assert npz_path.suffix == ".npz", f"Expected .npz, got {npz_path.suffix}"
    loaded = np.load(str(npz_path))
    model = loaded["data"]
    assert model.ndim == 3, f"Expected 3D, got {model.ndim}D"
    assert model.shape == (8, 8, 6), f"Expected (8,8,6), got {model.shape}"
    mean_val = float(Path(saved["mean_value"]).read_text(encoding="utf-8"))
    ok("interp3d", f"shape={model.shape}, mean={mean_val:.1f}")
except Exception as e:
    fail("interp3d", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_F)

# ── Summary ───────────────────────────────────────────────────────────
total = _pass + _fail
print(f"\n{'='*70}")
print(f"  Results: {_pass} passed, {_fail} failed / {total} total")
if _fail:
    print("  *** SOME TESTS FAILED ***")
print(f"{'='*70}")
sys.exit(1 if _fail else 0)
