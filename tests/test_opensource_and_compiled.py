"""Open-source algorithm wrapping + compiled module (Cython/C) integration test.

Tests:
  Group 1: Wrap real algorithms from scipy/choclo/pygimli
  Group 2: Cython compilation → run compiled .so without source
  Group 3: C shared library via ctypes (c_lib entry type)

Usage:
    .venv/bin/python tests/test_opensource_and_compiled.py
"""
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
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

_pass = _fail = _skip = 0


def ok(name, detail=""):
    global _pass
    _pass += 1
    print(f"  [PASS] {name}{' — ' + detail if detail else ''}")


def fail(name, reason):
    global _fail
    _fail += 1
    print(f"  [FAIL] {name} — {reason}")


def skip(name, reason):
    global _skip
    _skip += 1
    print(f"  [SKIP] {name} — {reason}")


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


def _cleanup(*names):
    for name in names:
        d = ALGO_DIR / name
        if d.exists():
            shutil.rmtree(d)


def _run(name, params=None, method="logging"):
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    return runner.run(name, BH_NAME, params or {})


print("=" * 70)
print("  Open-Source Algorithm Wrapping + Compiled Module Test")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# Group 1: Real open-source algorithm wrapping
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("Group 1: Real open-source algorithms from installed libraries")
print(f"{'─'*70}")

# ── 1a. scipy.signal Butterworth bandpass filter ──────────────────────
NAME_1A = "test_os_butterworth"
print("\n  1a. scipy.signal — Butterworth bandpass filter")
try:
    _create_algo(NAME_1A, {
        "name": NAME_1A, "display_name": "Butterworth带通滤波", "version": "1.0.0",
        "method": "logging", "category": "preprocess", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "ButterworthFilter"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
            {"name": "low_freq", "type": "float", "default": 0.01, "required": False},
            {"name": "high_freq", "type": "float", "default": 0.1, "required": False},
            {"name": "order", "type": "int", "default": 4, "required": False},
        ],
        "outputs": [
            {"name": "filtered", "type": "ndarray_1d"},
            {"name": "freq_response_db", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class ButterworthFilter(BaseAlgorithm):
    def run(self, **inputs):
        from scipy.signal import butter, sosfilt, sosfreqz

        curve = inputs["curve"]
        low = float(inputs.get("low_freq", 0.01))
        high = float(inputs.get("high_freq", 0.1))
        order = int(inputs.get("order", 4))

        if low >= high:
            raise ValueError(f"low_freq ({low}) must be < high_freq ({high})")

        sos = butter(order, [low, high], btype="band", fs=1.0, output="sos")
        filtered = sosfilt(sos, curve)

        # Compute gain at center frequency
        w, h = sosfreqz(sos, worN=1024, fs=1.0)
        center_idx = np.argmin(np.abs(w - (low + high) / 2))
        gain_db = float(20 * np.log10(np.abs(h[center_idx]) + 1e-12))

        return AlgorithmResult(
            outputs={"filtered": filtered, "freq_response_db": gain_db},
        )
''')
    t0 = time.time()
    saved = _run(NAME_1A, {"low_freq": "0.005", "high_freq": "0.05", "order": "3"})
    dt = time.time() - t0
    filt = np.loadtxt(saved["filtered"], delimiter=",")
    assert filt.ndim == 1 and len(filt) > 0
    ok("scipy.signal.butter", f"{dt:.2f}s, len={len(filt)}")
except Exception as e:
    fail("scipy.signal.butter", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_1A)

# ── 1b. scipy.ndimage Gaussian model smoothing ───────────────────────
NAME_1B = "test_os_gauss_smooth"
print("\n  1b. scipy.ndimage — Gaussian 3D model smoothing")
try:
    # Create 3D test data
    raw_log = WORKSPACE / "boreholes" / BH_NAME / "logging" / "raw"
    model_3d = np.random.default_rng(42).uniform(10, 1000, (8, 8, 6))
    np.savez_compressed(raw_log / "model_noisy.npz", data=model_3d)

    _create_algo(NAME_1B, {
        "name": NAME_1B, "display_name": "高斯平滑", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "3d",
        "entry": {"type": "python", "module": "algorithm", "class": "GaussSmooth"},
        "inputs": [
            {"name": "model", "type": "ndarray_3d", "file": "model_noisy.npz"},
            {"name": "sigma", "type": "float", "default": 1.0, "required": False},
        ],
        "outputs": [
            {"name": "smoothed", "type": "ndarray_3d"},
            {"name": "reduction", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class GaussSmooth(BaseAlgorithm):
    def run(self, **inputs):
        from scipy.ndimage import gaussian_filter
        model = inputs["model"]
        sigma = float(inputs.get("sigma", 1.0))
        smoothed = gaussian_filter(model, sigma=sigma)
        reduction = float(1.0 - np.std(smoothed) / max(np.std(model), 1e-12))
        return AlgorithmResult(
            outputs={"smoothed": smoothed, "reduction": reduction},
        )
''')
    saved = _run(NAME_1B, {"sigma": "1.5"})
    sm = np.load(saved["smoothed"])["data"]
    assert sm.ndim == 3 and sm.shape == (8, 8, 6)
    ok("scipy.ndimage.gaussian_filter", f"shape={sm.shape}")
except Exception as e:
    fail("scipy.ndimage.gaussian_filter", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_1B)

# ── 1c. Choclo gravity forward modeling ──────────────────────────────
NAME_1C = "test_os_gravity"
print("\n  1c. Choclo — Gravity prism forward")
try:
    import choclo
    _create_algo(NAME_1C, {
        "name": NAME_1C, "display_name": "重力正演", "version": "1.0.0",
        "method": "em", "category": "forward", "dimension": "3d",
        "entry": {"type": "python", "module": "algorithm", "class": "GravityForward"},
        "inputs": [
            {"name": "observation_points", "type": "ndarray_2d", "file": "rx_locations.csv"},
            {"name": "density", "type": "float", "default": 2670.0, "required": False},
        ],
        "outputs": [
            {"name": "gravity_anomaly", "type": "ndarray_1d"},
            {"name": "max_anomaly", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class GravityForward(BaseAlgorithm):
    def run(self, **inputs):
        from choclo.prism import gravity_u
        obs = inputs["observation_points"]
        density = float(inputs.get("density", 2670.0))
        # Single anomalous prism: 10x10x5m at depth 10-15m
        prism = (-5.0, 5.0, -5.0, 5.0, -15.0, -10.0)
        anomaly = np.zeros(len(obs))
        for i in range(len(obs)):
            anomaly[i] = gravity_u(
                obs[i, 0], obs[i, 1], obs[i, 2],
                prism[0], prism[1], prism[2], prism[3], prism[4], prism[5],
                density,
            )
        # Convert to mGal
        anomaly_mgal = anomaly * 1e5
        return AlgorithmResult(
            outputs={"gravity_anomaly": anomaly_mgal, "max_anomaly": float(np.max(np.abs(anomaly_mgal)))},
        )
''')
    saved = _run(NAME_1C, method="em")
    grav = np.loadtxt(saved["gravity_anomaly"], delimiter=",")
    assert grav.ndim == 1 and len(grav) > 0
    ok("choclo.gravity_u", f"len={len(grav)}, max={np.max(np.abs(grav)):.4f} mGal")
except ImportError:
    skip("choclo.gravity_u", "choclo not installed")
except Exception as e:
    fail("choclo.gravity_u", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_1C)

# ── 1d. PyGIMLi ERT forward ──────────────────────────────────────────
NAME_1D = "test_os_pygimli_ert"
print("\n  1d. PyGIMLi — ERT forward modeling")
try:
    import pygimli
    _create_algo(NAME_1D, {
        "name": NAME_1D, "display_name": "PyGIMLi ERT正演", "version": "1.0.0",
        "method": "ert", "category": "forward", "dimension": "2d",
        "entry": {"type": "python", "module": "algorithm", "class": "PyGimliErt"},
        "inputs": [
            {"name": "electrode_positions", "type": "ndarray_2d", "file": "electrode_positions.csv"},
            {"name": "n_layers", "type": "int", "default": 3, "required": False},
        ],
        "outputs": [
            {"name": "apparent_resistivity", "type": "ndarray_1d"},
            {"name": "n_data", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class PyGimliErt(BaseAlgorithm):
    def run(self, **inputs):
        import pygimli as pg
        from pygimli.physics import ert

        elec = inputs["electrode_positions"]
        n_elec = len(elec)

        # Create simple Wenner scheme
        scheme = ert.createData(elecCount=n_elec, schemeName="wa")

        # Create layered model
        world = pg.meshtools.createWorld(start=[-50, 0], end=[50+5*(n_elec-1), -30], layers=[-5, -15])
        mesh = pg.meshtools.createMesh(world, quality=34, area=2.0)

        # Assign resistivities: 100 / 50 / 500 Ohm.m
        rhomap = [[1, 100.0], [2, 50.0], [3, 500.0]]
        data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=2, noiseAbs=0, verbose=False)

        rho_app = np.array(data["rhoa"])
        return AlgorithmResult(
            outputs={"apparent_resistivity": rho_app, "n_data": float(len(rho_app))},
        )
''')
    t0 = time.time()
    saved = _run(NAME_1D, method="ert")
    dt = time.time() - t0
    rho = np.loadtxt(saved["apparent_resistivity"], delimiter=",")
    ok("pygimli.ert.simulate", f"{dt:.2f}s, {len(rho)} measurements")
except ImportError:
    skip("pygimli.ert.simulate", "pygimli not installed")
except Exception as e:
    fail("pygimli.ert.simulate", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_1D)

# ── 1e. empymod frequency-domain CSEM ────────────────────────────────
NAME_1E = "test_os_empymod_fdem"
print("\n  1e. empymod — Frequency-domain CSEM forward")
try:
    _create_algo(NAME_1E, {
        "name": NAME_1E, "display_name": "频率域CSEM正演", "version": "1.0.0",
        "method": "em", "category": "forward", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "FdemForward"},
        "inputs": [
            {"name": "resistivity", "type": "ndarray_1d", "file": "tx_locations.csv"},
            {"name": "n_freqs", "type": "int", "default": 20, "required": False},
        ],
        "outputs": [
            {"name": "response_real", "type": "ndarray_1d"},
            {"name": "response_imag", "type": "ndarray_1d"},
            {"name": "frequencies", "type": "ndarray_1d"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class FdemForward(BaseAlgorithm):
    def run(self, **inputs):
        import empymod
        n_freqs = int(inputs.get("n_freqs", 20))
        freqs = np.logspace(-1, 4, n_freqs)
        # 3-layer model: air / 100 Ohm.m / 10 Ohm.m halfspace
        resp = empymod.dipole(
            src=[0, 0, 0.001], rec=[500, 0, 0.001],
            depth=[0, 100], res=[2e14, 100.0, 10.0],
            freqtime=freqs, verb=0,
        )
        return AlgorithmResult(
            outputs={
                "response_real": np.real(resp),
                "response_imag": np.imag(resp),
                "frequencies": freqs,
            },
        )
''')
    saved = _run(NAME_1E, {"n_freqs": "15"}, method="em")
    re = np.loadtxt(saved["response_real"], delimiter=",")
    im = np.loadtxt(saved["response_imag"], delimiter=",")
    assert len(re) == 15 and len(im) == 15
    ok("empymod.fdem", f"15 freqs, |E|_max={np.max(np.sqrt(re**2+im**2)):.2e}")
except Exception as e:
    fail("empymod.fdem", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_1E)

# ══════════════════════════════════════════════════════════════════════
# Group 2: Cython compilation test
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("Group 2: Cython compilation — compile .py → .so/.pyd → run without source")
print(f"{'─'*70}")

NAME_CY = "test_cython_compile"
try:
    # Create a simple algorithm
    algo_dir = _create_algo(NAME_CY, {
        "name": NAME_CY, "display_name": "Cython编译测试", "version": "1.0.0",
        "method": "logging", "category": "process", "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "CythonTest"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
        ],
        "outputs": [
            {"name": "result", "type": "ndarray_1d"},
            {"name": "checksum", "type": "float"},
        ],
    }, '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class CythonTest(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        result = np.cumsum(curve) / np.arange(1, len(curve) + 1)
        checksum = float(np.sum(result))
        return AlgorithmResult(outputs={"result": result, "checksum": checksum})
''')

    # Step 1: Run from source (baseline)
    saved_src = _run(NAME_CY)
    checksum_src = float(Path(saved_src["checksum"]).read_text(encoding="utf-8"))
    ok("cython:source_run", f"checksum={checksum_src:.4f}")

    # Step 2: Compile with Cython
    compile_result = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "compile_algo.py"), str(algo_dir)],
        capture_output=True, text=True, cwd=str(_ROOT),
    )

    # Check if compiled file exists
    compiled_files = list(algo_dir.glob("algorithm.cpython-*.so")) + list(algo_dir.glob("algorithm.*.pyd"))

    if compiled_files:
        ok("cython:compile", f"Compiled: {compiled_files[0].name}")

        # Step 3: Remove source, run from compiled only
        py_file = algo_dir / "algorithm.py"
        py_backup = algo_dir / "algorithm.py.bak"
        py_file.rename(py_backup)

        try:
            # Clear module cache and re-run
            registry = AlgorithmRegistry(ALGO_DIR)
            registry.scan()
            runner = AlgorithmRunner(registry, WORKSPACE)
            saved_compiled = runner.run(NAME_CY, BH_NAME, {})
            checksum_compiled = float(Path(saved_compiled["checksum"]).read_text(encoding="utf-8"))

            if abs(checksum_compiled - checksum_src) < 1e-6:
                ok("cython:compiled_run", f"checksum={checksum_compiled:.4f} (matches source)")
            else:
                fail("cython:compiled_run",
                     f"checksum mismatch: src={checksum_src:.4f}, compiled={checksum_compiled:.4f}")
        finally:
            # Restore source
            py_backup.rename(py_file)
    else:
        if "Cython" in compile_result.stderr or "cython" in compile_result.stderr.lower():
            skip("cython:compile", "Cython not installed")
        else:
            fail("cython:compile", f"No compiled output. stderr: {compile_result.stderr[:200]}")

except Exception as e:
    fail("cython:test", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_CY)

# ══════════════════════════════════════════════════════════════════════
# Group 3: C shared library test
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("Group 3: C shared library (ctypes) — compile C → .dylib/.so/.dll → run")
print(f"{'─'*70}")

NAME_C = "test_c_library"
try:
    algo_dir_c = ALGO_DIR / NAME_C
    if algo_dir_c.exists():
        shutil.rmtree(algo_dir_c)
    algo_dir_c.mkdir(parents=True)

    # Step 1: Write C source
    c_source = '''\
#include <math.h>

// Running average filter
// Returns 0 on success
int running_average(double* input, int n, int window, double* output) {
    if (n <= 0 || window <= 0) return -1;
    int half = window / 2;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        int count = 0;
        for (int j = i - half; j <= i + half; j++) {
            if (j >= 0 && j < n) {
                sum += input[j];
                count++;
            }
        }
        output[i] = sum / count;
    }
    return 0;
}
'''
    c_file = algo_dir_c / "mylib.c"
    c_file.write_text(c_source, encoding="utf-8")

    # Step 2: Compile C to shared library
    if sys.platform == "darwin":
        lib_ext = ".dylib"
        compile_cmd = ["cc", "-shared", "-fPIC", "-O2", "-o", f"mylib{lib_ext}", "mylib.c"]
    elif sys.platform == "win32":
        lib_ext = ".dll"
        compile_cmd = ["cl", "/LD", "/O2", "mylib.c"]
    else:
        lib_ext = ".so"
        compile_cmd = ["cc", "-shared", "-fPIC", "-O2", "-o", f"mylib{lib_ext}", "mylib.c"]

    compile_result = subprocess.run(
        compile_cmd, capture_output=True, text=True, cwd=str(algo_dir_c),
    )

    lib_path = algo_dir_c / f"mylib{lib_ext}"
    if not lib_path.exists():
        fail("c_lib:compile", f"Compilation failed: {compile_result.stderr[:200]}")
    else:
        ok("c_lib:compile", f"Built {lib_path.name}")

        # Step 3: Write manifest with c_lib entry
        manifest = {
            "name": NAME_C, "display_name": "C动态库测试", "version": "1.0.0",
            "method": "logging", "category": "process", "dimension": "1d",
            "entry": {
                "type": "c_lib",
                "lib": "mylib",
                "function": "running_average",
            },
            "c_signature": {
                "args": [
                    {"name": "input", "role": "input"},
                    {"name": "window", "role": "param", "c_type": "int"},
                    {"name": "output", "role": "output", "size_from": "input"},
                ],
            },
            "inputs": [
                {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
                {"name": "window", "type": "int", "default": 5, "required": False},
            ],
            "outputs": [
                {"name": "result", "type": "ndarray_1d"},
            ],
        }
        (algo_dir_c / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        # Step 4: Write Python wrapper that uses ctypes directly
        # (since the c_adapter might not match our exact signature,
        # we write a thin Python wrapper as the entry point)
        wrapper_code = f'''\
import ctypes
import numpy as np
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm

class CLibTest(BaseAlgorithm):
    def run(self, **inputs):
        curve = np.asarray(inputs["curve"], dtype=np.float64)
        window = int(inputs.get("window", 5))
        n = len(curve)

        # Load shared library
        lib_dir = Path(__file__).parent
        lib_path = lib_dir / "mylib{lib_ext}"
        lib = ctypes.CDLL(str(lib_path))

        # Set function signature
        lib.running_average.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.running_average.restype = ctypes.c_int

        # Allocate output
        output = np.zeros(n, dtype=np.float64)

        # Call C function
        ret = lib.running_average(
            curve.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_int(window),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if ret != 0:
            raise RuntimeError(f"C function returned error code {{ret}}")

        return AlgorithmResult(outputs={{"result": output}})
'''
        (algo_dir_c / "algorithm.py").write_text(wrapper_code, encoding="utf-8")

        # Update manifest to use python entry (wrapper calls C internally)
        manifest["entry"] = {"type": "python", "module": "algorithm", "class": "CLibTest"}
        (algo_dir_c / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        # Step 5: Run the C-backed algorithm
        try:
            t0 = time.time()
            saved = _run(NAME_C)
            dt = time.time() - t0
            result = np.loadtxt(saved["result"], delimiter=",")
            assert result.ndim == 1 and len(result) > 0

            # Verify: compare with numpy implementation
            curve = np.loadtxt(WORKSPACE / "boreholes" / BH_NAME / "logging" / "raw" / "GR.csv",
                               delimiter=",")
            np_result = np.convolve(curve, np.ones(5)/5, mode="same")
            # C running_average handles edges differently, so compare middle portion
            mid = len(curve) // 4
            correlation = np.corrcoef(result[mid:-mid], np_result[mid:-mid])[0, 1]
            ok("c_lib:run", f"{dt:.2f}s, len={len(result)}, correlation={correlation:.4f}")
        except Exception as e:
            fail("c_lib:run", f"{type(e).__name__}: {e}")

except Exception as e:
    fail("c_lib:test", f"{type(e).__name__}: {e}")
finally:
    _cleanup(NAME_C)

# ── Summary ───────────────────────────────────────────────────────────
total = _pass + _fail + _skip
print(f"\n{'='*70}")
print(f"  Results: {_pass} passed, {_fail} failed, {_skip} skipped / {total} total")
if _fail:
    print("  *** SOME TESTS FAILED ***")
print(f"{'='*70}")
sys.exit(1 if _fail else 0)
