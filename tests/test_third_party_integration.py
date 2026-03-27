"""Third-party algorithm integration test.

Simulates an external developer submitting algorithms through the standard interface.
Tests: scaffold → implement → validate → run → visualize → edge cases.

Usage:
    .venv/bin/python tests/test_third_party_integration.py
"""
import json
import shutil
import subprocess
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
TOOLS_DIR = _ROOT / "tools"
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


# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  Third-Party Algorithm Integration Test")
print("=" * 70)

# ── Test Group 1: Scaffold + Implement + Validate + Run ───────────────
print(f"\n{'─'*70}")
print("Group 1: Full lifecycle — scaffold → implement → validate → run")
print(f"{'─'*70}")

MOCK_ALGO = "tem_preproc_bandpass"
mock_dir = ALGO_DIR / MOCK_ALGO

# Clean up from previous runs
if mock_dir.exists():
    shutil.rmtree(mock_dir)

# 1a. Scaffold a new algorithm using the tool
try:
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "scaffold_algo.py"),
         "--name", MOCK_ALGO,
         "--display-name", "TEM Bandpass Filter",
         "--method", "tem", "--category", "preprocess",
         "--template", "1d-preprocess"],
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert mock_dir.exists()
    ok("scaffold:create", f"Created {MOCK_ALGO}")
except Exception as e:
    fail("scaffold:create", str(e))

# 1b. Replace generated algorithm.py with real bandpass implementation
try:
    algo_code = '''\
"""TEM Bandpass Filter — third-party algorithm example."""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class TemPreprocBandpass(BaseAlgorithm):
    def run(self, **inputs) -> AlgorithmResult:
        curve = inputs["curve"]
        window_size = int(inputs.get("window_size", 11))
        threshold = float(inputs.get("threshold", 0.5))

        warnings = []

        # Simple moving average bandpass (real algorithm would use scipy)
        self.report_progress(0.2, "Applying filter...")
        kernel = np.ones(window_size) / window_size
        if len(curve) < window_size:
            raise ValueError(
                f"Curve length ({len(curve)}) < window_size ({window_size}). "
                f"Reduce window_size or provide longer data."
            )
        curve_filtered = np.convolve(curve, kernel, mode="same")

        # Quality metric: variance reduction ratio
        self.report_progress(0.8, "Computing quality metric...")
        residual = curve - curve_filtered
        if np.var(curve) > 0:
            quality_metric = float(1.0 - np.var(residual) / np.var(curve))
        else:
            quality_metric = 0.0
            warnings.append("Input curve has zero variance.")

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={
                "curve_filtered": curve_filtered,
                "quality_metric": quality_metric,
            },
            warnings=warnings,
        )
'''
    (mock_dir / "algorithm.py").write_text(algo_code, encoding="utf-8")
    ok("scaffold:implement", "Replaced algorithm.py with real code")
except Exception as e:
    fail("scaffold:implement", str(e))

# 1c. Validate with tool
try:
    result = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "validate_algo.py"), str(mock_dir)],
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert "[ERROR]" not in result.stdout, f"Validation errors:\n{result.stdout}"
    ok("validate:pass", "0 errors")
except Exception as e:
    fail("validate:pass", str(e))

# 1d. Load and run through AlgorithmRunner
try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    # Prepare input data: use existing GR.csv as "input.csv"
    raw_dir = WORKSPACE / "boreholes" / BH_NAME / "tem" / "raw"
    src = raw_dir / "emf.csv"
    dst = raw_dir / "input.csv"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)

    t0 = time.time()
    saved = runner.run(MOCK_ALGO, BH_NAME, {"window_size": "5", "threshold": "0.3"})
    dt = time.time() - t0

    assert "curve_filtered" in saved, f"Missing output. Got: {list(saved.keys())}"
    assert "quality_metric" in saved
    assert Path(saved["curve_filtered"]).exists()
    ok("runner:execute", f"{dt:.2f}s, outputs: {list(saved.keys())}")
except Exception as e:
    fail("runner:execute", f"{type(e).__name__}: {e}")

# 1e. Verify output files are correctly formatted
try:
    filtered = np.loadtxt(saved["curve_filtered"], delimiter=",")
    assert filtered.ndim == 1, f"Expected 1D, got {filtered.ndim}D"
    assert len(filtered) > 0

    qm = float(Path(saved["quality_metric"]).read_text(encoding="utf-8").strip())
    assert -1.0 <= qm <= 1.0, f"Quality metric out of range: {qm}"

    meta_path = Path(saved["curve_filtered"]).parent / "run_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "timestamp" in meta
    ok("output:verify", f"Filtered={len(filtered)} pts, QM={qm:.3f}")
except Exception as e:
    fail("output:verify", str(e))

# ── Test Group 2: Positional keyword calling convention ───────────────
print(f"\n{'─'*70}")
print("Group 2: Positional keyword args (SimPEG-style run signature)")
print(f"{'─'*70}")

MOCK_ALGO_POS = "ert_preproc_normalize_test"
mock_dir_pos = ALGO_DIR / MOCK_ALGO_POS

if mock_dir_pos.exists():
    shutil.rmtree(mock_dir_pos)

# Create algorithm with POSITIONAL keyword args (not **inputs)
try:
    mock_dir_pos.mkdir(parents=True)
    manifest = {
        "name": MOCK_ALGO_POS,
        "display_name": "ERT Data Normalization Test",
        "version": "1.0.0",
        "method": "ert",
        "category": "preprocess",
        "dimension": "2d",
        "entry": {"type": "python", "module": "algorithm", "class": "ErtNormalize"},
        "inputs": [
            {"name": "apparent_resistivity", "label": "Input data", "type": "ndarray_1d",
             "file": "apparent_resistivity.csv"},
            {"name": "scale_factor", "label": "Scale factor", "type": "float",
             "default": 1.0, "required": False},
        ],
        "outputs": [
            {"name": "normalized", "label": "Normalized data", "type": "ndarray_1d"},
            {"name": "stats", "label": "Statistics", "type": "float"},
        ],
    }
    (mock_dir_pos / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    algo_code = '''\
"""ERT normalization with positional keyword signature."""
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm


class ErtNormalize(BaseAlgorithm):
    def run(self, apparent_resistivity: np.ndarray, scale_factor: float = 1.0) -> AlgorithmResult:
        """Positional keyword style — not **inputs."""
        data = np.asarray(apparent_resistivity, dtype=float).ravel()
        mean_val = float(np.mean(data))
        normalized = (data / mean_val) * scale_factor
        return AlgorithmResult(
            outputs={"normalized": normalized, "stats": mean_val},
        )
'''
    (mock_dir_pos / "algorithm.py").write_text(algo_code, encoding="utf-8")

    # Reload registry
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    saved = runner.run(MOCK_ALGO_POS, BH_NAME, {"scale_factor": "2.0"})
    assert "normalized" in saved
    ok("positional_kwargs:run", f"Outputs: {list(saved.keys())}")
except Exception as e:
    fail("positional_kwargs:run", f"{type(e).__name__}: {e}")

# ── Test Group 3: Edge cases ──────────────────────────────────────────
print(f"\n{'─'*70}")
print("Group 3: Edge cases and error handling")
print(f"{'─'*70}")

# 3a. Algorithm returning wrong dimension
MOCK_WRONG_DIM = "test_wrong_dim"
mock_dir_wd = ALGO_DIR / MOCK_WRONG_DIM
if mock_dir_wd.exists():
    shutil.rmtree(mock_dir_wd)
try:
    mock_dir_wd.mkdir(parents=True)
    manifest = {
        "name": MOCK_WRONG_DIM,
        "display_name": "Wrong Dim Test",
        "version": "1.0.0", "method": "logging", "category": "preprocess",
        "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "WrongDim"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
        ],
        "outputs": [
            {"name": "result", "type": "ndarray_1d", "label": "Should be 1D"},
        ],
    }
    (mock_dir_wd / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (mock_dir_wd / "algorithm.py").write_text('''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class WrongDim(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        # BUG: return 2D instead of 1D
        return AlgorithmResult(outputs={"result": np.column_stack([curve, curve])})
''')

    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    try:
        runner.run(MOCK_WRONG_DIM, BH_NAME, {})
        fail("wrong_dim:detect", "Should have raised TypeError for 2D→1D mismatch")
    except TypeError:
        ok("wrong_dim:detect", "Correctly rejected 2D output for ndarray_1d manifest")
except Exception as e:
    fail("wrong_dim:detect", f"{type(e).__name__}: {e}")

# 3b. Algorithm with missing required input file
try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    # empymod needs resistivity.csv in tem/raw/ — test with non-existent borehole
    try:
        runner.run("empymod_tem_forward", "NONEXISTENT-BH", {})
        fail("missing_file:detect", "Should have raised FileNotFoundError")
    except FileNotFoundError:
        ok("missing_file:detect", "Correctly raised FileNotFoundError for missing borehole")
except Exception as e:
    fail("missing_file:detect", f"{type(e).__name__}: {e}")

# 3c. Algorithm that raises ValueError with context
MOCK_VALIDATION = "test_validation_error"
mock_dir_ve = ALGO_DIR / MOCK_VALIDATION
if mock_dir_ve.exists():
    shutil.rmtree(mock_dir_ve)
try:
    mock_dir_ve.mkdir(parents=True)
    manifest = {
        "name": MOCK_VALIDATION,
        "display_name": "Validation Error Test",
        "version": "1.0.0", "method": "logging", "category": "preprocess",
        "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "ValError"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
            {"name": "min_length", "type": "int", "default": 999999, "required": False},
        ],
        "outputs": [
            {"name": "result", "type": "ndarray_1d"},
        ],
    }
    (mock_dir_ve / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (mock_dir_ve / "algorithm.py").write_text('''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class ValError(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        min_length = int(inputs.get("min_length", 999999))
        if len(curve) < min_length:
            raise ValueError(
                f"Curve length ({len(curve)}) < required minimum ({min_length}). "
                f"Please provide longer data or reduce min_length parameter."
            )
        return AlgorithmResult(outputs={"result": curve})
''')

    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    try:
        runner.run(MOCK_VALIDATION, BH_NAME, {})
        fail("validation_error:message", "Should have raised ValueError")
    except ValueError as e:
        msg = str(e)
        has_context = "Curve length" in msg and "min_length" in msg
        if has_context:
            ok("validation_error:message", f"Clear message: '{msg[:80]}...'")
        else:
            fail("validation_error:message", f"Message lacks context: '{msg}'")
except Exception as e:
    fail("validation_error:message", f"{type(e).__name__}: {e}")

# 3d. Algorithm with warnings in result
MOCK_WARNINGS = "test_result_warnings"
mock_dir_w = ALGO_DIR / MOCK_WARNINGS
if mock_dir_w.exists():
    shutil.rmtree(mock_dir_w)
try:
    mock_dir_w.mkdir(parents=True)
    manifest = {
        "name": MOCK_WARNINGS,
        "display_name": "Warnings Test",
        "version": "1.0.0", "method": "logging", "category": "preprocess",
        "dimension": "1d",
        "entry": {"type": "python", "module": "algorithm", "class": "WarnTest"},
        "inputs": [
            {"name": "curve", "type": "ndarray_1d", "file": "GR.csv"},
        ],
        "outputs": [
            {"name": "result", "type": "ndarray_1d"},
        ],
    }
    (mock_dir_w / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (mock_dir_w / "algorithm.py").write_text('''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm

class WarnTest(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        warnings = [
            "Data contains NaN values (replaced with 0).",
            "SNR is below recommended threshold.",
        ]
        return AlgorithmResult(outputs={"result": curve.copy()}, warnings=warnings)
''')

    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    saved = runner.run(MOCK_WARNINGS, BH_NAME, {})

    meta_path = Path(saved["result"]).parent / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    warn_list = meta.get("warnings", [])
    if len(warn_list) == 2 and "NaN" in warn_list[0]:
        ok("warnings:persist", f"{len(warn_list)} warnings saved to run_meta.json")
    else:
        fail("warnings:persist", f"Expected 2 warnings, got: {warn_list}")
except Exception as e:
    fail("warnings:persist", f"{type(e).__name__}: {e}")

# 3e. Progress callback fires correctly
try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    progress_log = []

    def _capture_progress(frac, msg):
        progress_log.append((frac, msg))

    runner.run(MOCK_ALGO, BH_NAME, {"window_size": "5"},
               progress_cb=_capture_progress)

    if len(progress_log) >= 3:
        fracs = [p[0] for p in progress_log]
        assert fracs[-1] == 1.0, f"Last progress should be 1.0, got {fracs[-1]}"
        ok("progress:callback", f"{len(progress_log)} callbacks, fracs={fracs}")
    else:
        fail("progress:callback", f"Expected >=3 callbacks, got {len(progress_log)}")
except Exception as e:
    fail("progress:callback", f"{type(e).__name__}: {e}")

# ── Cleanup ───────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("Cleanup: Removing test algorithms")
print(f"{'─'*70}")

for d in [mock_dir, mock_dir_pos, mock_dir_wd, mock_dir_ve, mock_dir_w]:
    if d.exists():
        shutil.rmtree(d)
        print(f"  Removed {d.name}")

# ── Summary ───────────────────────────────────────────────────────────
total = _pass + _fail + _skip
print(f"\n{'='*70}")
print(f"  Results: {_pass} passed, {_fail} failed, {_skip} skipped / {total} total")
if _fail:
    print("  *** SOME TESTS FAILED ***")
print(f"{'='*70}")
sys.exit(1 if _fail else 0)
