"""End-to-end workflow test — simulates real user operations.

Tests: data import → single algorithm runs → pipeline runs → visualization.
Requires: tests/build_test_datasets.py to have been run first.

Usage:
    .venv/bin/python tests/test_e2e_workflow.py
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from algorithms.pipeline_runner import PipelineRunner
from algorithms.registry import AlgorithmRegistry
from algorithms.runner import AlgorithmRunner
from core.data_store import DataStore

WORKSPACE = Path.home() / "geophys-workspace"
BH_NAME = "TEST-WELL"
ALGO_DIR = _ROOT / "algorithms"
PIPE_DIR = _ROOT / "pipelines"


class _TestResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append(name)
        suffix = f" — {detail}" if detail else ""
        print(f"  [PASS] {name}{suffix}")

    def fail(self, name: str, reason: str) -> None:
        self.failed.append((name, reason))
        print(f"  [FAIL] {name} — {reason}")

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        print(f"  [SKIP] {name} — {reason}")

    def summary(self) -> None:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(f"\n{'='*60}")
        print(f"  Results: {len(self.passed)} passed, {len(self.failed)} failed, "
              f"{len(self.skipped)} skipped / {total} total")
        if self.failed:
            print("\n  Failures:")
            for name, reason in self.failed:
                print(f"    - {name}: {reason}")
        print(f"{'='*60}")


results = _TestResult()

# ── Phase 1: Setup ────────────────────────────────────────────────────
print("=" * 60)
print("  E2E Workflow Test")
print("=" * 60)

store = DataStore(WORKSPACE)
bh_list = store.list_boreholes()
if BH_NAME not in bh_list:
    print(f"ERROR: {BH_NAME} not found. Run build_test_datasets.py first.")
    sys.exit(1)

registry = AlgorithmRegistry(ALGO_DIR)
registry.scan()
runner = AlgorithmRunner(registry, WORKSPACE)

print(f"\nWorkspace: {WORKSPACE}")
print(f"Borehole:  {BH_NAME}")
print(f"Algorithms: {len(registry.list_manifests())} registered")

# ── Phase 2: Single Algorithm Runs ────────────────────────────────────
print(f"\n{'─'*60}")
print("Phase 2: Single Algorithm Runs")
print(f"{'─'*60}")

# 2a. Logging: Savitzky-Golay
try:
    t0 = time.time()
    saved = runner.run("log_preproc_savgol", BH_NAME, {
        "window_length": "11", "polyorder": "3",
    })
    dt = time.time() - t0
    assert "curve_filtered" in saved, "Missing output: curve_filtered"
    results.ok("log_preproc_savgol", f"{dt:.2f}s, {len(saved)} outputs")
except Exception as e:
    results.fail("log_preproc_savgol", str(e))

# 2b. Logging: Wavelet
try:
    t0 = time.time()
    saved = runner.run("log_preproc_wavelet", BH_NAME, {
        "wavelet": "db4", "level": "4",
    })
    dt = time.time() - t0
    results.ok("log_preproc_wavelet", f"{dt:.2f}s, {len(saved)} outputs")
except Exception as e:
    results.fail("log_preproc_wavelet", str(e))

# 2c. Logging: Kalman
try:
    t0 = time.time()
    saved = runner.run("log_preproc_kalman", BH_NAME, {
        "process_noise": "0.01", "measurement_noise": "1.0",
    })
    dt = time.time() - t0
    results.ok("log_preproc_kalman", f"{dt:.2f}s, {len(saved)} outputs")
except Exception as e:
    results.fail("log_preproc_kalman", str(e))

# 2d. TEM: empymod forward
try:
    t0 = time.time()
    saved = runner.run("empymod_tem_forward", BH_NAME, {})
    dt = time.time() - t0
    assert "emf" in saved, "Missing output: emf"
    assert "times" in saved, "Missing output: times"
    results.ok("empymod_tem_forward", f"{dt:.2f}s, outputs: {list(saved.keys())}")
except Exception as e:
    results.fail("empymod_tem_forward", str(e))

# 2e. TEM: 1D inversion
try:
    t0 = time.time()
    saved = runner.run("tem_1d_inversion", BH_NAME, {
        "n_layers": "15", "lambda_": "0.001",
    })
    dt = time.time() - t0
    assert "resistivity" in saved, "Missing output: resistivity"
    assert "misfit" in saved, "Missing output: misfit"

    # Check misfit value
    misfit_path = saved["misfit"]
    misfit_val = float(Path(misfit_path).read_text(encoding="utf-8").strip())
    results.ok("tem_1d_inversion", f"{dt:.2f}s, misfit={misfit_val:.3f}")
except Exception as e:
    results.fail("tem_1d_inversion", f"{type(e).__name__}: {e}")

# 2f. TEM: SimPEG 3D forward (cylindrical mesh)
try:
    t0 = time.time()
    saved = runner.run("simpeg_tem_forward", BH_NAME, {
        "loop_radius": "20.0", "n_times": "15",
    })
    dt = time.time() - t0
    assert "dbdt" in saved, "Missing output: dbdt"
    results.ok("simpeg_tem_forward", f"{dt:.2f}s, outputs: {list(saved.keys())}")
except ImportError as e:
    results.skip("simpeg_tem_forward", f"SimPEG not installed: {e}")
except Exception as e:
    results.fail("simpeg_tem_forward", f"{type(e).__name__}: {e}")

# 2g. EM: 3D crosshole forward
try:
    t0 = time.time()
    saved = runner.run("em_forward_3d", BH_NAME, {
        "frequency": "1000.0", "dx": "5.0", "dy": "5.0", "dz": "5.0",
    })
    dt = time.time() - t0
    assert "response" in saved, "Missing output: response"
    results.ok("em_forward_3d", f"{dt:.2f}s, outputs: {list(saved.keys())}")
except Exception as e:
    results.fail("em_forward_3d", f"{type(e).__name__}: {e}")

# 2h. ERT: SimPEG DC inversion
try:
    t0 = time.time()
    saved = runner.run("simpeg_dc_inversion", BH_NAME, {
        "max_iter": "5", "rho_background": "100.0",
    })
    dt = time.time() - t0
    assert "resistivity_model" in saved, "Missing output: resistivity_model"
    results.ok("simpeg_dc_inversion", f"{dt:.2f}s, outputs: {list(saved.keys())}")
except ImportError as e:
    results.skip("simpeg_dc_inversion", f"SimPEG not installed: {e}")
except Exception as e:
    results.fail("simpeg_dc_inversion", f"{type(e).__name__}: {e}")

# ── Phase 3: Pipeline Runs ────────────────────────────────────────────
print(f"\n{'─'*60}")
print("Phase 3: Pipeline Runs")
print(f"{'─'*60}")

pipe_runner = PipelineRunner(registry, WORKSPACE, PIPE_DIR)
pipe_runner.scan()

# 3a. Logging multi-filter pipeline
try:
    t0 = time.time()
    all_saved = pipe_runner.run("logging_multi_filter", BH_NAME)
    dt = time.time() - t0
    steps_done = list(all_saved.keys())
    results.ok("pipeline:logging_multi_filter", f"{dt:.2f}s, steps: {steps_done}")
except Exception as e:
    results.fail("pipeline:logging_multi_filter", f"{type(e).__name__}: {e}")
    traceback.print_exc()

# 3b. TEM 1D workflow pipeline
try:
    t0 = time.time()
    all_saved = pipe_runner.run("tem_1d_workflow", BH_NAME)
    dt = time.time() - t0
    steps_done = list(all_saved.keys())
    results.ok("pipeline:tem_1d_workflow", f"{dt:.2f}s, steps: {steps_done}")
except Exception as e:
    results.fail("pipeline:tem_1d_workflow", f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ── Phase 4: Output Verification ──────────────────────────────────────
print(f"\n{'─'*60}")
print("Phase 4: Output Verification")
print(f"{'─'*60}")

processed_dir = WORKSPACE / "boreholes" / BH_NAME
for method_dir in sorted(processed_dir.iterdir()):
    if not method_dir.is_dir():
        continue
    proc = method_dir / "processed"
    if not proc.exists():
        continue
    for algo_dir in sorted(proc.iterdir()):
        if not algo_dir.is_dir():
            continue
        files = list(algo_dir.glob("*"))
        csv_count = sum(1 for f in files if f.suffix == ".csv")
        txt_count = sum(1 for f in files if f.suffix == ".txt")
        npz_count = sum(1 for f in files if f.suffix == ".npz")
        meta = algo_dir / "run_meta.json"
        has_meta = meta.exists()

        total_kb = sum(f.stat().st_size for f in files) / 1024
        print(f"  {method_dir.name}/{algo_dir.name}: "
              f"{csv_count} csv, {txt_count} txt, {npz_count} npz, "
              f"meta={'yes' if has_meta else 'NO'}, {total_kb:.1f} KB")

# ── Phase 5: Visualization Smoke Test ─────────────────────────────────
print(f"\n{'─'*60}")
print("Phase 5: Visualization Smoke Test")
print(f"{'─'*60}")

try:
    import matplotlib
    matplotlib.use("Agg")
    from ui.result_canvas import ResultCanvas

    canvas = ResultCanvas()

    # Find a CSV output to plot
    csv_files = list((processed_dir / "logging" / "processed" / "log_preproc_savgol").glob("*.csv"))
    if csv_files:
        canvas.load_and_plot_csv([str(f) for f in csv_files[:3]])
        results.ok("viz:csv_plot", f"Plotted {len(csv_files)} CSV files")
    else:
        results.skip("viz:csv_plot", "No CSV output files found")
except Exception as e:
    results.fail("viz:csv_plot", f"{type(e).__name__}: {e}")

# ── Summary ───────────────────────────────────────────────────────────
results.summary()
sys.exit(1 if results.failed else 0)
