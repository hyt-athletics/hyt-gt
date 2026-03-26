"""Generate a new algorithm package with manifest.json and algorithm.py.

Usage:
    # Minimal (empty inputs/outputs):
    python tools/scaffold_algo.py --name ert_inv_mcmc_2d --display-name "2D ERT MCMC" --method ert --category inversion

    # With template (pre-filled inputs/outputs):
    python tools/scaffold_algo.py --name tem_preproc_denoise --display-name "TEM Denoising" --method tem --category preprocess --template 1d-preprocess

Templates:
    1d-preprocess   1D curve processing (ndarray_1d in/out + scalar params)
    2d-forward      2D forward modeling (ndarray_1d model → ndarray_2d response)
    3d-inversion    3D inversion (ndarray_2d data + ndarray_3d model → ndarray_3d result)
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ALGO_ROOT = _ROOT / "algorithms"

METHODS = ("ert", "ip", "em", "tem", "logging")
CATEGORIES = ("preprocess", "process", "forward", "inversion")
TEMPLATES = ("1d-preprocess", "2d-forward", "3d-inversion")

# ── Input/output templates ────────────────────────────────────────────

_TEMPLATE_IO = {
    "1d-preprocess": {
        "dimension": "1d",
        "inputs": [
            {"name": "curve", "label": "Input curve", "type": "ndarray_1d", "file": "input.csv"},
            {"name": "window_size", "label": "Window size", "type": "int", "default": 11, "required": False},
            {"name": "threshold", "label": "Threshold", "type": "float", "default": 0.5, "required": False},
        ],
        "outputs": [
            {"name": "curve_filtered", "label": "Processed curve", "type": "ndarray_1d"},
            {"name": "quality_metric", "label": "Quality metric", "type": "float"},
        ],
    },
    "2d-forward": {
        "dimension": "2d",
        "inputs": [
            {"name": "model", "label": "1D model (layer properties)", "type": "ndarray_1d", "file": "model.csv"},
            {"name": "depths", "label": "Layer depths", "type": "ndarray_1d", "file": "depths.csv"},
            {"name": "n_points", "label": "Output grid points", "type": "int", "default": 100, "required": False},
            {"name": "frequency", "label": "Frequency (Hz)", "type": "float", "default": 1.0, "required": False},
        ],
        "outputs": [
            {"name": "response", "label": "Forward response (2D)", "type": "ndarray_2d"},
            {"name": "coordinates", "label": "Observation coords", "type": "ndarray_2d"},
        ],
    },
    "3d-inversion": {
        "dimension": "3d",
        "inputs": [
            {"name": "observed_data", "label": "Observed data", "type": "ndarray_2d", "file": "observed.csv"},
            {"name": "initial_model", "label": "Initial model (optional)", "type": "ndarray_3d",
             "file": "initial_model.npz", "required": False},
            {"name": "dx", "label": "Grid spacing X (m)", "type": "float", "default": 10.0, "required": False},
            {"name": "dy", "label": "Grid spacing Y (m)", "type": "float", "default": 10.0, "required": False},
            {"name": "dz", "label": "Grid spacing Z (m)", "type": "float", "default": 5.0, "required": False},
            {"name": "max_iter", "label": "Max iterations", "type": "int", "default": 20, "required": False},
        ],
        "outputs": [
            {"name": "model_3d", "label": "Inverted 3D model", "type": "ndarray_3d"},
            {"name": "misfit", "label": "RMS misfit", "type": "float"},
        ],
    },
}


# ── Algorithm.py templates ────────────────────────────────────────────

def _make_algo_source(class_name: str, display_name: str, template: str | None) -> str:
    if template == "1d-preprocess":
        return f'''"""{display_name}"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class {class_name}(BaseAlgorithm):
    def run(self, **inputs) -> AlgorithmResult:
        # Required input (loaded from raw/input.csv)
        curve = inputs["curve"]

        # Optional parameters (with defaults from manifest)
        window_size = int(inputs.get("window_size", 11))
        threshold = float(inputs.get("threshold", 0.5))

        warnings = []

        # TODO: replace with actual algorithm
        self.report_progress(0.3, "Processing...")
        curve_filtered = curve.copy()
        quality_metric = 1.0

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={{
                "curve_filtered": curve_filtered,
                "quality_metric": quality_metric,
            }},
            warnings=warnings,
        )
'''

    if template == "2d-forward":
        return f'''"""{display_name}"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class {class_name}(BaseAlgorithm):
    def run(self, **inputs) -> AlgorithmResult:
        model = inputs["model"]
        depths = inputs["depths"]
        n_points = int(inputs.get("n_points", 100))
        frequency = float(inputs.get("frequency", 1.0))

        warnings = []

        # TODO: replace with actual forward modeling
        self.report_progress(0.5, "Computing forward response...")
        response = np.zeros((n_points, n_points))
        coordinates = np.zeros((n_points, 2))

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={{
                "response": response,
                "coordinates": coordinates,
            }},
            warnings=warnings,
        )
'''

    if template == "3d-inversion":
        return f'''"""{display_name}"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class {class_name}(BaseAlgorithm):
    def run(self, **inputs) -> AlgorithmResult:
        observed_data = inputs["observed_data"]
        initial_model = inputs.get("initial_model")
        dx = float(inputs.get("dx", 10.0))
        dy = float(inputs.get("dy", 10.0))
        dz = float(inputs.get("dz", 5.0))
        max_iter = int(inputs.get("max_iter", 20))

        warnings = []

        # TODO: replace with actual inversion
        nx, ny, nz = 10, 10, 10
        if initial_model is not None:
            model_3d = np.asarray(initial_model, dtype=float)
        else:
            model_3d = np.ones((nx, ny, nz)) * 100.0

        for i in range(max_iter):
            self.report_progress((i + 1) / max_iter, f"Iteration {{i+1}}/{{max_iter}}")
            # TODO: inversion iteration

        misfit = 0.0
        return AlgorithmResult(
            outputs={{
                "model_3d": model_3d,
                "misfit": misfit,
            }},
            warnings=warnings,
        )
'''

    # Default: minimal template
    return f'''"""{display_name}

TODO: implement algorithm logic.
See manifest.json for input/output specification.
"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class {class_name}(BaseAlgorithm):
    def run(self, **inputs) -> AlgorithmResult:
        # Access inputs defined in manifest.json:
        #   data = inputs["input_name"]           # required input
        #   param = inputs.get("param", default)  # optional with default
        #
        # Report progress for long-running tasks:
        #   self.report_progress(0.5, "Halfway done...")
        #
        # Add non-fatal warnings:
        #   warnings.append("Some condition detected")

        warnings = []

        # TODO: implement {display_name}
        raise NotImplementedError("{display_name} not implemented")
'''


def to_pascal_case(snake: str) -> str:
    """Convert snake_case name to PascalCase."""
    return "".join(part.capitalize() for part in snake.split("_"))


def scaffold(name: str, display_name: str, method: str, category: str,
             dimension: str, template: str | None) -> None:
    algo_dir = _ALGO_ROOT / name
    if algo_dir.exists():
        print(f"[error] {algo_dir} already exists", file=sys.stderr)
        sys.exit(1)

    algo_dir.mkdir(parents=True)
    class_name = to_pascal_case(name)

    # Build manifest
    if template and template in _TEMPLATE_IO:
        tmpl = _TEMPLATE_IO[template]
        inputs = tmpl["inputs"]
        outputs = tmpl["outputs"]
        dimension = tmpl.get("dimension", dimension)
    else:
        inputs = []
        outputs = []

    manifest = {
        "name": name,
        "display_name": display_name,
        "version": "1.0.0",
        "method": method,
        "category": category,
        "dimension": dimension,
        "pipeline_role": "step",
        "entry": {"type": "python", "module": "algorithm", "class": class_name},
        "inputs": inputs,
        "outputs": outputs,
    }

    (algo_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (algo_dir / "algorithm.py").write_text(
        _make_algo_source(class_name, display_name, template),
        encoding="utf-8",
    )

    print(f"[done] Algorithm package created at {algo_dir}")
    if template:
        print(f"       Template: {template} (inputs/outputs pre-filled)")
    else:
        print("       Note: inputs/outputs are empty. Fill manifest.json before use.")
    print(f"       Validate: python tools/validate_algo.py {algo_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a new algorithm package",
        epilog="Templates: " + ", ".join(TEMPLATES),
    )
    parser.add_argument("--name", required=True, help="Algorithm package name (snake_case)")
    parser.add_argument("--display-name", required=True, help="Human-readable display name")
    parser.add_argument("--method", required=True, choices=METHODS, help="Geophysical method")
    parser.add_argument("--category", required=True, choices=CATEGORIES, help="Algorithm category")
    parser.add_argument("--dimension", default="2d", help="Dimension (default: 2d)")
    parser.add_argument("--template", choices=TEMPLATES, help="Pre-filled template")
    args = parser.parse_args()

    scaffold(args.name, args.display_name, args.method, args.category, args.dimension, args.template)


if __name__ == "__main__":
    main()
