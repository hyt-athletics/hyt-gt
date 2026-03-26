"""Algorithm package validator — pre-submission checks.

Validates manifest.json completeness, class existence, type legality,
and input/output consistency.

Usage:
    python tools/validate_algo.py algorithms/my_algo/
    python tools/validate_algo.py --all
"""
import argparse
import ast
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ALGO_ROOT = _ROOT / "algorithms"

VALID_METHODS = {"ert", "ip", "em", "tem", "logging"}
VALID_CATEGORIES = {"preprocess", "process", "forward", "inversion"}
VALID_TYPES = {"ndarray_1d", "ndarray_2d", "ndarray_3d", "dataframe", "int", "float", "str", "mesh"}
ARRAY_TYPES = {"ndarray_1d", "ndarray_2d", "ndarray_3d", "dataframe", "mesh"}
SCALAR_TYPES = {"int", "float", "str"}

REQUIRED_MANIFEST_FIELDS = {"name", "display_name", "version", "method", "category", "dimension", "entry"}
REQUIRED_ENTRY_FIELDS = {"type", "module", "class"}


class _Report:
    def __init__(self, algo_dir: Path):
        self.algo_dir = algo_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.ok: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def passed(self, msg: str) -> None:
        self.ok.append(msg)

    def print(self) -> None:
        name = self.algo_dir.name
        if self.errors:
            status = "FAIL"
        elif self.warnings:
            status = "WARN"
        else:
            status = "PASS"
        print(f"\n{'='*60}")
        print(f"  {name}  [{status}]")
        print(f"{'='*60}")
        for msg in self.ok:
            print(f"  [OK]    {msg}")
        for msg in self.warnings:
            print(f"  [WARN]  {msg}")
        for msg in self.errors:
            print(f"  [ERROR] {msg}")

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


def validate(algo_dir: Path) -> _Report:
    report = _Report(algo_dir)

    # -- manifest.json existence
    manifest_path = algo_dir / "manifest.json"
    if not manifest_path.exists():
        report.error("manifest.json not found")
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        report.error(f"manifest.json is not valid JSON: {e}")
        return report
    report.passed("manifest.json is valid JSON")

    # -- required fields
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    if missing:
        report.error(f"Missing required fields: {', '.join(sorted(missing))}")
    else:
        report.passed("All required fields present")

    # -- name matches directory
    if manifest.get("name") != algo_dir.name:
        report.error(f"name '{manifest.get('name')}' does not match directory '{algo_dir.name}'")
    else:
        report.passed("name matches directory name")

    # -- method and category
    method = manifest.get("method", "")
    if method not in VALID_METHODS:
        report.error(f"Invalid method '{method}'. Must be one of: {', '.join(sorted(VALID_METHODS))}")
    else:
        report.passed(f"method '{method}' is valid")

    category = manifest.get("category", "")
    if category not in VALID_CATEGORIES:
        report.error(f"Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}")
    else:
        report.passed(f"category '{category}' is valid")

    # -- entry block
    entry = manifest.get("entry", {})
    entry_missing = REQUIRED_ENTRY_FIELDS - set(entry.keys())
    if entry_missing:
        report.error(f"entry missing fields: {', '.join(sorted(entry_missing))}")
    else:
        report.passed("entry block complete")

    # -- algorithm.py existence
    module_name = entry.get("module", "algorithm")
    py_path = algo_dir / f"{module_name}.py"
    compiled_paths = list(algo_dir.glob(f"{module_name}.*.so")) + list(algo_dir.glob(f"{module_name}.*.pyd"))

    if not py_path.exists() and not compiled_paths:
        report.error(f"{module_name}.py (or compiled .so/.pyd) not found")
    else:
        report.passed(f"Module file found: {py_path.name if py_path.exists() else compiled_paths[0].name}")

    # -- class exists in source (only check .py)
    class_name = entry.get("class", "")
    if py_path.exists() and class_name:
        source = py_path.read_text(encoding="utf-8")
        if f"class {class_name}" not in source:
            report.error(f"Class '{class_name}' not found in {module_name}.py")
        else:
            report.passed(f"Class '{class_name}' found in source")

        # check BaseAlgorithm inheritance
        if "BaseAlgorithm" not in source:
            report.warn(f"{module_name}.py does not reference BaseAlgorithm")

        # check for dead code after NotImplementedError
        if "raise NotImplementedError" in source:
            lines = source.split("\n")
            for i, line in enumerate(lines):
                if "raise NotImplementedError" in line:
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("return"):
                        report.warn("Dead code: 'return' after 'raise NotImplementedError' (unreachable)")

    # -- inputs validation
    inputs = manifest.get("inputs", [])
    if not inputs:
        report.warn("inputs is empty — algorithm will not appear in GUI")
    else:
        for inp in inputs:
            inp_name = inp.get("name", "<unnamed>")
            inp_type = inp.get("type", "")
            if not inp_name or inp_name == "<unnamed>":
                report.error("Input missing 'name' field")
            if inp_type not in VALID_TYPES:
                report.error(f"Input '{inp_name}' has invalid type '{inp_type}'")
            elif inp_type in ARRAY_TYPES and "file" not in inp:
                if inp.get("required", True):
                    report.warn(f"Input '{inp_name}' (type={inp_type}) has no 'file' field — runner may not find data")
            elif inp_type in SCALAR_TYPES and "default" not in inp:
                if not inp.get("required", True):
                    report.warn(f"Optional input '{inp_name}' (type={inp_type}) has no 'default' — will use None")
        report.passed(f"{len(inputs)} inputs defined")

    # -- outputs validation
    outputs = manifest.get("outputs", [])
    if not outputs:
        report.warn("outputs is empty — algorithm will not appear in GUI")
    else:
        for out in outputs:
            out_name = out.get("name", "<unnamed>")
            out_type = out.get("type", "")
            if not out_name or out_name == "<unnamed>":
                report.error("Output missing 'name' field")
            if out_type not in VALID_TYPES:
                report.error(f"Output '{out_name}' has invalid type '{out_type}'")
        report.passed(f"{len(outputs)} outputs defined")

    # -- check output names match AlgorithmResult keys (static analysis)
    if py_path.exists() and outputs:
        _check_output_keys(py_path, [o["name"] for o in outputs], report)

    return report


def _check_output_keys(py_path: Path, expected_keys: list[str], report: _Report) -> None:
    """Static check: output keys in AlgorithmResult match manifest."""
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except SyntaxError:
        report.warn("Could not parse algorithm.py for output key check (syntax error)")
        return

    # Find AlgorithmResult(...) calls and extract output dict keys
    found_keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name == "AlgorithmResult":
                for kw in node.keywords:
                    if kw.arg == "outputs" and isinstance(kw.value, ast.Dict):
                        for key in kw.value.keys:
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                found_keys.add(key.value)

    if not found_keys:
        report.warn("Could not extract output keys from AlgorithmResult call (may use dynamic dict)")
        return

    missing = set(expected_keys) - found_keys
    extra = found_keys - set(expected_keys)
    if missing:
        report.error(f"Manifest outputs not in AlgorithmResult: {', '.join(sorted(missing))}")
    if extra:
        report.warn(f"AlgorithmResult has keys not in manifest outputs: {', '.join(sorted(extra))}")
    if not missing and not extra:
        report.passed("Output keys match between manifest and AlgorithmResult")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate algorithm packages")
    parser.add_argument("path", nargs="?", help="Path to algorithm directory")
    parser.add_argument("--all", action="store_true", help="Validate all algorithms")
    args = parser.parse_args()

    if not args.all and not args.path:
        parser.error("Provide an algorithm path or use --all")

    if args.all:
        dirs = sorted(d for d in _ALGO_ROOT.iterdir() if d.is_dir() and not d.name.startswith("."))
    else:
        dirs = [Path(args.path).resolve()]

    total_errors = 0
    total_warnings = 0
    for d in dirs:
        report = validate(d)
        report.print()
        total_errors += len(report.errors)
        total_warnings += len(report.warnings)

    print(f"\n{'='*60}")
    print(f"  Summary: {len(dirs)} packages, {total_errors} errors, {total_warnings} warnings")
    print(f"{'='*60}")
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
