import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import AlgorithmResult
from .registry import AlgorithmRegistry


def _is_mesh_like(obj: Any) -> bool:
    """检测对象是否为网格类型（discretize / pyvista / MeshWithModel）。"""
    from src.core.mesh_io import MeshWithModel, _is_discretize_mesh, _is_pyvista_dataset
    return isinstance(obj, MeshWithModel) or _is_discretize_mesh(obj) or _is_pyvista_dataset(obj)


def _coerce(value: str, typ: str) -> Any:
    if typ == "int":
        return int(value)
    if typ == "float":
        return float(value)
    return value


class AlgorithmRunner:
    def __init__(self, registry: AlgorithmRegistry, workspace_dir: Path):
        self._registry = registry
        self.workspace_dir = workspace_dir

    def run(
        self,
        algo_name: str,
        borehole_name: str,
        user_params: dict[str, str],
        file_overrides: dict[str, Path] | None = None,
        progress_cb: Any = None,
    ) -> dict[str, Path]:
        """执行算法，将结果保存为文件。返回 {output_name: saved_file_path}。

        Args:
            file_overrides: 可选，{algo输入名: 文件路径}，优先于 raw/ 目录查找。
            progress_cb: 可选，进度回调 (fraction: float, message: str) -> None。
        """
        manifest = self._registry.get_manifest(algo_name)
        algo = self._registry.load(algo_name)
        if progress_cb:
            algo.set_progress_callback(progress_cb)

        inputs = self._build_inputs(manifest, borehole_name, user_params, file_overrides)
        result: AlgorithmResult = algo.run(**inputs)
        self._validate_result(result, manifest)
        return self._save_results(result, manifest, borehole_name, algo_name)

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _validate_result(self, result: AlgorithmResult, manifest: dict) -> None:
        """校验算法返回值是否符合 manifest 定义。"""
        if not isinstance(result, AlgorithmResult):
            raise TypeError(
                f"算法返回类型错误：期望 AlgorithmResult，实际 {type(result).__name__}"
            )
        _NDIM_MAP = {"ndarray_1d": 1, "ndarray_2d": 2, "ndarray_3d": 3}
        for spec in manifest.get("outputs", []):
            name = spec["name"]
            typ = spec.get("type", "")
            data = result.outputs.get(name)
            if data is None:
                continue
            if typ in _NDIM_MAP and isinstance(data, np.ndarray):
                expected = _NDIM_MAP[typ]
                if data.ndim != expected:
                    raise TypeError(
                        f"输出 '{name}' 维度不匹配：manifest 要求 {typ}（{expected}D），"
                        f"实际 {data.ndim}D shape={data.shape}"
                    )

    def _build_inputs(
        self,
        manifest: dict,
        borehole_name: str,
        user_params: dict[str, str],
        file_overrides: dict[str, Path] | None = None,
    ) -> dict[str, Any]:
        """根据 manifest inputs 定义构建算法输入字典。

        - ndarray_*/dataframe 类型：从工作区原始数据文件读取
        - int/float/str 类型：从 user_params 获取，缺省使用 manifest 默认值
        - file_overrides 中的路径优先于 raw/ 目录查找
        """
        inputs: dict[str, Any] = {}
        method = manifest.get("method", "")
        raw_dir = (
            self.workspace_dir / "boreholes" / borehole_name / method / "raw"
        )
        overrides = file_overrides or {}

        for spec in manifest["inputs"]:
            name = spec["name"]
            typ = spec["type"]

            _FILE_TYPES = ("ndarray_1d", "ndarray_2d", "ndarray_3d", "dataframe", "mesh")
            if typ in _FILE_TYPES:
                if name in overrides:
                    fpath = overrides[name]
                else:
                    default_ext = ".npz" if typ in ("ndarray_3d", "mesh") else ".csv"
                    fname = spec.get("file", f"{name}{default_ext}")
                    fpath = raw_dir / fname
                if not fpath.exists():
                    if spec.get("required", True):
                        raise FileNotFoundError(
                            f"输入文件不存在: {fpath}\n"
                            f"请将 '{fname}' 放入 boreholes/{borehole_name}/{method}/raw/ 目录。"
                        )
                    continue
                if typ == "dataframe":
                    inputs[name] = pd.read_csv(fpath, index_col=0)
                elif typ == "ndarray_3d":
                    inputs[name] = np.load(fpath)["data"]
                elif typ == "mesh":
                    from src.core.mesh_io import load_mesh
                    fmt = spec.get("format", "auto")
                    inputs[name] = load_mesh(fpath, fmt)
                else:
                    inputs[name] = np.loadtxt(fpath, delimiter=",")
            else:
                if name in user_params and user_params[name].strip():
                    inputs[name] = _coerce(user_params[name].strip(), typ)
                elif "default" in spec:
                    inputs[name] = spec["default"]
                elif spec.get("required", True):
                    raise ValueError(
                        f"缺少必填参数 '{name}'（{spec.get('label', name)}）"
                    )

        return inputs

    def _save_results(
        self,
        result: AlgorithmResult,
        manifest: dict,
        borehole_name: str,
        algo_name: str,
    ) -> dict[str, Path]:
        method = manifest.get("method", "")
        out_dir = (
            self.workspace_dir
            / "boreholes"
            / borehole_name
            / method
            / "processed"
            / algo_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[str, Path] = {}
        for spec in manifest["outputs"]:
            name = spec["name"]
            data = result.outputs.get(name)
            if data is None:
                continue

            if isinstance(data, pd.DataFrame):
                out_path = out_dir / f"{name}.csv"
                data.to_csv(out_path)
            elif isinstance(data, np.ndarray) and data.ndim >= 3:
                out_path = out_dir / f"{name}.npz"
                np.savez_compressed(out_path, data=data)
            elif isinstance(data, np.ndarray):
                out_path = out_dir / f"{name}.csv"
                np.savetxt(out_path, data, delimiter=",")
            elif _is_mesh_like(data):
                from src.core.mesh_io import save_mesh
                out_path = save_mesh(data, out_dir, name)
            else:
                out_path = out_dir / f"{name}.txt"
                out_path.write_text(str(data), encoding="utf-8")

            saved[name] = out_path

        # 写运行记录
        run_meta = {
            "algo": algo_name,
            "borehole": borehole_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "outputs": {k: str(v) for k, v in saved.items()},
            "warnings": result.warnings,
        }
        (out_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return saved
