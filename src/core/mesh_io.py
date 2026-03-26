"""网格对象的序列化与反序列化。

支持三种格式：
- discretize mesh → JSON（mesh.save / mesh.load）
- PyVista dataset → VTU（dataset.save / pyvista.read）
- 3D numpy array → NPZ（np.savez_compressed / np.load）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MeshWithModel:
    """容器：网格 + 关联的模型值（电阻率、电导率等）。"""
    mesh: Any
    models: dict[str, np.ndarray] = field(default_factory=dict)


def save_mesh(data: Any, out_dir: Path, name: str) -> Path:
    """将网格/3D数组保存到文件，返回保存路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, MeshWithModel):
        return _save_mesh_with_model(data, out_dir, name)

    if isinstance(data, np.ndarray) and data.ndim >= 3:
        out_path = out_dir / f"{name}.npz"
        np.savez_compressed(out_path, data=data)
        return out_path

    # discretize mesh
    if _is_discretize_mesh(data):
        out_path = out_dir / f"{name}.json"
        data.save(str(out_path))
        return out_path

    # PyVista dataset
    if _is_pyvista_dataset(data):
        out_path = out_dir / f"{name}.vtu"
        data.save(str(out_path))
        return out_path

    raise TypeError(
        f"不支持的网格类型: {type(data).__name__}。"
        f"支持: MeshWithModel, np.ndarray(3D+), discretize.BaseMesh, pyvista.DataSet"
    )


def load_mesh(path: Path, fmt: str = "auto") -> Any:
    """从文件加载网格对象。

    Args:
        path: 文件路径
        fmt: 格式提示 — "auto"(按扩展名)、"discretize"、"vtk"、"npz"
    """
    if not path.exists():
        raise FileNotFoundError(f"网格文件不存在: {path}")

    suffix = path.suffix.lower()

    if fmt == "auto":
        if suffix == ".npz":
            fmt = "npz"
        elif suffix == ".json":
            fmt = "discretize"
        elif suffix in (".vtk", ".vtu", ".vts", ".vtr"):
            fmt = "vtk"
        else:
            raise ValueError(
                f"无法自动推断格式: {suffix}。请通过 manifest 指定 format 字段。"
            )

    if fmt == "npz":
        return np.load(path)["data"]

    if fmt == "discretize":
        try:
            import json as _json
            from discretize import TensorMesh
        except ImportError as e:
            raise ImportError(
                "加载 discretize 网格需要安装 3D 依赖: uv pip install geophys-tool[3d]"
            ) from e
        data = _json.loads(path.read_text(encoding="utf-8"))
        return TensorMesh.deserialize(data)

    if fmt == "vtk":
        try:
            import pyvista
        except ImportError as e:
            raise ImportError(
                "加载 VTK 文件需要安装 3D 依赖: uv pip install geophys-tool[3d]"
            ) from e
        return pyvista.read(str(path))

    raise ValueError(f"未知格式: {fmt}。支持: auto, npz, discretize, vtk")


def _save_mesh_with_model(data: MeshWithModel, out_dir: Path, name: str) -> Path:
    """保存 MeshWithModel：网格导出为 VTU（附带模型属性）。"""
    mesh = data.mesh

    if _is_discretize_mesh(mesh):
        try:
            vtk_obj = mesh.to_vtk(models=data.models)
            out_path = out_dir / f"{name}.vtu"
            vtk_obj.save(str(out_path))
            return out_path
        except Exception:
            out_path = out_dir / f"{name}.json"
            mesh.save(str(out_path))
            for model_name, arr in data.models.items():
                np.savez_compressed(out_dir / f"{name}_{model_name}.npz", data=arr)
            return out_path

    if _is_pyvista_dataset(mesh):
        for model_name, arr in data.models.items():
            mesh[model_name] = arr
        out_path = out_dir / f"{name}.vtu"
        mesh.save(str(out_path))
        return out_path

    raise TypeError(f"MeshWithModel 中的 mesh 类型不支持: {type(mesh).__name__}")


def _is_discretize_mesh(obj: Any) -> bool:
    try:
        from discretize.base.base_mesh import BaseMesh
        return isinstance(obj, BaseMesh)
    except ImportError:
        return False


def _is_pyvista_dataset(obj: Any) -> bool:
    try:
        import pyvista
        return isinstance(obj, pyvista.DataSet)
    except ImportError:
        return False
