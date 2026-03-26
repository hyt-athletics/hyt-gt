"""三维网格生成工具。

为正演/反演算法提供可复用的网格构建函数。
基于 discretize 库，参考 simpeg_dc_inversion 中的 2D 模式扩展到 3D。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def make_tensor_mesh_3d(
    x_extent: float,
    y_extent: float,
    z_extent: float,
    dx: float,
    dy: float,
    dz: float,
    n_pad: int = 5,
    pad_factor: float = 1.4,
) -> Any:
    """构建带几何级数扩展的三维张量网格。

    核心区域为均匀网格，边界向外以 pad_factor 倍几何扩展 n_pad 层。

    Args:
        x/y/z_extent: 核心区域范围（米）
        dx/dy/dz: 核心区域网格间距（米）
        n_pad: 边界扩展层数
        pad_factor: 扩展倍率

    Returns:
        discretize.TensorMesh（3D）
    """
    try:
        from discretize import TensorMesh
    except ImportError as e:
        raise ImportError(
            "构建三维网格需要安装 discretize: uv pip install geophys-tool[3d]"
        ) from e

    nx = max(1, int(round(x_extent / dx)))
    ny = max(1, int(round(y_extent / dy)))
    nz = max(1, int(round(z_extent / dz)))

    hx = [(dx * pad_factor**n_pad, n_pad, -pad_factor), (dx, nx), (dx * pad_factor**n_pad, n_pad, pad_factor)]
    hy = [(dy * pad_factor**n_pad, n_pad, -pad_factor), (dy, ny), (dy * pad_factor**n_pad, n_pad, pad_factor)]
    hz = [(dz, n_pad, pad_factor), (dz, nz)]

    mesh = TensorMesh([hx, hy, hz], origin="CCN")
    return mesh


def make_borehole_mesh(
    collars: np.ndarray,
    cell_size: float = 5.0,
    depth_extent: float = 500.0,
    padding: float = 100.0,
    n_pad: int = 5,
    pad_factor: float = 1.3,
) -> Any:
    """根据钻孔位置构建自适应三维张量网格。

    以钻孔包围盒为核心区域，向外扩展。

    Args:
        collars: 钻孔孔口坐标，shape (n_boreholes, 3) — [x, y, z]
        cell_size: 核心区域网格间距（米）
        depth_extent: 深度范围（米，向下为正）
        padding: 核心区域外扩距离（米）
        n_pad: 边界扩展层数
        pad_factor: 扩展倍率

    Returns:
        discretize.TensorMesh（3D）
    """
    collars = np.atleast_2d(collars)
    x_min, y_min = collars[:, 0].min() - padding, collars[:, 1].min() - padding
    x_max, y_max = collars[:, 0].max() + padding, collars[:, 1].max() + padding

    return make_tensor_mesh_3d(
        x_extent=x_max - x_min,
        y_extent=y_max - y_min,
        z_extent=depth_extent,
        dx=cell_size,
        dy=cell_size,
        dz=cell_size,
        n_pad=n_pad,
        pad_factor=pad_factor,
    )


def mesh_to_pyvista(mesh: Any, models: dict[str, np.ndarray] | None = None) -> Any:
    """将 discretize TensorMesh 转换为 PyVista 对象，可附带模型值。

    Args:
        mesh: discretize.TensorMesh
        models: 可选，{模型名: 数组}，长度需与 mesh.nC 一致

    Returns:
        pyvista.RectilinearGrid
    """
    try:
        import pyvista  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "转换为 PyVista 需要安装: uv pip install geophys-tool[3d]"
        ) from e

    vtk_obj = mesh.to_vtk(models=models or {})
    return vtk_obj
