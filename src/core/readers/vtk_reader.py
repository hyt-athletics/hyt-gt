"""VTK/VTU/VTS/VTR 格式读取器。

使用 PyVista 解析 VTK 系列文件，将 cell/point 数据提取为 numpy 数组。
需要安装 3D 可选依赖：uv pip install geophys-tool[3d]
"""
from pathlib import Path

import numpy as np

from .base import BaseReader


class VtkReader(BaseReader):
    """读取 VTK 系列文件（.vtk / .vtu / .vts / .vtr）。"""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".vtk", ".vtu", ".vts", ".vtr")

    def read(self, path: Path) -> dict[str, np.ndarray]:
        try:
            import pyvista
        except ImportError as e:
            raise ImportError(
                "读取 VTK 文件需要安装 pyvista: uv pip install geophys-tool[3d]"
            ) from e

        dataset = pyvista.read(str(path))
        result: dict[str, np.ndarray] = {}

        # cell data（如反演结果中的模型值）
        for name in dataset.cell_data:
            arr = np.asarray(dataset.cell_data[name], dtype=np.float64)
            result[f"cell_{name}"] = arr

        # point data（如节点上的场值）
        for name in dataset.point_data:
            arr = np.asarray(dataset.point_data[name], dtype=np.float64)
            result[f"point_{name}"] = arr

        if not result:
            result["points"] = np.asarray(dataset.points, dtype=np.float64)

        return result
