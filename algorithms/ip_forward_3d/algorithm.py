"""三维井-地IP正演

输入/输出规范见 manifest.json。
"""
import warnings

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm
from core.mesh_utils import make_tensor_mesh_3d


class IpForward3d(BaseAlgorithm):
    """三维井-地激发极化正演。"""

    def run(
        self,
        resistivity: np.ndarray,
        chargeability: np.ndarray,
        electrode_positions: np.ndarray,
        dx: float = 10.0,
        dy: float = 10.0,
        dz: float = 5.0,
    ) -> AlgorithmResult:
        resistivity = np.asarray(resistivity, dtype=float)
        chargeability = np.asarray(chargeability, dtype=float)
        electrode_positions = np.asarray(electrode_positions, dtype=float)

        # ── 构建三维张量网格 ──────────────────────────────────────
        x_extent = float(electrode_positions[:, 0].ptp()) + 4 * dx
        y_extent = float(electrode_positions[:, 1].ptp()) + 4 * dy
        z_extent = float(np.abs(electrode_positions[:, 2]).max()) + 4 * dz

        mesh = make_tensor_mesh_3d(
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
            dx=dx, dy=dy, dz=dz,
        )

        # TODO: 实现三维 IP 正演
        #   1. 将 resistivity / chargeability 映射到 mesh 单元
        #   2. 构建 DC + IP 正演算子
        #   3. 计算各电极排列的视电阻率和视极化率

        n_data = electrode_positions.shape[0] // 4  # 粗估数据量
        n_data = max(n_data, 1)
        warnings.warn("算法尚未完整实现，返回占位数据")
        apparent_resistivity = np.ones(n_data, dtype=float) * np.mean(resistivity)
        apparent_chargeability = np.ones(n_data, dtype=float) * np.mean(chargeability)

        return AlgorithmResult(
            outputs={
                "apparent_resistivity": apparent_resistivity,
                "apparent_chargeability": apparent_chargeability,
                "mesh_file": mesh,
            },
        )
