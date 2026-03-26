"""三维TEM有限内存法反演

输入/输出规范见 manifest.json。
"""
import warnings

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm
from core.mesh_utils import make_tensor_mesh_3d


class TemInvLbfgs3d(BaseAlgorithm):
    """三维瞬变电磁 L-BFGS 反演。"""

    def run(
        self,
        observed_data: np.ndarray,
        tx_locations: np.ndarray,
        rx_locations: np.ndarray,
        times: np.ndarray,
        initial_model: np.ndarray | None = None,
        n_layers_x: int = 20,
        n_layers_y: int = 20,
        n_layers_z: int = 10,
        lambda_: float = 1e-2,
        max_iter: int = 20,
    ) -> AlgorithmResult:
        observed_data = np.asarray(observed_data, dtype=float)
        tx_locations = np.asarray(tx_locations, dtype=float)
        rx_locations = np.asarray(rx_locations, dtype=float)
        times = np.asarray(times, dtype=float)

        # ── 推断网格尺寸并构建三维网格 ────────────────────────────
        all_pts = np.vstack([tx_locations[:, :3], rx_locations[:, :3]])
        x_extent = float(all_pts[:, 0].ptp()) or 100.0
        y_extent = float(all_pts[:, 1].ptp()) or 100.0
        z_extent = float(np.abs(all_pts[:, 2]).max()) or 50.0

        dx = x_extent / max(n_layers_x, 1)
        dy = y_extent / max(n_layers_y, 1)
        dz = z_extent / max(n_layers_z, 1)

        mesh = make_tensor_mesh_3d(
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
            dx=dx, dy=dy, dz=dz,
        )

        # TODO: 实现三维 TEM L-BFGS 反演
        #   1. 构建正演算子
        #   2. 定义目标函数（数据拟合 + lambda_ * 正则化）
        #   3. L-BFGS 迭代 max_iter 次
        #   4. 输出恢复的电导率模型

        if initial_model is not None:
            m0 = np.asarray(initial_model, dtype=float)
        else:
            m0 = np.ones((n_layers_x, n_layers_y, n_layers_z), dtype=float) * 1e-2

        warnings.warn("算法尚未完整实现，返回占位数据")
        recovered_model = m0.reshape(n_layers_x, n_layers_y, n_layers_z)
        predicted_data = np.zeros_like(observed_data)
        misfit = float(np.sum((observed_data - predicted_data) ** 2))

        return AlgorithmResult(
            outputs={
                "recovered_model": recovered_model,
                "predicted_data": predicted_data,
                "misfit": misfit,
                "mesh_file": mesh,
            },
        )
