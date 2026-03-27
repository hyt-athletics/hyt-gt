"""三维地-井瞬变电磁正演

输入/输出规范见 manifest.json。
"""
import warnings

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm
from core.mesh_utils import make_tensor_mesh_3d


class TemForward3d(BaseAlgorithm):
    """三维地-井瞬变电磁正演（时间域）。"""

    def run(
        self,
        conductivity: np.ndarray,
        tx_locations: np.ndarray,
        rx_locations: np.ndarray,
        t_min: float = 1e-5,
        t_max: float = 1e-2,
        n_times: int = 30,
        dx: float = 10.0,
        dy: float = 10.0,
        dz: float = 5.0,
    ) -> AlgorithmResult:
        conductivity = np.asarray(conductivity, dtype=float)
        tx_locations = np.asarray(tx_locations, dtype=float)
        rx_locations = np.asarray(rx_locations, dtype=float)

        # ── 构建三维张量网格 ──────────────────────────────────────
        all_pts = np.vstack([tx_locations[:, :3], rx_locations[:, :3]])
        x_extent = float(all_pts[:, 0].ptp()) + 4 * dx
        y_extent = float(all_pts[:, 1].ptp()) + 4 * dy
        z_extent = float(np.abs(all_pts[:, 2]).max()) + 4 * dz

        mesh = make_tensor_mesh_3d(
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
            dx=dx, dy=dy, dz=dz,
        )

        # ── 生成时间通道 ─────────────────────────────────────────
        times = np.geomspace(t_min, t_max, n_times)

        # TODO: 实现三维时间域电磁正演
        #   1. 将 conductivity 映射到 mesh 单元
        #   2. 构建 TDEM 正演算子（如 SimPEG TDEM / 自研 FDTD）
        #   3. 对每个发射-接收对在各时间通道计算响应

        n_rx = rx_locations.shape[0]
        warnings.warn("算法尚未完整实现，返回占位数据")
        response = np.zeros((n_rx, n_times), dtype=float)

        return AlgorithmResult(
            outputs={
                "response": response,
                "times": times,
                "mesh_file": mesh,
            },
        )
