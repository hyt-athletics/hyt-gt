"""三维井-地IP Occam法反演

输入/输出规范见 manifest.json。
"""
import warnings

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm
from core.mesh_utils import make_tensor_mesh_3d


class IpInvOccam3d(BaseAlgorithm):
    """三维井-地激发极化 Occam 反演。"""

    def run(
        self,
        apparent_resistivity: np.ndarray,
        apparent_chargeability: np.ndarray,
        electrode_positions: np.ndarray,
        initial_resistivity: np.ndarray | None = None,
        dx: float = 10.0,
        dy: float = 10.0,
        dz: float = 5.0,
        alpha_s: float = 1e-3,
        alpha_x: float = 1.0,
        alpha_y: float = 1.0,
        alpha_z: float = 1.0,
        max_iter: int = 20,
    ) -> AlgorithmResult:
        apparent_resistivity = np.asarray(apparent_resistivity, dtype=float).ravel()
        apparent_chargeability = np.asarray(apparent_chargeability, dtype=float).ravel()
        electrode_positions = np.asarray(electrode_positions, dtype=float)

        # ── 构建三维张量网格 ──────────────────────────────────────
        x_extent = float(electrode_positions[:, 0].ptp()) + 4 * dx
        y_extent = float(electrode_positions[:, 1].ptp()) + 4 * dy
        z_extent = float(np.abs(electrode_positions[:, 2]).max()) + 4 * dz

        mesh = make_tensor_mesh_3d(
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
            dx=dx, dy=dy, dz=dz,
        )

        # ── 推断模型维度 ─────────────────────────────────────────
        nx = max(1, int(round(x_extent / dx)))
        ny = max(1, int(round(y_extent / dy)))
        nz = max(1, int(round(z_extent / dz)))

        # TODO: 实现三维 IP Occam 反演
        #   1. 构建 DC + IP 正演算子
        #   2. 初始化电阻率模型（initial_resistivity 或均匀模型）
        #   3. Occam 平滑约束迭代（alpha_s/x/y/z 控制正则化权重）
        #   4. 同时反演电阻率和极化率

        rho_mean = float(np.mean(apparent_resistivity))
        if initial_resistivity is not None:
            rho0 = np.asarray(initial_resistivity, dtype=float)
        else:
            rho0 = np.ones((nx, ny, nz), dtype=float) * rho_mean

        warnings.warn("算法尚未完整实现，返回占位数据")
        resistivity_model = rho0.reshape(nx, ny, nz)
        chargeability_model = np.ones((nx, ny, nz), dtype=float) * float(np.mean(apparent_chargeability))
        predicted_rho = np.ones_like(apparent_resistivity) * rho_mean
        misfit = float(np.sum((apparent_resistivity - predicted_rho) ** 2))

        return AlgorithmResult(
            outputs={
                "resistivity_model": resistivity_model,
                "chargeability_model": chargeability_model,
                "predicted_rho": predicted_rho,
                "misfit": misfit,
                "mesh_file": mesh,
            },
        )
