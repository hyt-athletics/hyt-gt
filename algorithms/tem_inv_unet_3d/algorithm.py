"""基于U-Net的三维地-井TEM智能反演

输入/输出规范见 manifest.json。
"""
import warnings

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class TemInvUnet3d(BaseAlgorithm):
    """基于 U-Net 的三维瞬变电磁智能反演。"""

    def run(
        self,
        observed_data: np.ndarray,
        tx_locations: np.ndarray,
        rx_locations: np.ndarray,
        times: np.ndarray,
        model_path: str = "",
    ) -> AlgorithmResult:
        observed_data = np.asarray(observed_data, dtype=float)
        tx_locations = np.asarray(tx_locations, dtype=float)
        rx_locations = np.asarray(rx_locations, dtype=float)
        times = np.asarray(times, dtype=float)

        # TODO: 实现 U-Net 三维 TEM 反演
        #   1. 加载预训练模型（model_path）
        #   2. 将 observed_data 预处理为网络输入张量
        #   3. 前向推理得到三维电导率模型
        #   4. 计算置信度图（如 MC-Dropout 或集成方差）

        # 占位：假定输出为 32x32x16 的体
        nx, ny, nz = 32, 32, 16
        warnings.warn("算法尚未完整实现，返回占位数据")
        recovered_model = np.ones((nx, ny, nz), dtype=float) * 1e-2
        confidence = np.ones((nx, ny, nz), dtype=float) * 0.5

        return AlgorithmResult(
            outputs={
                "recovered_model": recovered_model,
                "confidence": confidence,
            },
        )
