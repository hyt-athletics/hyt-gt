"""三维井-地IP数据预处理

输入/输出规范见 manifest.json。
"""
import warnings

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class IpPreproc3d(BaseAlgorithm):
    """三维井-地 IP 数据预处理（异常值剔除与质量控制）。"""

    def run(
        self,
        raw_data: np.ndarray,
        min_voltage: float = 1e-6,
        max_rho: float = 1e5,
    ) -> AlgorithmResult:
        raw_data = np.asarray(raw_data, dtype=float)

        # TODO: 实现 IP 数据预处理
        #   1. 剔除电压低于 min_voltage 的记录
        #   2. 剔除视电阻率超过 max_rho 的记录
        #   3. 基于统计方法（如 MAD）进一步去除异常值
        #   4. 生成质量控制掩码

        warnings.warn("算法尚未完整实现，返回占位数据")
        mask = np.ones(raw_data.shape[0], dtype=bool)
        cleaned_data = raw_data.copy()

        return AlgorithmResult(
            outputs={
                "cleaned_data": cleaned_data,
                "mask": mask,
            },
        )
