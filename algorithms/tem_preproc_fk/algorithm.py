"""TEM F-K滤波去噪

TODO: 实现算法逻辑。
输入/输出规范见 manifest.json。
"""
from algorithms.base import AlgorithmResult, BaseAlgorithm


class TemPreprocFk(BaseAlgorithm):
    def run(self, **inputs):
        # TODO: 实现 TEM F-K滤波去噪 算法
        raise NotImplementedError("TEM F-K滤波去噪 尚未实现")
        return AlgorithmResult(outputs={})
