"""基于U-Net的2.5D井地电阻率智能反演

TODO: 实现算法逻辑。
输入/输出规范见 manifest.json。
"""
from algorithms.base import AlgorithmResult, BaseAlgorithm


class ErtInvUnet25d(BaseAlgorithm):
    def run(self, **inputs):
        # TODO: 实现 基于U-Net的2.5D井地电阻率智能反演 算法
        raise NotImplementedError("基于U-Net的2.5D井地电阻率智能反演 尚未实现")
        return AlgorithmResult(outputs={})
