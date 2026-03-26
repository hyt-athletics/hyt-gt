"""empymod TEM 1D 正演算法。

使用 empymod 计算水平层状介质中的瞬变电磁响应（地面回线或井中装置）。
empymod 支持从 DC 到 GPR 的全频段 EM 正演，采用快速 Hankel 变换。

参考：
  empymod 文档: https://empymod.emsig.xyz
  Werthmuller (2017), Geophysics, doi:10.1190/geo2016-0626.1
"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class EmpymodTEMForward(BaseAlgorithm):
    """empymod 1D TEM 正演：水平层状大地，回线源或磁偶极子源。"""

    def run(
        self,
        resistivity: np.ndarray,
        depths: np.ndarray,
        src_depth: float = 0.0,
        rec_depth: float = 0.0,
        loop_radius: float = 50.0,
        t_min: float = -5.0,
        t_max: float = -2.0,
        n_times: int = 30,
    ) -> AlgorithmResult:
        import empymod

        resistivity = np.asarray(resistivity, dtype=float).ravel()
        depths_arr = np.asarray(depths, dtype=float).ravel()

        # empymod 要求层数 = len(resistivity)
        # depths 定义为每层顶部深度（不含第一层，从0开始）
        # 加入地表（深度0）
        if depths_arr[0] != 0.0:
            depths_arr = np.concatenate([[0.0], depths_arr])
        # 保证层数匹配
        if len(depths_arr) != len(resistivity) - 1:
            raise ValueError(
                f"depths 长度({len(depths_arr)})应比 resistivity 层数({len(resistivity)})少1。"
                f"\n第一层为空气层，depths 从地面以下界面开始。"
            )

        times = np.logspace(t_min, t_max, n_times)

        # 发射源：垂直磁偶极子（模拟回线源，偏移量 loop_radius 处等效）
        # empymod.dipole: src/rec 为 [x, y, z]，msrc/mrec=True 表示磁偶极子
        src = [0.0, 0.0, float(src_depth)]
        rec = [float(loop_radius), 0.0, float(rec_depth)]

        # 时域正演（signal=-1: 阶跃断开，对应 TEM 关断后衰减）
        # src/rec 为 [x, y, z] 三元素坐标；msrc/mrec=True 指定磁偶极子
        emf_raw = empymod.dipole(
            src=src,
            rec=rec,
            depth=depths_arr,
            res=resistivity,
            freqtime=times,
            signal=-1,
            msrc=True,   # 磁偶极子发射
            mrec=True,   # 磁场接收
            verb=0,
        )

        emf = np.abs(emf_raw.real)
        # 计算视电阻率（均匀半空间近似公式，仅供参考）
        mu0 = 4 * np.pi * 1e-7
        rho_app = self._apparent_resistivity(emf, times, loop_radius, mu0)

        warnings = []
        if np.any(emf <= 0):
            warnings.append("部分时道 EMF 为负值或零，视电阻率计算可能不准确。")

        return AlgorithmResult(
            outputs={"emf": emf, "times": times, "rho_app": rho_app},
            warnings=warnings,
        )

    @staticmethod
    def _apparent_resistivity(
        dbdt: np.ndarray,
        times: np.ndarray,
        radius: float,
        mu0: float,
    ) -> np.ndarray:
        """中心回线装置视电阻率近似公式（Nabighian & Macnae, 1991）。"""
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (mu0 / (20 * np.pi)) ** (2 / 3) * (
                (radius**2 * mu0) / (times * dbdt)
            ) ** (2 / 3)
        return np.where(np.isfinite(rho) & (rho > 0), rho, np.nan)
