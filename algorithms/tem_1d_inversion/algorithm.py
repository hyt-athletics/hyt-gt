"""TEM 一维 Occam 反演。

使用 empymod 作为正演引擎，通过 L-BFGS-B 优化求解 Occam 平滑约束反演问题：

    Φ(m) = ‖log F(m) − log d_obs‖² + λ ‖Lm‖²

其中：
  - m = log₁₀(ρ_i)  各层电阻率参数（对数空间）
  - F(m)             empymod 正演函数
  - d_obs            观测 dB/dt 曲线
  - L                一阶差分粗糙度矩阵（Occam 约束）
  - λ                正则化系数

输入/输出规范见 manifest.json。
"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm

_MU0 = 4e-7 * np.pi


class TEM1DInversion(BaseAlgorithm):
    def run(
        self,
        emf: np.ndarray,
        times: np.ndarray,
        n_layers: int = 20,
        lambda_: float = 1e-3,
        depth_max: float = 500.0,
        loop_radius: float = 50.0,
    ) -> AlgorithmResult:
        from scipy.optimize import minimize

        emf_2d = np.atleast_2d(np.asarray(emf, dtype=float))
        times = np.asarray(times, dtype=float).ravel()
        n_t = len(times)

        if emf_2d.shape[-1] != n_t:
            raise ValueError(
                f"emf 列数 ({emf_2d.shape[-1]}) 与 times 长度 ({n_t}) 不一致。"
            )

        # 对多深度点取均值作为单道数据
        emf_obs = np.abs(emf_2d).mean(axis=0)
        d_obs = np.log10(np.maximum(emf_obs, 1e-30))

        # 对数均匀分层
        # layer_tops[0]=0 为地面，layer_tops[1:] 为各层界面
        # empymod 约定：depth 数组含空气-地表界面（0 m）共 n 个值；
        #               resistivity 数组为 n+1 个值（含空气层 2e14）
        n = int(n_layers)
        layer_tops = np.logspace(0, np.log10(float(depth_max)), n)
        layer_tops[0] = 0.0   # 地面作为第一个界面

        # 一阶差分粗糙度矩阵
        L = np.zeros((n - 1, n))
        for i in range(n - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        lam = float(lambda_)

        # 初始模型：100 Ω·m 均匀半空间
        m0 = np.full(n, np.log10(100.0))
        r = float(loop_radius)
        dep_interfaces = layer_tops  # n 个界面（含地面 0）

        def forward(m_log10):
            """正演：返回 log10(|dB/dt|)"""
            import empymod
            # rho: n+1 个值（空气层 + n 层）；depth: n 个界面 → 满足 len(rho)=len(depth)+1
            rho = np.concatenate([[2e14], 10.0 ** m_log10])
            try:
                out = empymod.dipole(
                    src=[0.0, 0.0, 0.001],
                    rec=[r, 0.0, 0.0],
                    depth=dep_interfaces,
                    res=rho,
                    freqtime=times,
                    signal=-1,
                    msrc=True,
                    mrec=True,
                    verb=0,
                )
                return np.log10(np.maximum(np.abs(out.real), 1e-30))
            except Exception as exc:
                return np.full(n_t, -30.0)

        def objective(m):
            d_pred = forward(m)
            data = np.sum((d_obs - d_pred) ** 2)
            reg = lam * np.sum((L @ m) ** 2)
            return data + reg

        result = minimize(
            objective,
            m0,
            method="L-BFGS-B",
            bounds=[(-2.0, 6.0)] * n,   # 0.01 – 1e6 Ω·m
            options={"maxiter": 15, "ftol": 1e-8},
        )

        resistivity = 10.0 ** result.x

        d_final = forward(result.x)
        misfit = float(np.sqrt(np.mean((d_obs - d_final) ** 2)))

        warnings = []
        if misfit > 0.3:
            warnings.append(
                f"对数 RMS 拟合误差偏大 ({misfit:.3f})，建议检查数据质量或调整 loop_radius / lambda_。"
            )

        return AlgorithmResult(
            outputs={
                "resistivity": resistivity,
                "layer_tops": layer_tops,
                "misfit": misfit,
            },
            warnings=warnings,
        )
