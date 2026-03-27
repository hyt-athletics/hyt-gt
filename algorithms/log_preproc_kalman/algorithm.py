"""测井 Kalman-RTS 平滑滤波去噪。

对测井曲线应用一维 Kalman 滤波 + Rauch-Tung-Striebel (RTS) 后向平滑：
- 状态模型：x[k+1] = x[k] + w[k]，w ~ N(0, Q)（随机游走）
- 观测模型：z[k] = x[k] + v[k]，v ~ N(0, R)
- Q/R 比值控制平滑程度：Q/R 越小平滑越强
- RTS 后向平滑保证因果性且利用了全局信息

输入/输出规范见 manifest.json。
"""
import math

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class LogPreprocKalman(BaseAlgorithm):
    def run(self, **inputs):
        curve = np.asarray(inputs["curve"], dtype=float)
        Q = float(inputs.get("process_noise", 0.01))
        R = float(inputs.get("measurement_noise", 1.0))
        n = len(curve)

        # ── 前向 Kalman 滤波 ─────────────────────────────────────────
        x_f = np.empty(n)   # 滤波均值
        p_f = np.empty(n)   # 滤波方差
        x_p = np.empty(n)   # 预测均值
        p_p = np.empty(n)   # 预测方差

        x_f[0] = curve[0]
        p_f[0] = R

        for k in range(1, n):
            # 预测
            x_p[k] = x_f[k - 1]
            p_p[k] = p_f[k - 1] + Q
            # 更新
            K = p_p[k] / (p_p[k] + R)
            x_f[k] = x_p[k] + K * (curve[k] - x_p[k])
            p_f[k] = (1.0 - K) * p_p[k]

        # ── 后向 RTS 平滑 ────────────────────────────────────────────
        x_s = x_f.copy()
        p_s = p_f.copy()

        for k in range(n - 2, -1, -1):
            G = p_f[k] / p_p[k + 1]
            x_s[k] = x_f[k] + G * (x_s[k + 1] - x_p[k + 1])
            p_s[k] = p_f[k] + G ** 2 * (p_s[k + 1] - p_p[k + 1])

        # ── SNR 改善量 ───────────────────────────────────────────────
        resid_var = float(((curve - x_s) ** 2).mean())
        orig_var = float((curve ** 2).mean())
        warnings = []
        if resid_var < 1e-12:
            snr = 0.0
            warnings.append("残差方差接近零，曲线可能已足够平滑。")
        else:
            snr = 10.0 * math.log10(max(orig_var / resid_var, 1e-9))
            if snr < 0:
                warnings.append(
                    f"SNR 改善量为负 ({snr:.1f} dB)，process_noise 过小，曲线过度平滑，"
                    "建议适当增大 process_noise。"
                )

        return AlgorithmResult(
            outputs={"curve_filtered": x_s, "snr_improvement": snr},
            warnings=warnings,
        )
