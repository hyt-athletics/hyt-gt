"""测井曲线 Savitzky-Golay 平滑滤波。

对任意测井曲线（GR、电阻率、密度等）应用 Savitzky-Golay 多项式平滑滤波，
在抑制高频噪声的同时较好地保留曲线峰值形态。

输入/输出规范见 manifest.json。
"""
import math

from algorithms.base import AlgorithmResult, BaseAlgorithm


class LogPreprocSavgol(BaseAlgorithm):
    def run(self, **inputs):
        from scipy.signal import savgol_filter

        curve = inputs["curve"]
        window_length = int(inputs.get("window_length", 11))
        polyorder = int(inputs.get("polyorder", 3))

        # 参数校验
        if window_length % 2 == 0:
            window_length += 1  # 强制为奇数
        if polyorder >= window_length:
            raise ValueError(
                f"polyorder ({polyorder}) 须小于 window_length ({window_length})。"
                f"建议设 polyorder={max(2, window_length - 2)}。"
            )
        if len(curve) < window_length:
            raise ValueError(
                f"曲线长度 ({len(curve)}) 小于窗口长度 ({window_length})。"
                f"请减小 window_length 或提供更长的曲线。"
            )

        curve_filtered = savgol_filter(curve, window_length, polyorder)

        # 信噪比改善量：10*log10(var(原始) / var(残差))
        residual_var = float(((curve - curve_filtered) ** 2).mean())
        warnings = []
        if residual_var < 1e-12:
            snr_improvement = 0.0
            warnings.append("残差方差接近零，曲线可能已足够平滑，无需滤波。")
        else:
            original_var = float((curve ** 2).mean())
            snr_improvement = 10.0 * math.log10(max(original_var / residual_var, 1e-9))
            if snr_improvement < 0:
                warnings.append(
                    f"SNR 改善量为负 ({snr_improvement:.1f} dB)，滤波参数可能过度平滑了信号，"
                    f"建议减小 window_length 或 polyorder。"
                )

        return AlgorithmResult(
            outputs={
                "curve_filtered": curve_filtered,
                "snr_improvement": snr_improvement,
            },
            warnings=warnings,
        )
