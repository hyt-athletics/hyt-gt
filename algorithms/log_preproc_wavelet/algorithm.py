"""测井小波阈值法去噪。

使用离散小波变换（DWT）的 Donoho-Johnstone 通用阈值（VisuShrink）对测井曲线去噪：
1. 多层 DWT 分解
2. 从最细尺度细节系数用 MAD 估计噪声标准差
3. 对所有细节系数做软阈值处理（保留逼近系数不变）
4. 重构信号

输入/输出规范见 manifest.json。
"""
import math

import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class LogPreprocWavelet(BaseAlgorithm):
    def run(self, **inputs):
        import pywt

        curve = np.asarray(inputs["curve"], dtype=float)
        wavelet = str(inputs.get("wavelet", "db4"))
        level = int(inputs.get("level", 4))

        # 限制分解层数不超过 pywt 允许的最大值
        max_level = pywt.dwt_max_level(len(curve), pywt.Wavelet(wavelet))
        level = min(level, max_level)

        # DWT 分解
        coeffs = pywt.wavedec(curve, wavelet, level=level)

        # MAD 估计最细尺度噪声标准差 → VisuShrink 通用阈值
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * math.sqrt(2.0 * math.log(len(curve)))

        # 对所有细节系数做软阈值（逼近系数保留）
        coeffs_thresh = [coeffs[0]] + [
            pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
        ]

        # 重构（截断到原始长度，防止边界溢出）
        filtered = pywt.waverec(coeffs_thresh, wavelet)[: len(curve)]

        # SNR 改善量
        resid_var = float(((curve - filtered) ** 2).mean())
        orig_var = float((curve ** 2).mean())
        warnings = []
        if resid_var < 1e-12:
            snr = 0.0
            warnings.append("残差方差接近零，曲线可能已足够平滑，无需滤波。")
        else:
            snr = 10.0 * math.log10(max(orig_var / resid_var, 1e-9))
            if snr < 0:
                warnings.append(
                    f"SNR 改善量为负 ({snr:.1f} dB)，threshold 过大导致过度平滑，"
                    "建议减小 level 或换用细节保留更好的小波基（如 sym6）。"
                )

        return AlgorithmResult(
            outputs={"curve_filtered": filtered, "snr_improvement": snr},
            warnings=warnings,
        )
