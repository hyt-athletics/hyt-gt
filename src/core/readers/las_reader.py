from pathlib import Path

import numpy as np

from .base import BaseReader

_DEPTH_MNEMONICS = frozenset({"DEPT", "DEPTH", "MD", "TVD", "TDEP"})


class LasReader(BaseReader):
    """读取 LAS（Log ASCII Standard）测井文件。

    使用 lasio 解析，跳过深度曲线，返回各测井曲线数据。
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".las",)

    def read(self, path: Path) -> dict[str, np.ndarray]:
        try:
            import lasio
        except ImportError as e:
            raise ImportError("请先安装 lasio：uv add lasio") from e

        las = lasio.read(str(path))
        result: dict[str, np.ndarray] = {}
        for curve in las.curves:
            name = curve.mnemonic.strip().upper()
            if name in _DEPTH_MNEMONICS:
                continue
            result[name] = np.asarray(curve.data, dtype=np.float64)
        return result
