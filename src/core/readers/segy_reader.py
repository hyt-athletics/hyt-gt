from pathlib import Path

import numpy as np

from .base import BaseReader


class SegyReader(BaseReader):
    """读取 SEG-Y 地震数据文件。

    使用 segyio 解析，返回 2D 道集矩阵（traces, shape: n_traces × n_samples）
    和采样时间轴（times）。

    适用场景：井间地震、VSP 等有规则道集的 SEG-Y 文件。
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".segy", ".sgy")

    def read(self, path: Path) -> dict[str, np.ndarray]:
        try:
            import segyio
        except ImportError as e:
            raise ImportError("请先安装 segyio：uv add segyio") from e

        with segyio.open(str(path), ignore_geometry=True) as f:
            times = f.samples.astype(np.float64)
            traces = np.stack(
                [t.astype(np.float64) for t in f.trace], axis=0
            )

        return {"times": times, "traces": traces}
