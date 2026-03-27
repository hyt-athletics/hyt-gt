from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

ProgressCallback = Callable[[float, str], None]


@dataclass
class AlgorithmResult:
    outputs: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class BaseAlgorithm(ABC):
    """算法开发者必须继承此类，实现 run() 方法。

    inputs  的键名与 manifest.json inputs[].name  一致。
    outputs 的键名与 manifest.json outputs[].name 一致。

    长时间运行的算法可在 run() 中调用 self.report_progress(0.5, "50% done")
    向 UI 报告进度（可选，不调用也不会出错）。
    """

    _progress_cb: ProgressCallback | None = None

    def set_progress_callback(self, cb: ProgressCallback) -> None:
        self._progress_cb = cb

    def report_progress(self, fraction: float, message: str = "") -> None:
        """报告进度。fraction: 0.0~1.0，message: 可选状态文本。"""
        if self._progress_cb:
            self._progress_cb(min(max(fraction, 0.0), 1.0), message)

    @abstractmethod
    def run(self, **inputs: Any) -> AlgorithmResult: ...
