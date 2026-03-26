from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseReader(ABC):
    """所有数据格式 Reader 的抽象基类。

    子类约定：
    - read() 返回 {channel_name: np.ndarray}，通道名为合法文件名（不含特殊字符）
    - channel 的数据为一维或二维 float64 数组
    - 不在 read() 内保存文件，由 DataImporter 统一落盘
    """

    @abstractmethod
    def read(self, path: Path) -> dict[str, np.ndarray]:
        """读取文件，返回 {channel_name: data_array}。"""

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """本 Reader 处理的小写扩展名，如 ('.las',)。"""
