"""Reader 注册表：按文件扩展名自动选择对应的 Reader 实现。

新增格式只需：
1. 在 src/core/readers/ 创建新的 Reader 类（继承 BaseReader）
2. 在下方 _REGISTRY 中注册扩展名 → Reader 类的映射
"""
from pathlib import Path

from .base import BaseReader
from .binary_reader import BinaryReader
from .las_reader import LasReader
from .segy_reader import SegyReader
from .text_reader import TextReader

_REGISTRY: dict[str, type[BaseReader]] = {
    ".las":  LasReader,
    ".segy": SegyReader,
    ".sgy":  SegyReader,
    ".csv":  TextReader,
    ".txt":  TextReader,
    ".dat":  TextReader,
    ".asc":  TextReader,
    ".bin":  BinaryReader,
}

# VTK 格式（需要 pyvista 可选依赖）
try:
    from .vtk_reader import VtkReader
    _REGISTRY.update({
        ".vtk": VtkReader, ".vtu": VtkReader,
        ".vts": VtkReader, ".vtr": VtkReader,
    })
except ImportError:
    pass


def get_reader(path: Path) -> BaseReader:
    """按文件扩展名返回对应的 Reader 实例。

    Raises:
        ValueError: 扩展名未注册时，列出所有已支持格式
    """
    ext = path.suffix.lower()
    cls = _REGISTRY.get(ext)
    if cls is None:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"不支持的文件格式：{ext}\n"
            f"当前支持：{supported}\n"
            f"如需扩展，在 src/core/readers/__init__.py 注册新 Reader 即可。"
        )
    return cls()


def supported_extensions() -> list[str]:
    """返回所有已注册的文件扩展名列表。"""
    return sorted(_REGISTRY.keys())
