"""统一数据导入器。

自动识别文件格式 → 调用对应 Reader → 将所有通道存为 raw/ CSV 文件。
"""
from pathlib import Path

import numpy as np

from .readers import get_reader


class DataImporter:
    """将任意受支持格式的数据文件导入到工作区 raw/ 目录。

    所有通道统一存储为 CSV，保持与 AlgorithmRunner 的兼容性。
    """

    def __init__(self, workspace_dir: Path):
        self._workspace_dir = workspace_dir

    def import_file(
        self,
        data_path: Path,
        borehole_name: str,
        method: str,
        channel_names: list[str] | None = None,
    ) -> list[str]:
        """导入文件，返回已写入 raw/ 的通道名列表。

        Args:
            data_path:     源数据文件路径
            borehole_name: 目标钻孔名称
            method:        勘探方法（ert / ip / em / tem / logging）
            channel_names: 仅导入这些通道；None 表示导入全部

        Returns:
            实际写入的通道名列表

        Raises:
            ValueError:  文件格式不支持
            FileNotFoundError: 文件不存在
        """
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在：{data_path}")

        reader = get_reader(data_path)
        channels = reader.read(data_path)

        if channel_names is not None:
            channels = {k: v for k, v in channels.items() if k in channel_names}

        raw_dir = (
            self._workspace_dir / "boreholes" / borehole_name / method / "raw"
        )
        raw_dir.mkdir(parents=True, exist_ok=True)

        imported: list[str] = []
        for name, data in channels.items():
            arr = np.asarray(data, dtype=np.float64)
            np.savetxt(raw_dir / f"{name}.csv", arr, delimiter=",")
            imported.append(name)

        return imported
