"""自定义二进制格式读取器。

通过伴随的 spec 文件（<数据文件>.spec.json）描述二进制布局，
无需修改平台代码即可支持仪器厂商私有格式。

spec 文件格式示例（my_data.bin.spec.json）：
{
    "byte_order": "big",          // "big" | "little"（默认 "little"）
    "header_bytes": 0,            // 文件开头跳过的字节数（默认 0）
    "record_dtype": [             // structured dtype 定义，按顺序列出字段
        ["depth",      "float32"],          // 标量字段：[名称, dtype]
        ["amplitude",  "float32",  1024],   // 数组字段：[名称, dtype, count]
        ["flag",       "int16"]
    ]
}

注：每条记录按 record_dtype 重复解析直到文件末尾。
"""
import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseReader

_BYTE_ORDER_MAP = {"big": ">", "little": "<", "native": "="}


class BinaryReader(BaseReader):
    """读取通过 spec.json 描述的自定义二进制文件。"""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".bin",)

    def read(self, path: Path) -> dict[str, np.ndarray]:
        spec = self._load_spec(path)
        dtype = self._build_dtype(spec)
        data = self._parse_binary(path, spec, dtype)
        return {name: np.asarray(data[name], dtype=np.float64).ravel()
                for name in dtype.names}

    # ── 私有方法 ──────────────────────────────────────────────────────

    def _load_spec(self, path: Path) -> dict[str, Any]:
        spec_path = path.with_suffix(path.suffix + ".spec.json")
        if not spec_path.exists():
            raise FileNotFoundError(
                f"未找到 spec 文件：{spec_path.name}\n"
                f"请在与数据文件同目录下创建描述二进制格式的 .spec.json 文件。\n"
                f"格式说明见 src/core/readers/binary_reader.py 顶部注释。"
            )
        return json.loads(spec_path.read_text(encoding="utf-8"))

    def _build_dtype(self, spec: dict) -> np.dtype:
        order = _BYTE_ORDER_MAP.get(spec.get("byte_order", "little"), "<")
        dt_fields = []
        for field in spec["record_dtype"]:
            name = field[0]
            base_dt = np.dtype(field[1]).newbyteorder(order)
            count = int(field[2]) if len(field) > 2 else 1
            dt_fields.append((name, base_dt, (count,)) if count > 1 else (name, base_dt))
        return np.dtype(dt_fields)

    def _parse_binary(
        self, path: Path, spec: dict, dtype: np.dtype
    ) -> np.ndarray:
        header_bytes = int(spec.get("header_bytes", 0))
        raw = path.read_bytes()[header_bytes:]
        n_records = len(raw) // dtype.itemsize
        if n_records == 0:
            raise ValueError(
                f"文件 {path.name} 数据为空或 header_bytes 设置过大。"
            )
        return np.frombuffer(raw[: n_records * dtype.itemsize], dtype=dtype)
