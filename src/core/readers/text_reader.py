from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseReader

# 自动探测这些分隔符（按优先级）
_SEP_CANDIDATES = [",", "\t", ";", r"\s+"]


class TextReader(BaseReader):
    """读取文本格式数据文件（CSV / TSV / 空格分隔等）。

    - 第一行若为非数字则视为列头，作为通道名
    - 无列头时按 "col_0", "col_1" ... 命名
    - 自动探测分隔符：逗号、制表、分号、空白
    - 跳过以 # 开头的注释行
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".csv", ".txt", ".dat", ".asc")

    def read(self, path: Path) -> dict[str, np.ndarray]:
        df = self._read_df(path)
        result: dict[str, np.ndarray] = {}
        for col in df.columns:
            safe_name = str(col).strip().replace(" ", "_")
            result[safe_name] = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
        return result

    def _read_df(self, path: Path) -> pd.DataFrame:
        # 用前 _PROBE_LINES 行探测分隔符，避免对大文件多次全量解析
        sep = self._detect_separator(path)
        return self._parse_with_sep(path, sep)

    def _detect_separator(self, path: Path, probe_lines: int = 20) -> str:
        """读取文件前 N 行（跳过注释），逐个尝试分隔符，返回首个成功的。"""
        head_lines: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                if raw_line.strip().startswith("#") or not raw_line.strip():
                    continue
                head_lines.append(raw_line)
                if len(head_lines) >= probe_lines:
                    break

        if not head_lines:
            raise ValueError(
                f"文件 {path.name} 无有效数据行（仅含注释或空行）。"
            )

        snippet = "".join(head_lines)
        for sep in _SEP_CANDIDATES:
            try:
                df = pd.read_csv(
                    pd.io.common.StringIO(snippet),
                    sep=sep, engine="python", skip_blank_lines=True,
                )
                if df.shape[1] >= 1:
                    numeric = df.apply(pd.to_numeric, errors="coerce")
                    if not numeric.dropna(axis=1, how="all").empty:
                        return sep
            except (pd.errors.ParserError, ValueError):
                continue

        raise ValueError(
            f"无法解析文件 {path.name}：尝试了逗号/制表/分号/空格分隔均失败。\n"
            f"请确认文件为纯文本数值格式，注释行以 # 开头。"
        )

    def _parse_with_sep(self, path: Path, sep: str) -> pd.DataFrame:
        """用已确定的分隔符单次读取整个文件。"""
        df = pd.read_csv(
            path, sep=sep, engine="python",
            skip_blank_lines=True, comment="#",
        )

        all_numeric_header = all(
            not pd.isna(pd.to_numeric(str(c), errors="coerce"))
            for c in df.columns
        )
        if all_numeric_header:
            df = pd.read_csv(
                path, sep=sep, engine="python",
                skip_blank_lines=True, comment="#", header=None,
            )
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        numeric_cols = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = numeric_cols.dropna(axis=1, how="all")
        if numeric_cols.empty:
            raise ValueError(
                f"文件 {path.name} 未包含可转换为数值的列。"
            )
        return numeric_cols
