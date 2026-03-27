import json
from pathlib import Path


class DataStore:
    """工作区目录管理，替代数据库。"""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.boreholes_dir = workspace_dir / "boreholes"
        self.algo_dir = workspace_dir / "algorithms"

    def init_workspace(self) -> None:
        """初始化工作区目录结构（首次运行时创建）。"""
        self.boreholes_dir.mkdir(parents=True, exist_ok=True)
        self.algo_dir.mkdir(parents=True, exist_ok=True)
        ws_config = self.workspace_dir / "workspace.json"
        if not ws_config.exists():
            ws_config.write_text(
                json.dumps({"version": "0.1.0"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def list_boreholes(self) -> list[str]:
        if not self.boreholes_dir.exists():
            return []
        return sorted(p.name for p in self.boreholes_dir.iterdir() if p.is_dir())

    def create_borehole(self, name: str, meta: dict | None = None) -> Path:
        bh_dir = self.boreholes_dir / name
        bh_dir.mkdir(parents=True, exist_ok=True)
        meta_path = bh_dir / "borehole_meta.json"
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps(meta or {"name": name}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return bh_dir

    def ensure_method_dirs(self, borehole_name: str, method: str) -> tuple[Path, Path]:
        """确保方法目录存在，返回 (raw_dir, processed_dir)。"""
        base = self.boreholes_dir / borehole_name / method
        raw = base / "raw"
        processed = base / "processed"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        return raw, processed
