import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from .base import BaseAlgorithm


class AlgorithmRegistry:
    def __init__(self, algo_dir: Path):
        self._algo_dir = algo_dir
        self._manifests: dict[str, dict] = {}
        self._paths: dict[str, Path] = {}
        self._module_cache: dict[str, ModuleType] = {}

    def scan(self) -> list[dict]:
        """扫描 algorithms/ 目录，读取所有 manifest.json。"""
        self._module_cache.clear()
        results = []
        if not self._algo_dir.exists():
            return results
        for p in self._algo_dir.iterdir():
            manifest_path = p / "manifest.json"
            if not (p.is_dir() and manifest_path.exists()):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[registry] 跳过 {p.name}：manifest 解析失败 — {e}")
                continue
            name = manifest["name"]
            self._manifests[name] = manifest
            self._paths[name] = p
            results.append(manifest)
        return results

    def list_manifests(self) -> list[dict]:
        return list(self._manifests.values())

    def get_manifest(self, name: str) -> dict:
        if name not in self._manifests:
            raise KeyError(f"算法未注册: {name}。请先调用 scan()。")
        return self._manifests[name]

    def load(self, name: str) -> BaseAlgorithm:
        """动态加载算法类实例。模块首次加载后缓存，后续调用直接实例化。"""
        manifest = self._manifests[name]
        algo_dir = self._paths[name]
        entry = manifest["entry"]

        if entry["type"] == "python":
            return self._load_python(name, algo_dir, entry)
        if entry["type"] == "c_lib":
            from .c_adapter import CLibAlgorithm
            ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
            lib_path = algo_dir / (entry["lib"] + ext)
            return CLibAlgorithm(lib_path, manifest)
        raise ValueError(f"未知 entry type: {entry['type']}。支持: python, c_lib")

    def _load_python(self, name: str, algo_dir: Path, entry: dict) -> BaseAlgorithm:
        if name in self._module_cache:
            return getattr(self._module_cache[name], entry["class"])()

        module_name = entry["module"]
        candidates = (
            list(algo_dir.glob(f"{module_name}.pyd"))             # Windows 编译模块
            + list(algo_dir.glob(f"{module_name}.cpython-*.so"))  # Linux/Mac 编译模块
            + [algo_dir / f"{module_name}.py"]                     # 源码（开发模式回退）
        )
        module_file = next((f for f in candidates if f.exists()), None)
        if module_file is None:
            raise FileNotFoundError(
                f"算法模块不存在: {algo_dir}/{module_name}.[pyd|cpython-*.so|py]\n"
                f"请确认算法包目录中包含 algorithm.py 或编译后的二进制模块。"
            )
        # 确保 src/ 在 sys.path 中，算法模块无需自行管理路径
        src_dir = str(Path(__file__).resolve().parent.parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        spec = importlib.util.spec_from_file_location(module_name, module_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._module_cache[name] = mod
        return getattr(mod, entry["class"])()
