import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8 支持（避免中文乱码）
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        os.system("chcp 65001 >nul 2>&1")
    except Exception:
        pass

# Windows 高 DPI 缩放支持
if sys.platform == "win32":
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.runner import AlgorithmRunner
from src.core.data_store import DataStore
from src.core.font_utils import qt_cjk_font
from src.ui.main_window import MainWindow

WORKSPACE = Path.home() / "geophys-workspace"


def main() -> None:
    # 初始化工作区目录
    store = DataStore(WORKSPACE)
    store.init_workspace()

    # 算法注册（扫描本项目 algorithms/ 目录，方便开发时测试）
    algo_dir = Path(__file__).parent / "algorithms"
    registry = AlgorithmRegistry(algo_dir)
    registry.scan()

    runner = AlgorithmRunner(registry, WORKSPACE)
    boreholes = store.list_boreholes()
    pipelines_dir = Path(__file__).parent / "pipelines"

    app = QApplication(sys.argv)

    # 全局 CJK 字体：确保 Windows / macOS / Linux 均能正常显示中文
    cjk_font = qt_cjk_font()
    if cjk_font:
        app.setFont(QFont(cjk_font, 9))

    win = MainWindow(runner, registry, boreholes, pipelines_dir)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
