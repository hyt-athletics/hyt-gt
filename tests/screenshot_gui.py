"""启动 GUI 并在 1.5 秒后自动截图保存，然后退出。

用法：
    uv run python tests/screenshot_gui.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.runner import AlgorithmRunner
from src.core.data_store import DataStore
from src.ui.main_window import MainWindow

WORKSPACE = Path.home() / "geophys-workspace"
OUT = _ROOT / "tests" / "data" / "screenshot_gui.png"

store = DataStore(WORKSPACE)
store.init_workspace()
boreholes = store.list_boreholes()

registry = AlgorithmRegistry(_ROOT / "algorithms")
registry.scan()
runner = AlgorithmRunner(registry, WORKSPACE)

app = QApplication(sys.argv)

win = MainWindow(runner, registry, boreholes, _ROOT / "pipelines")
win.show()

# 展开所有树节点，确保截图能看到内容
win._algo_tree.expandAll()

def _capture():
    screen = app.primaryScreen()
    pixmap = screen.grabWindow(win.winId())
    pixmap.save(str(OUT))
    print(f"截图已保存：{OUT}")
    app.quit()

QTimer.singleShot(1200, _capture)
sys.exit(app.exec())
