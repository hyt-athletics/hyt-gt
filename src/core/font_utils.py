"""跨平台 CJK 字体工具。

为 PyQt6 界面和 matplotlib 图表提供统一的中文字体配置，
优先级：Microsoft YaHei（Win）→ PingFang SC（Mac）→ Noto Sans CJK SC（Linux）。
"""
from __future__ import annotations

_CJK_CANDIDATES = [
    "Microsoft YaHei",      # Windows
    "PingFang SC",          # macOS
    "Hiragino Sans GB",     # macOS 备选
    "Noto Sans CJK SC",     # Linux / 跨平台
    "WenQuanYi Micro Hei",  # Linux
    "SimHei",               # Windows 备选
]


def qt_cjk_font() -> str | None:
    """返回当前平台可用的首选 CJK 字体名，找不到返回 None。

    调用方示例（在 QApplication 创建后）：
        name = qt_cjk_font()
        if name:
            app.setFont(QFont(name, 9))
    """
    from PyQt6.QtGui import QFontDatabase
    available = set(QFontDatabase.families())
    return next((f for f in _CJK_CANDIDATES if f in available), None)


def setup_matplotlib_cjk() -> None:
    """配置 matplotlib rcParams 以正确渲染中文字符。

    调用方示例（在 import matplotlib 之后，plt 调用之前）：
        from src.core.font_utils import setup_matplotlib_cjk
        setup_matplotlib_cjk()
    """
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    available_names = {f.name for f in fm.fontManager.ttflist}
    cjk_font = next((f for f in _CJK_CANDIDATES if f in available_names), None)

    if cjk_font:
        plt.rcParams["font.family"] = [cjk_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
