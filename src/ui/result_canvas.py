from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ResultCanvas(FigureCanvasQTAgg):
    """中央可视化区域：支持测井道图和 2D 伪剖面。"""

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(8, 6))
        super().__init__(self._fig)
        self.setParent(parent)
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        ax = self._fig.add_subplot(111)
        ax.text(
            0.5, 0.5, "运行算法后结果将显示在此处",
            ha="center", va="center", fontsize=13, color="#bbbbbb",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.draw()

    def plot_curves(
        self,
        curves: dict[str, np.ndarray],
        depth: np.ndarray | None = None,
    ) -> None:
        """多曲线测井道图：每条曲线独立子轴，Y 轴反转（深度向下）。"""
        n = len(curves)
        if n == 0:
            return
        self._fig.clear()
        axes = self._fig.subplots(1, n, sharey=True) if n > 1 else [self._fig.add_subplot(111)]
        y = depth if depth is not None else np.arange(next(iter(curves.values())).shape[0])
        for ax, (name, data) in zip(axes, curves.items()):
            ax.plot(data, y, linewidth=1.4)
            ax.set_title(name, fontsize=8, pad=3)
            ax.invert_yaxis()
            ax.grid(linestyle=":", linewidth=0.4, color="#cccccc")
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("深度 (m)", fontsize=8)
        for ax in axes[1:]:
            ax.set_yticklabels([])
        self._fig.tight_layout()
        self.draw()

    def plot_2d(self, data: np.ndarray, title: str = "") -> None:
        """2D 伪剖面（ERT/IP 反演结果）。"""
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        im = ax.imshow(data, aspect="auto", origin="upper", cmap="jet_r")
        self._fig.colorbar(im, ax=ax, label="电阻率 (Ω·m)")
        ax.set_title(title, fontsize=10)
        self._fig.tight_layout()
        self.draw()

    def load_and_plot_csv(self, csv_paths: list[str]) -> None:
        """从 CSV 文件列表读取数据并绘制曲线。"""
        curves: dict[str, np.ndarray] = {}
        depth: np.ndarray | None = None
        for path_str in csv_paths:
            p = Path(path_str)
            if not p.exists() or p.suffix.lower() != ".csv":
                continue
            try:
                arr = np.loadtxt(str(p), delimiter=",", ndmin=1)
                if arr.size == 0:
                    continue
                arr = arr.ravel()
                if depth is None:
                    depth = np.arange(len(arr))
                curves[p.stem] = arr
            except Exception:
                continue
        if curves:
            self.plot_curves(curves, depth)

    def load_and_plot_npz(self, npz_paths: list[str]) -> None:
        """从 NPZ 文件加载 3D 数据，取中间切片做 2D 降级显示。"""
        for path_str in npz_paths:
            p = Path(path_str)
            if not p.exists() or p.suffix.lower() != ".npz":
                continue
            try:
                arr = np.load(str(p))["data"]
            except Exception:
                continue
            if arr.ndim == 3:
                mid = arr.shape[2] // 2
                self.plot_2d(arr[:, :, mid], title=f"{p.stem} (z={mid})")
                return
            if arr.ndim == 2:
                self.plot_2d(arr, title=p.stem)
                return
