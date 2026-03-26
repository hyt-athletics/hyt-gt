"""三维可视化控制面板 — EMIGMA / Voxler 风格。

组织结构：
  ┌──────────────────────────────────┐
  │ 场景对象树（☑ 可切换可见性）      │
  ├──────────────────────────────────┤
  │ 显示属性 / 色标 / 对数缩放       │
  ├──────────────────────────────────┤
  │ 透明度滑块                       │
  ├──────────────────────────────────┤
  │ 正交剖面 X/Y/Z 滑块             │
  ├──────────────────────────────────┤
  │ 视图选项（投影/标注/比例尺）      │
  └──────────────────────────────────┘
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

_COLORMAPS = [
    "jet_r", "viridis", "coolwarm", "hot_r", "terrain",
    "RdYlBu_r", "Spectral_r", "plasma", "cividis",
]

_CATEGORY_ICONS = {
    "mesh": "▣",
    "borehole": "⦿",
    "surface": "△",
    "slice": "◫",
    "annotation": "✦",
}


class Viewer3DControlPanel(QDockWidget):
    """EMIGMA / Voxler 风格三维控制面板。"""

    def __init__(self, viewer: object, parent: QWidget | None = None):
        super().__init__("3D 控制面板", parent)
        self.setObjectName("dock_3d_controls")
        self.setMinimumWidth(240)
        self.setMaximumWidth(360)

        self._viewer = viewer
        self._updating = False

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # ══ 场景对象树（EMIGMA 模式） ════════════════════════════════
        tree_group = QGroupBox("场景对象")
        tree_layout = QVBoxLayout(tree_group)
        tree_layout.setContentsMargins(2, 2, 2, 2)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["对象", "类型"])
        self._tree.setColumnWidth(0, 160)
        self._tree.setIndentation(12)
        self._tree.setRootIsDecorated(False)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        tree_layout.addWidget(self._tree)
        layout.addWidget(tree_group)

        # ══ 显示属性 ════════════════════════════════════════════════
        prop_group = QGroupBox("显示属性")
        prop_layout = QVBoxLayout(prop_group)
        prop_layout.setSpacing(4)

        self._prop_combo = QComboBox()
        self._prop_combo.currentTextChanged.connect(self._on_display_changed)
        prop_layout.addWidget(self._prop_combo)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("色标:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_display_changed)
        cmap_row.addWidget(self._cmap_combo, 1)
        prop_layout.addLayout(cmap_row)

        check_row = QHBoxLayout()
        self._log_check = QCheckBox("对数")
        self._log_check.stateChanged.connect(self._on_display_changed)
        check_row.addWidget(self._log_check)
        self._ortho_check = QCheckBox("正交投影")
        self._ortho_check.stateChanged.connect(self._on_ortho_changed)
        check_row.addWidget(self._ortho_check)
        prop_layout.addLayout(check_row)

        layout.addWidget(prop_group)

        # ══ 透明度 ══════════════════════════════════════════════════
        opacity_group = QGroupBox("模型透明度")
        opacity_layout = QHBoxLayout(opacity_group)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(5, 100)
        self._opacity_slider.setValue(85)
        self._opacity_slider.valueChanged.connect(self._on_display_changed)
        self._opacity_label = QLabel("85%")
        opacity_layout.addWidget(self._opacity_slider)
        opacity_layout.addWidget(self._opacity_label)
        layout.addWidget(opacity_group)

        # ══ 正交剖面 ════════════════════════════════════════════════
        slice_group = QGroupBox("正交剖面")
        slice_layout = QVBoxLayout(slice_group)
        slice_layout.setSpacing(3)

        self._slice_checks: dict[str, QCheckBox] = {}
        self._slice_sliders: dict[str, QSlider] = {}
        self._slice_labels: dict[str, QLabel] = {}

        for axis in ("X", "Y", "Z"):
            row = QHBoxLayout()
            cb = QCheckBox(axis)
            cb.stateChanged.connect(lambda _, a=axis.lower(): self._on_slice_toggle(a))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(500)
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda _, a=axis.lower(): self._on_slice_moved(a))
            lbl = QLabel("—")
            lbl.setFixedWidth(50)
            row.addWidget(cb)
            row.addWidget(slider, 1)
            row.addWidget(lbl)
            slice_layout.addLayout(row)
            self._slice_checks[axis.lower()] = cb
            self._slice_sliders[axis.lower()] = slider
            self._slice_labels[axis.lower()] = lbl

        layout.addWidget(slice_group)

        # ══ 视图选项 ════════════════════════════════════════════════
        view_group = QGroupBox("标注")
        view_layout = QVBoxLayout(view_group)

        ann_row = QHBoxLayout()
        self._north_btn = QPushButton("北向箭头")
        self._north_btn.setCheckable(True)
        self._north_btn.clicked.connect(self._on_north_clicked)
        ann_row.addWidget(self._north_btn)
        self._scale_btn = QPushButton("比例尺")
        self._scale_btn.setCheckable(True)
        self._scale_btn.clicked.connect(self._on_scale_clicked)
        ann_row.addWidget(self._scale_btn)
        view_layout.addLayout(ann_row)

        reset_btn = QPushButton("重置视角")
        reset_btn.clicked.connect(self._on_reset_camera)
        view_layout.addWidget(reset_btn)

        layout.addWidget(view_group)
        layout.addStretch()
        self.setWidget(container)

    # ── 公共方法 ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        """从 Viewer3D 同步场景对象树和属性列表。"""
        self._updating = True

        # 属性列表
        self._prop_combo.clear()
        names = getattr(self._viewer, "scalar_names", [])
        self._prop_combo.addItems(names)

        # 场景对象树
        self._tree.clear()
        objects = getattr(self._viewer, "scene_objects", {})
        for name, obj in objects.items():
            icon = _CATEGORY_ICONS.get(obj.category, "?")
            item = QTreeWidgetItem([f"{icon} {name}", obj.category])
            item.setCheckState(0, Qt.CheckState.Checked if obj.visible else Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, name)
            self._tree.addTopLevelItem(item)

        # 剖面滑块范围
        bounds = getattr(self._viewer, "bounds", None)
        if bounds and len(bounds) >= 6:
            for axis, (lo_idx, hi_idx) in {"x": (0, 1), "y": (2, 3), "z": (4, 5)}.items():
                lo, hi = bounds[lo_idx], bounds[hi_idx]
                self._slice_sliders[axis].setToolTip(f"{lo:.0f} ~ {hi:.0f} m")

        self._updating = False

    # ── 事件处理 ──────────────────────────────────────────────────────

    def _on_tree_item_changed(self, item: QTreeWidgetItem, _col: int) -> None:
        if self._updating:
            return
        name = item.data(0, Qt.ItemDataRole.UserRole)
        visible = item.checkState(0) == Qt.CheckState.Checked
        if hasattr(self._viewer, "set_object_visible"):
            self._viewer.set_object_visible(name, visible)

    def _on_display_changed(self) -> None:
        if self._updating:
            return
        self._opacity_label.setText(f"{self._opacity_slider.value()}%")
        prop = self._prop_combo.currentText()
        if not prop or not hasattr(self._viewer, "update_mesh_display"):
            return
        self._viewer.update_mesh_display(
            scalars=prop,
            cmap=self._cmap_combo.currentText(),
            opacity=self._opacity_slider.value() / 100.0,
            log_scale=self._log_check.isChecked(),
        )
        self.refresh()

    def _on_ortho_changed(self, _state: int) -> None:
        if hasattr(self._viewer, "set_parallel_projection"):
            self._viewer.set_parallel_projection(self._ortho_check.isChecked())

    def _on_slice_toggle(self, axis: str) -> None:
        enabled = self._slice_checks[axis].isChecked()
        self._slice_sliders[axis].setEnabled(enabled)
        if enabled:
            self._on_slice_moved(axis)
        else:
            self._slice_labels[axis].setText("—")
            if hasattr(self._viewer, "remove_object"):
                self._viewer.remove_object(f"剖面-{axis.upper()}")

    def _on_slice_moved(self, axis: str) -> None:
        if not self._slice_checks[axis].isChecked():
            return
        bounds = getattr(self._viewer, "bounds", None)
        if not bounds:
            return
        lo, hi = bounds[{"x": 0, "y": 2, "z": 4}[axis]], bounds[{"x": 1, "y": 3, "z": 5}[axis]]
        frac = self._slice_sliders[axis].value() / 1000.0
        pos = lo + (hi - lo) * frac
        self._slice_labels[axis].setText(f"{pos:.0f}m")
        if hasattr(self._viewer, "add_clip_plane"):
            self._viewer.add_clip_plane(normal=axis, position=pos)

    def _on_north_clicked(self) -> None:
        if self._north_btn.isChecked():
            if hasattr(self._viewer, "add_north_arrow"):
                self._viewer.add_north_arrow()
        else:
            if hasattr(self._viewer, "remove_object"):
                self._viewer.remove_object("北向箭头")

    def _on_scale_clicked(self) -> None:
        if self._scale_btn.isChecked():
            if hasattr(self._viewer, "add_scale_bar"):
                self._viewer.add_scale_bar()
        else:
            if hasattr(self._viewer, "remove_object"):
                self._viewer.remove_object("比例尺")

    def _on_reset_camera(self) -> None:
        if hasattr(self._viewer, "_plotter") and self._viewer._plotter:
            self._viewer._plotter.reset_camera()
