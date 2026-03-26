from pathlib import Path

from PyQt6.QtCore import QElapsedTimer, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.algorithms.pipeline_runner import PipelineRunner
from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.runner import AlgorithmRunner
from src.core.constants import CATEGORY_ORDER, METHOD_ORDER, Category, Method
from src.core.data_store import DataStore
from src.core.importer import DataImporter
from src.core.readers import supported_extensions
from src.ui.result_canvas import ResultCanvas


def _ask_method(parent, default: str) -> tuple[str, bool]:
    methods = [m.value for m in Method]
    default_idx = methods.index(default) if default in methods else 0
    return QInputDialog.getItem(
        parent, "选择勘探方法", "数据将存入哪个方法的 raw/ 目录？",
        methods, default_idx, False,
    )


class _RunWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(float, str)

    def __init__(self, runner: AlgorithmRunner, algo_name: str, borehole: str, params: dict):
        super().__init__()
        self._runner = runner
        self._algo_name = algo_name
        self._borehole = borehole
        self._params = params

    def run(self) -> None:
        try:
            saved = self._runner.run(
                self._algo_name, self._borehole, self._params,
                progress_cb=self.progress.emit,
            )
            self.finished.emit({k: str(v) for k, v in saved.items()})
        except Exception as exc:
            exc_type = type(exc).__name__
            self.error.emit(f"[{exc_type}] {exc}")


class _PipelineWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, pipeline_runner: PipelineRunner, pipeline_name: str, borehole: str):
        super().__init__()
        self._pipeline_runner = pipeline_runner
        self._pipeline_name = pipeline_name
        self._borehole = borehole

    def run(self) -> None:
        try:
            all_saved = self._pipeline_runner.run(self._pipeline_name, self._borehole)
            flat = {}
            for algo, outputs in all_saved.items():
                for out_name, path in outputs.items():
                    flat[f"{algo}/{out_name}"] = str(path)
            self.finished.emit(flat)
        except Exception as exc:
            self.error.emit(str(exc))


class _ProjectTreeWorker(QThread):
    """在后台线程遍历工作区目录，避免阻塞 UI。"""
    finished = pyqtSignal(list)

    def __init__(self, workspace_dir: Path):
        super().__init__()
        self._ws = workspace_dir

    def run(self) -> None:
        tree_data: list[tuple[str, list[tuple[str, dict[str, list[str]]]]]] = []
        if not self._ws.exists():
            self.finished.emit(tree_data)
            return
        for bh_dir in sorted(self._ws.iterdir()):
            if not bh_dir.is_dir():
                continue
            methods: list[tuple[str, dict[str, list[str]]]] = []
            for method_dir in sorted(bh_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                subs: dict[str, list[str]] = {}
                for sub in ("raw", "processed"):
                    sub_dir = method_dir / sub
                    if not sub_dir.exists():
                        continue
                    files = sorted(
                        str(f) for f in sub_dir.iterdir() if f.is_file()
                    )
                    if files:
                        subs[sub] = files
                methods.append((method_dir.name, subs))
            tree_data.append((bh_dir.name, methods))
        self.finished.emit(tree_data)


class MainWindow(QMainWindow):
    def __init__(
        self,
        runner: AlgorithmRunner,
        registry: AlgorithmRegistry,
        boreholes: list[str],
        pipelines_dir: Path | None = None,
    ):
        super().__init__()
        self._runner = runner
        self._registry = registry
        self._worker: _RunWorker | None = None
        self._pipeline_worker: _PipelineWorker | None = None
        self._tree_worker: _ProjectTreeWorker | None = None
        self._selected_algo: str | None = None
        self._elapsed_timer = QElapsedTimer()
        self._param_widgets: dict[str, QSpinBox | QDoubleSpinBox | QLineEdit] = {}

        ws = runner.workspace_dir
        p_dir = pipelines_dir or (Path(__file__).parent.parent.parent / "pipelines")
        self._pipeline_runner = PipelineRunner(registry, ws, p_dir)
        self._pipeline_runner.scan()

        ws_label = str(ws).replace(str(Path.home()), "~")
        self.setWindowTitle(f"井中探测数据处理与反演子系统 v0.3 — {ws_label}")
        self.resize(1280, 760)

        self._build_central()
        self._build_left_dock()
        self._build_right_dock(boreholes)
        self._build_bottom_dock()
        self._build_menubar()
        self._build_toolbar()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # 构建各区域
    # ------------------------------------------------------------------

    def _build_central(self) -> None:
        self._canvas = ResultCanvas(self)
        self._viewer3d = None
        self._3d_controls = None

        self._central_tabs = QTabWidget()
        self._central_tabs.addTab(self._canvas, "2D 图表")

        try:
            from src.ui.viewer_3d import Viewer3D
            from src.ui.viewer_3d_controls import Viewer3DControlPanel
            self._viewer3d = Viewer3D(self)
            self._central_tabs.addTab(self._viewer3d, "3D 视图")
            self._3d_controls = Viewer3DControlPanel(self._viewer3d, self)
            self._3d_controls.setVisible(False)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._3d_controls)
            self._central_tabs.currentChanged.connect(self._on_tab_changed)
        except Exception:
            pass

        self.setCentralWidget(self._central_tabs)

    def _on_tab_changed(self, index: int) -> None:
        is_3d = self._viewer3d is not None and self._central_tabs.widget(index) is self._viewer3d
        if self._3d_controls:
            self._3d_controls.setVisible(is_3d)
            if is_3d:
                self._3d_controls.refresh()

    def _build_left_dock(self) -> None:
        dock = QDockWidget("项目树", self)
        dock.setObjectName("dock_project")
        dock.setMinimumWidth(200)
        dock.setMaximumWidth(320)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        self._proj_tree = QTreeWidget()
        self._proj_tree.setHeaderHidden(True)
        self._proj_tree.itemDoubleClicked.connect(self._on_proj_item_dblclicked)
        layout.addWidget(self._proj_tree)

        refresh_btn = QPushButton("刷新项目树")
        refresh_btn.setFixedHeight(26)
        refresh_btn.clicked.connect(self._refresh_project_tree)
        layout.addWidget(refresh_btn)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._refresh_project_tree()

    def _build_right_dock(self, boreholes: list[str]) -> None:
        self._dock_algo = QDockWidget("算法面板", self)
        self._dock_algo.setObjectName("dock_algo")
        self._dock_algo.setMinimumWidth(260)
        self._dock_algo.setMaximumWidth(400)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # 钻孔选择
        bh_group = QGroupBox("钻孔")
        bh_layout = QHBoxLayout(bh_group)
        self._bh_combo = QComboBox()
        if boreholes:
            self._bh_combo.addItems(boreholes)
        else:
            self._bh_combo.addItem("（无钻孔，请先创建）")
        bh_layout.addWidget(self._bh_combo)
        layout.addWidget(bh_group)

        # 算法树
        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout(algo_group)
        self._algo_tree = QTreeWidget()
        self._algo_tree.setHeaderHidden(True)
        self._algo_tree.setMinimumHeight(220)
        self._build_algo_tree()
        self._algo_tree.itemClicked.connect(self._on_tree_item_clicked)
        algo_layout.addWidget(self._algo_tree)
        layout.addWidget(algo_group)

        # 当前算法标签
        self._selected_label = QLabel("← 从上方选择算法")
        self._selected_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._selected_label)

        # 参数表单
        param_group = QGroupBox("算法参数")
        self._param_form = QFormLayout(param_group)
        layout.addWidget(param_group)

        # 运行按钮
        self._run_btn = QPushButton("▶  运行算法")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        # 流水线
        pipe_group = QGroupBox("流水线")
        pipe_layout = QVBoxLayout(pipe_group)
        self._pipe_combo = QComboBox()
        for p in self._pipeline_runner.list_pipelines():
            self._pipe_combo.addItem(p.get("display_name", p["name"]), p["name"])
        pipe_layout.addWidget(self._pipe_combo)
        self._pipe_run_btn = QPushButton("▶  运行流水线")
        self._pipe_run_btn.setFixedHeight(30)
        self._pipe_run_btn.setEnabled(bool(self._pipeline_runner.list_pipelines()))
        self._pipe_run_btn.clicked.connect(self._on_run_pipeline)
        pipe_layout.addWidget(self._pipe_run_btn)
        layout.addWidget(pipe_group)

        layout.addStretch()
        scroll.setWidget(container)
        self._dock_algo.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_algo)

    def _build_bottom_dock(self) -> None:
        self._dock_log = QDockWidget("执行日志", self)
        self._dock_log.setObjectName("dock_log")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFixedHeight(140)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px;")

        self._dock_log.setWidget(self._log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_log)

    def _build_menubar(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("文件(&F)")
        act_import = QAction("导入数据(&I)", self)
        act_import.setShortcut("Ctrl+O")
        act_import.triggered.connect(self._on_import_data)
        file_menu.addAction(act_import)
        file_menu.addSeparator()
        act_quit = QAction("退出(&Q)", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = mb.addMenu("视图(&V)")
        act_reset = QAction("重置面板布局", self)
        act_reset.triggered.connect(self._reset_layout)
        view_menu.addAction(act_reset)
        act_toggle_log = QAction("显示/隐藏日志", self)
        act_toggle_log.triggered.connect(
            lambda: self._dock_log.setVisible(not self._dock_log.isVisible())
        )
        view_menu.addAction(act_toggle_log)

        help_menu = mb.addMenu("帮助(&H)")
        act_guide = QAction("开发者指南", self)
        act_guide.triggered.connect(self._on_open_dev_guide)
        help_menu.addAction(act_guide)
        help_menu.addSeparator()
        act_about = QAction("关于", self)
        act_about.triggered.connect(
            lambda: QMessageBox.about(
                self, "关于",
                "井中探测数据处理与反演子系统 v0.3\n\nPyQt6 + matplotlib + scipy\n\n"
                f"工作区: {self._runner.workspace_dir}\n"
                f"算法目录: {self._registry._algo_dir}",
            )
        )
        help_menu.addAction(act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("主工具栏", self)
        tb.setObjectName("toolbar_main")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_new_bh = QAction("+ 新建钻孔", self)
        act_new_bh.triggered.connect(self._on_create_borehole)
        tb.addAction(act_new_bh)

        act_import = QAction("Import 导入数据", self)
        act_import.triggered.connect(self._on_import_data)
        tb.addAction(act_import)

        tb.addSeparator()

        self._tb_run_act = QAction("▶ 运行算法", self)
        self._tb_run_act.triggered.connect(self._on_run)
        tb.addAction(self._tb_run_act)

        act_pipe = QAction("⛓ 运行流水线", self)
        act_pipe.triggered.connect(self._on_run_pipeline)
        tb.addAction(act_pipe)

        tb.addSeparator()

        act_refresh = QAction("↻ 刷新", self)
        act_refresh.triggered.connect(self._refresh_project_tree)
        tb.addAction(act_refresh)

    def _build_statusbar(self) -> None:
        self._status_msg = QLabel("就绪")
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedWidth(120)
        self._elapsed = QLabel("")
        self.statusBar().addWidget(self._status_msg, 1)
        self.statusBar().addPermanentWidget(self._elapsed)
        self.statusBar().addPermanentWidget(self._progress)

    # ------------------------------------------------------------------
    # 算法树
    # ------------------------------------------------------------------

    def _build_algo_tree(self) -> None:
        manifests = [
            m for m in self._registry.list_manifests()
            if m.get("inputs") and m.get("outputs")
        ]
        grouped: dict[str, dict[str, list[dict]]] = {}
        for m in manifests:
            method = m.get("method", "other")
            cat = m.get("category", "other")
            grouped.setdefault(method, {}).setdefault(cat, []).append(m)

        method_order = [m.value for m in METHOD_ORDER]
        methods = [m for m in method_order if m in grouped] + [
            m for m in grouped if m not in method_order
        ]
        for method in methods:
            try:
                label = Method(method).label
            except ValueError:
                label = method.upper()
            method_item = QTreeWidgetItem([label])
            method_item.setData(0, Qt.ItemDataRole.UserRole, None)
            self._algo_tree.addTopLevelItem(method_item)

            cats = grouped[method]
            cat_order = [c.value for c in CATEGORY_ORDER]
            ordered_cats = [c for c in cat_order if c in cats] + [
                c for c in cats if c not in cat_order
            ]
            for cat in ordered_cats:
                try:
                    cat_label = Category(cat).label
                except ValueError:
                    cat_label = cat
                cat_item = QTreeWidgetItem([cat_label])
                cat_item.setData(0, Qt.ItemDataRole.UserRole, None)
                method_item.addChild(cat_item)
                for manifest in cats[cat]:
                    algo_item = QTreeWidgetItem([manifest["display_name"]])
                    algo_item.setData(0, Qt.ItemDataRole.UserRole, manifest["name"])
                    cat_item.addChild(algo_item)
            method_item.setExpanded(True)

    # ------------------------------------------------------------------
    # 项目树
    # ------------------------------------------------------------------

    def _refresh_project_tree(self) -> None:
        self._tree_worker = _ProjectTreeWorker(self._runner.workspace_dir)
        self._tree_worker.finished.connect(self._on_project_tree_ready)
        self._tree_worker.start()

    def _on_project_tree_ready(self, tree_data: list) -> None:
        self._proj_tree.clear()
        for bh_name, methods in tree_data:
            bh_item = QTreeWidgetItem([bh_name])
            bh_item.setData(0, Qt.ItemDataRole.UserRole, None)
            self._proj_tree.addTopLevelItem(bh_item)
            for method_name, subs in methods:
                m_item = QTreeWidgetItem([method_name])
                m_item.setData(0, Qt.ItemDataRole.UserRole, None)
                bh_item.addChild(m_item)
                for sub_name, files in subs.items():
                    sub_item = QTreeWidgetItem([sub_name])
                    sub_item.setData(0, Qt.ItemDataRole.UserRole, None)
                    m_item.addChild(sub_item)
                    for fpath in files:
                        f_item = QTreeWidgetItem([Path(fpath).name])
                        f_item.setData(0, Qt.ItemDataRole.UserRole, fpath)
                        sub_item.addChild(f_item)
            bh_item.setExpanded(True)

    # ------------------------------------------------------------------
    # 槽函数
    # ------------------------------------------------------------------

    def _on_proj_item_dblclicked(self, item: QTreeWidgetItem, _col: int) -> None:
        path_str = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        p = Path(path_str)
        if p.suffix.lower() == ".csv":
            self._canvas.load_and_plot_csv([path_str])
            self._central_tabs.setCurrentIndex(0)
            self._log_info(f"预览：{p.name}")
        elif p.suffix.lower() in (".vtk", ".vtu", ".vts", ".vtr") and self._viewer3d:
            self._viewer3d.load_file(path_str)
            self._central_tabs.setCurrentIndex(1)
            self._log_info(f"3D 预览：{p.name}")
        elif p.suffix.lower() == ".npz":
            self._canvas.load_and_plot_npz([path_str])
            self._central_tabs.setCurrentIndex(0)
            self._log_info(f"预览：{p.name}")

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        algo_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not algo_name:
            return
        self._selected_algo = algo_name
        self._selected_label.setText(item.text(0))
        self._selected_label.setStyleSheet("font-weight: bold; color: #1a1a2e;")
        self._run_btn.setEnabled(True)
        self._refresh_params(algo_name)

    def _refresh_params(self, algo_name: str) -> None:
        while self._param_form.rowCount():
            self._param_form.removeRow(0)
        self._param_widgets.clear()

        try:
            manifest = self._registry.get_manifest(algo_name)
        except KeyError:
            return

        has_params = False
        for spec in manifest["inputs"]:
            if spec["type"] in ("ndarray_1d", "ndarray_2d", "dataframe"):
                hint = QLabel(f"← raw/{spec.get('file', spec['name'] + '.csv')}")
                hint.setStyleSheet("color: gray; font-style: italic;")
                self._param_form.addRow(spec["label"], hint)
            else:
                w = self._make_param_widget(spec)
                self._param_form.addRow(spec["label"], w)
                self._param_widgets[spec["name"]] = w
            has_params = True

        if not has_params:
            self._param_form.addRow(QLabel("（此算法无需额外参数）"))

    def _make_param_widget(self, spec: dict) -> QSpinBox | QDoubleSpinBox | QLineEdit:
        ptype = spec.get("type", "str")
        if ptype == "int":
            w = QSpinBox()
            w.setRange(-99999, 99999)
            w.setValue(int(spec.get("default", 0)))
            return w
        if ptype == "float":
            w = QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(-1e9, 1e9)
            w.setValue(float(spec.get("default", 0.0)))
            return w
        return QLineEdit(str(spec.get("default", "")))

    def _collect_params(self) -> dict:
        params = {}
        for name, w in self._param_widgets.items():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                params[name] = str(w.value())
            else:
                params[name] = w.text()
        return params

    def _on_run(self) -> None:
        algo_name = self._selected_algo
        borehole = self._bh_combo.currentText()
        if not algo_name:
            QMessageBox.warning(self, "提示", "请先从算法列表选择算法。")
            return
        if borehole.startswith("（"):
            QMessageBox.warning(self, "提示", "请先点击工具栏「新建钻孔」创建钻孔。")
            return

        # Pre-flight: check required input files exist
        manifest = self._registry.get_manifest(algo_name)
        method = manifest.get("method", "")
        raw_dir = self._runner.workspace_dir / "boreholes" / borehole / method / "raw"
        missing_files = []
        for inp in manifest.get("inputs", []):
            if inp.get("type", "") in ("ndarray_1d", "ndarray_2d", "ndarray_3d", "dataframe", "mesh"):
                fname = inp.get("file", "")
                if fname and inp.get("required", True) and not (raw_dir / fname).exists():
                    missing_files.append(fname)
        if missing_files:
            file_list = "\n".join(f"  - {f}" for f in missing_files)
            QMessageBox.warning(
                self, "输入文件缺失",
                f"算法需要以下输入文件，但在 raw/ 目录中未找到：\n\n"
                f"{file_list}\n\n"
                f"请先将数据导入到：\n{raw_dir}",
            )
            return

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳  运行中...")
        self._status_msg.setText(f"运行：{algo_name}")
        self._progress.setVisible(True)
        self._elapsed_timer.start()

        self._worker = _RunWorker(self._runner, algo_name, borehole, self._collect_params())
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.start()

    def _on_run_pipeline(self) -> None:
        pipeline_name = self._pipe_combo.currentData()
        borehole = self._bh_combo.currentText()
        if not pipeline_name:
            QMessageBox.warning(self, "提示", "没有可用的流水线。")
            return
        if borehole.startswith("（"):
            QMessageBox.warning(self, "提示", "请先在工作区创建钻孔目录。")
            return

        self._pipe_run_btn.setEnabled(False)
        self._pipe_run_btn.setText("⏳  运行中...")
        self._status_msg.setText(f"流水线：{pipeline_name}")
        self._progress.setVisible(True)
        self._elapsed_timer.start()

        self._pipeline_worker = _PipelineWorker(
            self._pipeline_runner, pipeline_name, borehole
        )
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        self._pipeline_worker.start()

    def _on_finished(self, saved: dict) -> None:
        elapsed_ms = self._elapsed_timer.elapsed()
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  运行算法")
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._elapsed.setText(f"{elapsed_ms / 1000:.1f}s")
        self._status_msg.setText("完成")

        if saved:
            self._log_info(
                f"算法完成（{elapsed_ms}ms）：" + "  ".join(f"{k}={Path(v).name}" for k, v in saved.items())
            )
            csv_paths = [v for v in saved.values() if v.endswith(".csv")]
            vtk_paths = [v for v in saved.values() if v.endswith((".vtk", ".vtu", ".vts", ".vtr"))]
            npz_paths = [v for v in saved.values() if v.endswith(".npz")]
            if vtk_paths and self._viewer3d:
                self._viewer3d.load_file(vtk_paths[0])
                self._central_tabs.setCurrentIndex(1)
            elif csv_paths:
                self._canvas.load_and_plot_csv(csv_paths)
                self._central_tabs.setCurrentIndex(0)
            elif npz_paths:
                self._canvas.load_and_plot_npz(npz_paths)
                self._central_tabs.setCurrentIndex(0)
        else:
            self._log_info("算法运行完成，但没有输出文件。")
        self._refresh_project_tree()

    def _on_progress(self, fraction: float, message: str) -> None:
        pct = int(fraction * 100)
        self._progress.setRange(0, 100)
        self._progress.setValue(pct)
        if message:
            self._status_msg.setText(f"{pct}% — {message}")

    def _on_error(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  运行算法")
        self._progress.setVisible(False)
        self._status_msg.setText("失败")
        self._log_info(f"[错误] {msg}")

        # Add context-specific hints
        hint = ""
        if "[FileNotFoundError]" in msg:
            hint = "\n\n请检查钻孔 raw/ 目录中的输入文件是否存在。"
        elif "[ImportError]" in msg or "[ModuleNotFoundError]" in msg:
            hint = "\n\n请安装缺少的依赖：uv add <package>"
        elif "[NotImplementedError]" in msg:
            hint = "\n\n该算法尚未实现，请联系算法开发者。"
        QMessageBox.critical(self, "运行失败", msg + hint)

    def _on_pipeline_finished(self, saved: dict) -> None:
        elapsed_ms = self._elapsed_timer.elapsed()
        self._pipe_run_btn.setEnabled(True)
        self._pipe_run_btn.setText("▶  运行流水线")
        self._progress.setVisible(False)
        self._elapsed.setText(f"{elapsed_ms / 1000:.1f}s")
        self._status_msg.setText("流水线完成")

        if saved:
            self._log_info(f"流水线完成（{elapsed_ms}ms）：" + "  ".join(saved.keys()))
            csv_paths = [v for v in saved.values() if v.endswith(".csv")]
            if csv_paths:
                self._canvas.load_and_plot_csv(csv_paths[:5])
        else:
            self._log_info("流水线完成，但没有输出文件。")
        self._refresh_project_tree()

    def _on_pipeline_error(self, msg: str) -> None:
        self._pipe_run_btn.setEnabled(True)
        self._pipe_run_btn.setText("▶  运行流水线")
        self._progress.setVisible(False)
        self._status_msg.setText("失败")
        self._log_info(f"[错误] {msg}")
        QMessageBox.critical(self, "流水线失败", msg)

    def _on_import_data(self) -> None:
        borehole = self._bh_combo.currentText()
        if borehole.startswith("（"):
            QMessageBox.warning(self, "提示", "请先在工作区创建钻孔目录，再导入数据。")
            return

        exts = supported_extensions()
        all_glob = " ".join(f"*{e}" for e in exts)
        filters = (
            f"所有支持格式 ({all_glob})"
            ";;LAS 测井文件 (*.las)"
            ";;SEG-Y 地震数据 (*.segy *.sgy)"
            ";;文本数据 (*.csv *.txt *.dat *.asc)"
            ";;自定义二进制 (*.bin)"
        )
        data_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", filters)
        if not data_path:
            return

        suffix = Path(data_path).suffix.lower()
        default_method = Method.LOGGING if suffix == ".las" else Method.TEM
        method, ok = _ask_method(self, default_method)
        if not ok:
            return

        try:
            importer = DataImporter(self._runner.workspace_dir)
            channels = importer.import_file(Path(data_path), borehole, method)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            return

        self._log_info(f"导入成功：{len(channels)} 通道 → {borehole}/{method}/raw/")
        self._refresh_project_tree()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _log_info(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _on_create_borehole(self) -> None:
        name, ok = QInputDialog.getText(self, "新建钻孔", "请输入钻孔名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        store = DataStore(self._runner.workspace_dir)
        store.init_workspace()
        store.create_borehole(name)
        self._log_info(f"钻孔 '{name}' 创建成功")

        # Refresh borehole dropdown
        bh_list = store.list_boreholes()
        self._bh_combo.clear()
        if bh_list:
            self._bh_combo.addItems(bh_list)
            self._bh_combo.setCurrentText(name)
        else:
            self._bh_combo.addItem("（无钻孔，请先创建）")
        self._refresh_project_tree()

    def _on_open_dev_guide(self) -> None:
        guide_path = Path(__file__).resolve().parent.parent.parent / "docs" / "developer_guide.md"
        if guide_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide_path)))
        else:
            QMessageBox.information(self, "提示", f"开发者指南未找到：\n{guide_path}")

    def _reset_layout(self) -> None:
        self._dock_algo.setFloating(False)
        self._dock_log.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_algo)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_log)
