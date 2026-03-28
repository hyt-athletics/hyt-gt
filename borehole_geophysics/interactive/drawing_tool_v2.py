"""
升级版交互绘图工具 v2

新增功能：
  - 撤销/重做 (Ctrl+Z / Ctrl+Y)
  - 拖拽移动地质体 (M模式)
  - 顶点编辑 (T模式)
  - 地层线绘制 (L模式)
  - 双击编辑物性
  - 复制粘贴 (Ctrl+C / Ctrl+V)
  - 网格吸附 (按G切换)
  - 实时坐标显示
  - 右键菜单
"""

import numpy as np
import copy
import matplotlib

# 设置中文字体（必须在导入pyplot之前）
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from interactive.body_manager import ModelManager, GeologicalBody, BODY_COLORS
from interactive.commands import (
    CommandHistory, AddBodyCommand, DeleteBodyCommand,
    MoveBodyCommand, MoveVertexCommand, EditPropertyCommand,
    AddLayerCommand, PasteBodyCommand
)
from interactive.layer_tool import GeoLayer, layers_to_bodies


# 清除matplotlib默认快捷键
_keys_to_clear = [
    'keymap.save', 'keymap.grid', 'keymap.grid_minor',
    'keymap.home', 'keymap.pan', 'keymap.zoom',
    'keymap.xscale', 'keymap.yscale', 'keymap.fullscreen',
]
for _k in _keys_to_clear:
    try:
        plt.rcParams[_k] = []
    except (KeyError, ValueError):
        pass
plt.rcParams['keymap.quit'] = ['ctrl+q']


# 模式常量
IDLE = 'idle'
DRAW_POLY = 'draw_poly'
DRAW_ELLIPSE_CENTER = 'draw_ellipse_center'
DRAW_ELLIPSE_DRAG = 'draw_ellipse_drag'
DRAW_RECT_1 = 'draw_rect_1'
DRAW_RECT_2 = 'draw_rect_2'
SELECT = 'select'
MOVE = 'move'
MOVE_DRAGGING = 'move_dragging'
EDIT_VERTEX = 'edit_vertex'
VERTEX_DRAGGING = 'vertex_dragging'
DRAW_LAYER = 'draw_layer'


class InteractiveDrawingToolV2:
    """
    升级版交互绘图工具
    
    使用: tool = InteractiveDrawingToolV2(); tool.run()
    """
    
    def __init__(self, model_manager=None, on_compute=None, on_export=None):
        """
        参数:
            model_manager: ModelManager
            on_compute: 按C时调用的回调函数
            on_export: 按X时调用的回调函数
        """
        self.manager = model_manager or ModelManager()
        if not hasattr(self.manager, 'layers'):
            self.manager.layers = []
        
        self.history = CommandHistory(max_size=100)
        self.clipboard = None       # 复制的地质体
        
        # 回调
        self.on_compute = on_compute
        self.on_export = on_export
        
        # 状态
        self.mode = IDLE
        self.selected_idx = None
        self.snap_to_grid = False
        self.grid_spacing = 50      # 吸附间距
        
        # 临时数据
        self._temp_vertices = []
        self._temp_center = None
        self._drag_start = None       # 拖拽起始位置
        self._drag_body_idx = None    # 正在拖拽的体索引
        self._drag_vertex_idx = None  # 正在拖拽的顶点索引
        
        # 图形对象
        self.fig = None
        self.ax = None
        self._temp_artists = []
        self._rubber_band = None
        self._body_patches = []
        self._layer_artists = []
        self._vertex_markers = []
        self._status_text = None
        self._coord_text = None
        self._help_text = None
    
    # =============================================================
    #  启动
    # =============================================================
    
    def run(self):
        self._print_help()
        self._setup_figure()
        plt.show()
    
    def _setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.subplots_adjust(bottom=0.12, top=0.93, left=0.08, right=0.95)
        
        self.fig.suptitle('Interactive Geological Modeler v2', fontsize=14, fontweight='bold')
        
        x_range = self.manager.domain['x_range']
        z_range = self.manager.domain['z_range']
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(z_range[1], z_range[0])
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Z (m, depth ↓)', fontsize=12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # 钻孔
        well = self.manager.well_info
        self.ax.axvline(x=well['x'], color='red', linewidth=2.5, alpha=0.8, zorder=10)
        self.ax.plot(well['x'], well['z_top'], 'rv', markersize=10, zorder=11)
        
        # 状态栏
        self._status_text = self.fig.text(
            0.5, 0.06, '', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                     edgecolor='gray', alpha=0.9)
        )
        
        # 坐标显示
        self._coord_text = self.fig.text(
            0.95, 0.06, 'X: --- Z: ---', ha='right', va='center', fontsize=9,
            color='#555'
        )
        
        # 快捷键提示
        self.fig.text(
            0.5, 0.02,
            '[P]多边形 [E]椭圆 [R]矩形 [L]地层线 [S]选择 [M]移动 [T]顶点编辑 '
            '[C]正演 [V]3D [X]VTK [H]帮助',
            ha='center', va='center', fontsize=8, color='#555'
        )
        
        self._update_status('就绪 — 按H查看帮助')
        
        # 事件连接
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        
        self._redraw_all()
    
    # =============================================================
    #  坐标吸附
    # =============================================================
    
    def _snap(self, x, z):
        """如果开启了吸附，将坐标对齐到网格"""
        if self.snap_to_grid:
            g = self.grid_spacing
            x = round(x / g) * g
            z = round(z / g) * g
        return x, z
    
    # =============================================================
    #  键盘事件
    # =============================================================
    
    def _on_key(self, event):
        if event.key is None:
            return
        key = event.key.lower() if len(event.key) > 1 or not event.key[0].isupper() else event.key
        
        # 撤销/重做
        if key == 'ctrl+z':
            self._do_undo()
        elif key == 'ctrl+y':
            self._do_redo()
        
        # 复制/粘贴
        elif key == 'ctrl+c':
            self._do_copy()
        elif key == 'ctrl+v':
            self._do_paste()
        
        # 保存/加载
        elif key == 'ctrl+s':
            self._save_model()
        elif key == 'ctrl+l':
            self._load_model()
        
        # 模式切换
        elif key == 'p':
            self._enter_mode_polygon()
        elif key == 'e':
            self._enter_mode_ellipse()
        elif key == 'r':
            self._enter_mode_rect()
        elif key == 'l':
            self._enter_mode_layer()
        elif key == 's':
            self._enter_mode_select()
        elif key == 'm':
            self._enter_mode_move()
        elif key == 't':
            self._enter_mode_vertex_edit()
        
        # 操作
        elif key in ('delete', 'backspace'):
            self._do_delete()
        elif key == 'escape':
            self._cancel_current()
        elif key == 'enter':
            self._do_finish()
        
        # 功能
        elif key == 'c':
            self._do_compute()
        elif key == 'v':
            self._do_view_3d()
        elif key == 'x':
            self._do_export_vtk()
        elif key == 'g':
            self._toggle_grid_snap()
        elif key == 'h':
            self._toggle_help()
        elif key == 'i':
            self.manager.print_summary()
    
    # =============================================================
    #  鼠标点击
    # =============================================================
    
    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        x, z = self._snap(event.xdata, event.ydata)
        
        # 双击 → 编辑物性
        if event.dblclick and event.button == 1:
            self._do_double_click(x, z)
            return
        
        # ---------- 各模式处理 ----------
        if self.mode == DRAW_POLY:
            if event.button == 1:
                self._poly_add_vertex(x, z)
            elif event.button == 3:
                self._poly_finish()
        
        elif self.mode == DRAW_ELLIPSE_CENTER:
            if event.button == 1:
                self._ellipse_set_center(x, z)
        
        elif self.mode == DRAW_ELLIPSE_DRAG:
            if event.button == 1:
                self._ellipse_finish(x, z)
            elif event.button == 3:
                self._cancel_current()
        
        elif self.mode == DRAW_RECT_1:
            if event.button == 1:
                self._rect_set_corner1(x, z)
        
        elif self.mode == DRAW_RECT_2:
            if event.button == 1:
                self._rect_finish(x, z)
            elif event.button == 3:
                self._cancel_current()
        
        elif self.mode == SELECT:
            if event.button == 1:
                self._select_at(x, z)
        
        elif self.mode == MOVE:
            if event.button == 1:
                self._move_start(x, z)
        
        elif self.mode == EDIT_VERTEX:
            if event.button == 1:
                self._vertex_click(x, z)
        
        elif self.mode == DRAW_LAYER:
            if event.button == 1:
                self._layer_add_point(x, z)
            elif event.button == 3:
                self._layer_finish()
    
    # =============================================================
    #  鼠标释放（拖拽结束）
    # =============================================================
    
    def _on_release(self, event):
        if event.inaxes != self.ax:
            return
        
        x, z = self._snap(event.xdata, event.ydata)
        
        if self.mode == MOVE_DRAGGING:
            self._move_end(x, z)
        elif self.mode == VERTEX_DRAGGING:
            self._vertex_drag_end(x, z)
    
    # =============================================================
    #  鼠标移动
    # =============================================================
    
    def _on_move(self, event):
        if event.inaxes != self.ax:
            return
        
        x, z = event.xdata, event.ydata
        
        # 更新坐标显示
        if self._coord_text:
            snap_marker = " [SNAP]" if self.snap_to_grid else ""
            self._coord_text.set_text(f'X:{x:.0f} Z:{z:.0f}{snap_marker}')
        
        sx, sz = self._snap(x, z)
        
        # 橡皮筋
        if self.mode == DRAW_POLY and self._temp_vertices:
            self._update_rubber_band_polygon(sx, sz)
        elif self.mode == DRAW_ELLIPSE_DRAG and self._temp_center:
            self._update_rubber_band_ellipse(sx, sz)
        elif self.mode == DRAW_RECT_2 and self._temp_center:
            self._update_rubber_band_rect(sx, sz)
        elif self.mode == DRAW_LAYER and self._temp_vertices:
            self._update_rubber_band_layer(sx, sz)
        
        # 拖拽
        elif self.mode == MOVE_DRAGGING and self._drag_start:
            self._move_preview(sx, sz)
        elif self.mode == VERTEX_DRAGGING and self._drag_start:
            self._vertex_drag_preview(sx, sz)
        
        self.fig.canvas.draw_idle()
    
    # =============================================================
    #  撤销/重做
    # =============================================================
    
    def _do_undo(self):
        success, msg = self.history.undo()
        if success:
            self._redraw_all()
            self._update_status(f'↩ {msg}')
        else:
            self._update_status(msg)
    
    def _do_redo(self):
        success, msg = self.history.redo()
        if success:
            self._redraw_all()
            self._update_status(f'↪ {msg}')
        else:
            self._update_status(msg)
    
    # =============================================================
    #  复制/粘贴
    # =============================================================
    
    def _do_copy(self):
        if self.selected_idx is not None and self.selected_idx < len(self.manager.bodies):
            self.clipboard = self.manager.bodies[self.selected_idx]
            self._update_status(f'📋 已复制 "{self.clipboard.name}"')
        else:
            self._update_status('⚠️ 请先选择一个地质体 (S模式)')
    
    def _do_paste(self):
        if self.clipboard is None:
            self._update_status('⚠️ 剪贴板为空，请先 Ctrl+C 复制')
            return
        cmd = PasteBodyCommand(self.manager, self.clipboard, offset_x=50, offset_z=50)
        self.history.execute(cmd)
        self._redraw_all()
        self._update_status(f'📋 已粘贴 "{self.clipboard.name}" (偏移50m)')
    
    # =============================================================
    #  多边形绘制
    # =============================================================
    
    def _enter_mode_polygon(self):
        self._cancel_current()
        self.mode = DRAW_POLY
        self._temp_vertices = []
        self._update_status('🔷 多边形 — 左键加点 | 右键/Enter完成 | Delete撤销 | Esc取消')
    
    def _poly_add_vertex(self, x, z):
        self._temp_vertices.append((x, z))
        dot, = self.ax.plot(x, z, 'ko', markersize=7, zorder=20)
        self._temp_artists.append(dot)
        
        if len(self._temp_vertices) > 1:
            px, pz = self._temp_vertices[-2]
            line, = self.ax.plot([px, x], [pz, z], 'k-', linewidth=1.5, zorder=19)
            self._temp_artists.append(line)
        
        self._clear_rubber_band()
        n = len(self._temp_vertices)
        self._update_status(f'🔷 多边形 — {n}个点 | 右键/Enter完成 | Delete撤销')
        self.fig.canvas.draw_idle()
    
    def _poly_finish(self):
        if len(self._temp_vertices) < 3:
            self._update_status('⚠️ 至少需要3个点')
            return
        
        verts = self._temp_vertices.copy()
        self._clear_temp()
        
        props = self._ask_properties()
        if props is None:
            self.mode = IDLE
            self._update_status('已取消')
            return
        
        body = GeologicalBody(
            name=props['name'], vertices_xz=verts,
            density=props['density'], y_range=props['y_range'],
            color=self.manager.get_next_color()
        )
        cmd = AddBodyCommand(self.manager, body)
        self.history.execute(cmd)
        
        self._redraw_all()
        self.mode = IDLE
        self._update_status(f'✅ 已创建 "{body.name}" (ρ={body.density})')
    
    # =============================================================
    #  椭圆绘制
    # =============================================================
    
    def _enter_mode_ellipse(self):
        self._cancel_current()
        self.mode = DRAW_ELLIPSE_CENTER
        self._update_status('⭕ 椭圆 — 左键点击设置中心')
    
    def _ellipse_set_center(self, x, z):
        self._temp_center = (x, z)
        dot, = self.ax.plot(x, z, 'k+', markersize=15, markeredgewidth=2, zorder=20)
        self._temp_artists.append(dot)
        self.mode = DRAW_ELLIPSE_DRAG
        self._update_status('⭕ 椭圆 — 移动鼠标调整大小 | 左键确认')
        self.fig.canvas.draw_idle()
    
    def _ellipse_finish(self, x, z):
        cx, cz = self._temp_center
        a, b = abs(x - cx), abs(z - cz)
        if a < 10 or b < 10:
            self._update_status('⚠️ 椭圆太小')
            return
        
        n_pts = 36
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        verts = [(cx + a * np.cos(t), cz + b * np.sin(t)) for t in angles]
        self._clear_temp()
        
        props = self._ask_properties()
        if props is None:
            self.mode = IDLE
            return
        
        body = GeologicalBody(
            name=props['name'], vertices_xz=verts,
            density=props['density'], y_range=props['y_range'],
            color=self.manager.get_next_color()
        )
        cmd = AddBodyCommand(self.manager, body)
        self.history.execute(cmd)
        
        self._redraw_all()
        self.mode = IDLE
        self._update_status(f'✅ 已创建椭圆 "{body.name}"')
    
    # =============================================================
    #  矩形绘制
    # =============================================================
    
    def _enter_mode_rect(self):
        self._cancel_current()
        self.mode = DRAW_RECT_1
        self._update_status('▬ 矩形 — 左键设置第一个角')
    
    def _rect_set_corner1(self, x, z):
        self._temp_center = (x, z)
        dot, = self.ax.plot(x, z, 'k+', markersize=15, markeredgewidth=2, zorder=20)
        self._temp_artists.append(dot)
        self.mode = DRAW_RECT_2
        self._update_status('▬ 矩形 — 移动鼠标 | 左键设置对角')
        self.fig.canvas.draw_idle()
    
    def _rect_finish(self, x2, z2):
        x1, z1 = self._temp_center
        if abs(x2 - x1) < 10 or abs(z2 - z1) < 10:
            self._update_status('⚠️ 矩形太小')
            return
        xmin, xmax = min(x1, x2), max(x1, x2)
        zmin, zmax = min(z1, z2), max(z1, z2)
        verts = [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)]
        self._clear_temp()
        
        props = self._ask_properties()
        if props is None:
            self.mode = IDLE
            return
        
        body = GeologicalBody(
            name=props['name'], vertices_xz=verts,
            density=props['density'], y_range=props['y_range'],
            color=self.manager.get_next_color()
        )
        cmd = AddBodyCommand(self.manager, body)
        self.history.execute(cmd)
        
        self._redraw_all()
        self.mode = IDLE
        self._update_status(f'✅ 已创建矩形 "{body.name}"')
    
    # =============================================================
    #  选择模式
    # =============================================================
    
    def _enter_mode_select(self):
        self._cancel_current()
        self.mode = SELECT
        self.selected_idx = None
        self._redraw_all()
        self._update_status('👆 选择 — 左键选中 | Delete删除 | 双击编辑 | Esc退出')
    
    def _select_at(self, x, z):
        idx = self.manager.find_body_at(x, z)
        self.selected_idx = idx
        self._redraw_all()
        if idx is not None:
            b = self.manager.bodies[idx]
            self._update_status(
                f'✓ 选中 [{idx}] "{b.name}" ρ={b.density} | Delete删 | Ctrl+C复制')
        else:
            self._update_status('👆 未选中 — 请点击地质体内部')
    
    # =============================================================
    #  移动模式（拖拽整个地质体）
    # =============================================================
    
    def _enter_mode_move(self):
        self._cancel_current()
        self.mode = MOVE
        self._update_status('✥ 移动 — 在地质体上按住左键拖拽 | Esc退出')
    
    def _move_start(self, x, z):
        idx = self.manager.find_body_at(x, z)
        if idx is None:
            self._update_status('✥ 请在地质体上按下鼠标')
            return
        
        self._drag_body_idx = idx
        self._drag_start = (x, z)
        self.mode = MOVE_DRAGGING
        b = self.manager.bodies[idx]
        self._update_status(f'✥ 正在拖拽 "{b.name}" — 松开鼠标完成')
    
    def _move_preview(self, x, z):
        """拖拽过程中的实时预览"""
        if self._drag_start is None or self._drag_body_idx is None:
            return
        
        sx, sz = self._drag_start
        dx, dz = x - sx, z - sz
        
        body = self.manager.bodies[self._drag_body_idx]
        preview_verts = [(vx + dx, vz + dz) for vx, vz in body.vertices_xz]
        
        self._clear_rubber_band()
        xs = [v[0] for v in preview_verts] + [preview_verts[0][0]]
        zs = [v[1] for v in preview_verts] + [preview_verts[0][1]]
        self._rubber_band, = self.ax.plot(xs, zs, 'k--', linewidth=2, alpha=0.5, zorder=25)
        self._temp_artists.append(self._rubber_band)
    
    def _move_end(self, x, z):
        if self._drag_start is None or self._drag_body_idx is None:
            self.mode = MOVE
            return
        
        sx, sz = self._drag_start
        dx, dz = x - sx, z - sz
        
        if abs(dx) > 1 or abs(dz) > 1:  # 最小移动距离
            cmd = MoveBodyCommand(self.manager, self._drag_body_idx, dx, dz)
            self.history.execute(cmd)
        
        self._drag_start = None
        self._drag_body_idx = None
        self._clear_temp()
        self._redraw_all()
        self.mode = MOVE
        self._update_status('✥ 移动完成 | 继续拖拽其他体 | Ctrl+Z撤销')
    
    # =============================================================
    #  顶点编辑模式（拖拽单个顶点）
    # =============================================================
    
    def _enter_mode_vertex_edit(self):
        self._cancel_current()
        self.mode = EDIT_VERTEX
        self.selected_idx = None
        self._update_status('✎ 顶点编辑 — 先点击选中地质体，再拖拽顶点')
    
    def _vertex_click(self, x, z):
        # 先看是否点击了某个已显示的顶点
        if self.selected_idx is not None:
            body = self.manager.bodies[self.selected_idx]
            for vi, (vx, vz) in enumerate(body.vertices_xz):
                if abs(x - vx) < 15 and abs(z - vz) < 15:
                    # 开始拖拽这个顶点
                    self._drag_body_idx = self.selected_idx
                    self._drag_vertex_idx = vi
                    self._drag_start = (vx, vz)
                    self.mode = VERTEX_DRAGGING
                    self._update_status(f'✎ 拖拽顶点 {vi} — 松开鼠标完成')
                    return
        
        # 没点到顶点，尝试选择新的体
        idx = self.manager.find_body_at(x, z)
        if idx is not None:
            self.selected_idx = idx
            self._redraw_all()
            self._show_vertices(idx)
            b = self.manager.bodies[idx]
            self._update_status(f'✎ 选中 "{b.name}" — 拖拽蓝色方块编辑顶点')
        else:
            self.selected_idx = None
            self._redraw_all()
            self._update_status('✎ 请点击地质体选中它')
    
    def _show_vertices(self, body_idx):
        """显示地质体的顶点标记"""
        body = self.manager.bodies[body_idx]
        for vi, (vx, vz) in enumerate(body.vertices_xz):
            marker, = self.ax.plot(vx, vz, 's', color='blue',
                                  markersize=8, markeredgecolor='white',
                                  markeredgewidth=1.5, zorder=30)
            self._vertex_markers.append(marker)
        self.fig.canvas.draw_idle()
    
    def _vertex_drag_preview(self, x, z):
        """顶点拖拽预览"""
        if self._drag_body_idx is None or self._drag_vertex_idx is None:
            return
        
        body = self.manager.bodies[self._drag_body_idx]
        preview_verts = list(body.vertices_xz)
        preview_verts[self._drag_vertex_idx] = (x, z)
        
        self._clear_rubber_band()
        xs = [v[0] for v in preview_verts] + [preview_verts[0][0]]
        zs = [v[1] for v in preview_verts] + [preview_verts[0][1]]
        self._rubber_band, = self.ax.plot(xs, zs, 'b--', linewidth=2, alpha=0.5, zorder=25)
        self._temp_artists.append(self._rubber_band)
    
    def _vertex_drag_end(self, x, z):
        if self._drag_body_idx is None or self._drag_vertex_idx is None:
            self.mode = EDIT_VERTEX
            return
        
        cmd = MoveVertexCommand(
            self.manager, self._drag_body_idx,
            self._drag_vertex_idx, x, z
        )
        self.history.execute(cmd)
        
        self._drag_body_idx = None
        self._drag_vertex_idx = None
        self._drag_start = None
        self._clear_temp()
        self._redraw_all()
        
        if self.selected_idx is not None:
            self._show_vertices(self.selected_idx)
        
        self.mode = EDIT_VERTEX
        self._update_status('✎ 顶点已更新 | 继续编辑 | Ctrl+Z撤销')
    
    # =============================================================
    #  地层线绘制
    # =============================================================
    
    def _enter_mode_layer(self):
        self._cancel_current()
        self.mode = DRAW_LAYER
        self._temp_vertices = []
        self._update_status('━ 地层线 — 从左到右依次点击控制点 | 右键/Enter完成')
    
    def _layer_add_point(self, x, z):
        self._temp_vertices.append((x, z))
        dot, = self.ax.plot(x, z, 'D', color='brown', markersize=8, zorder=20)
        self._temp_artists.append(dot)
        
        if len(self._temp_vertices) > 1:
            px, pz = self._temp_vertices[-2]
            line, = self.ax.plot([px, x], [pz, z], '-', color='brown',
                               linewidth=2, zorder=19)
            self._temp_artists.append(line)
        
        self._clear_rubber_band()
        n = len(self._temp_vertices)
        self._update_status(f'━ 地层线 — {n}个控制点 | 右键/Enter完成')
        self.fig.canvas.draw_idle()
    
    def _layer_finish(self):
        if len(self._temp_vertices) < 2:
            self._update_status('⚠️ 至少需要2个控制点')
            return
        
        pts = self._temp_vertices.copy()
        self._clear_temp()
        
        # 请求地层线属性
        props = self._ask_layer_properties()
        if props is None:
            self.mode = IDLE
            return
        
        layer = GeoLayer(
            name=props['name'],
            control_points=pts,
            density_above=props['density_above'],
            density_below=props['density_below'],
            y_range=props.get('y_range', (0, 1000)),
            color=props.get('color', '#795548')
        )
        
        cmd = AddLayerCommand(self.manager, layer)
        self.history.execute(cmd)
        
        self._redraw_all()
        self.mode = IDLE
        self._update_status(f'✅ 地层线 "{layer.name}" (上:{layer.density_above} 下:{layer.density_below})')
    
    # =============================================================
    #  双击编辑物性
    # =============================================================
    
    def _do_double_click(self, x, z):
        idx = self.manager.find_body_at(x, z)
        if idx is None:
            return
        
        body = self.manager.bodies[idx]
        
        try:
            import tkinter as tk
            from tkinter import simpledialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            new_density = simpledialog.askfloat(
                "编辑物性",
                f"地质体: {body.name}\n"
                f"当前密度差: {body.density} g/cm³\n\n"
                f"输入新的密度差:",
                initialvalue=body.density,
                parent=root
            )
            
            if new_density is not None:
                new_name = simpledialog.askstring(
                    "编辑名称",
                    f"当前名称: {body.name}\n输入新名称 (留空不改):",
                    initialvalue=body.name,
                    parent=root
                )
                
                if new_density != body.density:
                    cmd = EditPropertyCommand(self.manager, idx, 'density', new_density)
                    self.history.execute(cmd)
                
                if new_name and new_name != body.name:
                    cmd2 = EditPropertyCommand(self.manager, idx, 'name', new_name)
                    self.history.execute(cmd2)
                
                self._redraw_all()
                self._update_status(f'✅ 已更新 "{body.name}" ρ={body.density}')
            
            root.destroy()
        except Exception:
            # 终端备用
            print(f"\n  编辑: {body.name}")
            try:
                new_d = float(input(f"  密度差 [{body.density}]: ").strip() or str(body.density))
                if new_d != body.density:
                    cmd = EditPropertyCommand(self.manager, idx, 'density', new_d)
                    self.history.execute(cmd)
                    self._redraw_all()
                    self._update_status(f'✅ 密度已更新: {new_d}')
            except ValueError:
                pass
    
    # =============================================================
    #  删除
    # =============================================================
    
    def _do_delete(self):
        if self.mode == DRAW_POLY and self._temp_vertices:
            self._temp_vertices.pop()
            self._refresh_temp_polygon()
            self._update_status(f'🔷 撤销了一个点，剩余 {len(self._temp_vertices)} 个')
            return
        
        if self.mode == DRAW_LAYER and self._temp_vertices:
            self._temp_vertices.pop()
            self._refresh_temp_layer()
            self._update_status(f'━ 撤销了一个点，剩余 {len(self._temp_vertices)} 个')
            return
        
        if self.selected_idx is not None and self.selected_idx < len(self.manager.bodies):
            cmd = DeleteBodyCommand(self.manager, self.selected_idx)
            self.history.execute(cmd)
            self.selected_idx = None
            self._redraw_all()
            self._update_status(f'🗑️ {cmd.description} | Ctrl+Z 可撤销')
    
    def _do_finish(self):
        if self.mode == DRAW_POLY:
            self._poly_finish()
        elif self.mode == DRAW_LAYER:
            self._layer_finish()
    
    # =============================================================
    #  网格吸附开关
    # =============================================================
    
    def _toggle_grid_snap(self):
        self.snap_to_grid = not self.snap_to_grid
        if self.snap_to_grid:
            self._update_status(f'📐 网格吸附已开启 (间距{self.grid_spacing}m)')
        else:
            self._update_status('📐 网格吸附已关闭')
    
    # =============================================================
    #  橡皮筋预览
    # =============================================================
    
    def _update_rubber_band_polygon(self, x, z):
        last = self._temp_vertices[-1]
        first = self._temp_vertices[0]
        self._clear_rubber_band()
        self._rubber_band, = self.ax.plot(
            [last[0], x, first[0]], [last[1], z, first[1]],
            'k--', linewidth=1, alpha=0.4)
        self._temp_artists.append(self._rubber_band)
    
    def _update_rubber_band_ellipse(self, x, z):
        cx, cz = self._temp_center
        a, b = abs(x - cx), abs(z - cz)
        if a < 5 or b < 5:
            return
        self._clear_rubber_band()
        angles = np.linspace(0, 2 * np.pi, 50)
        self._rubber_band, = self.ax.plot(
            cx + a * np.cos(angles), cz + b * np.sin(angles),
            'k--', linewidth=1, alpha=0.5)
        self._temp_artists.append(self._rubber_band)
    
    def _update_rubber_band_rect(self, x, z):
        x1, z1 = self._temp_center
        self._clear_rubber_band()
        self._rubber_band, = self.ax.plot(
            [x1, x, x, x1, x1], [z1, z1, z, z, z1],
            'k--', linewidth=1, alpha=0.5)
        self._temp_artists.append(self._rubber_band)
    
    def _update_rubber_band_layer(self, x, z):
        last = self._temp_vertices[-1]
        self._clear_rubber_band()
        self._rubber_band, = self.ax.plot(
            [last[0], x], [last[1], z],
            '--', color='brown', linewidth=1.5, alpha=0.5)
        self._temp_artists.append(self._rubber_band)
    
    # =============================================================
    #  功能按钮
    # =============================================================
    
    def _do_compute(self):
        if self.on_compute:
            self._update_status('正在计算...')
            self.on_compute()
            self._update_status('✅ 计算完成')
        else:
            self._update_status('⚠️ 正演计算未连接')
    
    def _do_view_3d(self):
        if len(self.manager.bodies) == 0 and not getattr(self.manager, 'layers', []):
            self._update_status('⚠️ 没有地质体')
            return
        self._update_status('打开3D查看器...')
        from interactive.model_viewer import show_model_3d
        from borehole.well import VerticalWell
        
        # 合并地质体和地层线生成的体
        all_bodies = list(self.manager.bodies)
        if hasattr(self.manager, 'layers') and self.manager.layers:
            layer_bodies = layers_to_bodies(
                self.manager.layers,
                self.manager.domain['x_range'],
                self.manager.domain['z_range']
            )
            all_bodies.extend(layer_bodies)
        
        well = VerticalWell(**self.manager.well_info)
        show_model_3d(all_bodies, well, self.manager.domain)
        self._update_status('3D查看器已关闭')
    
    def _do_export_vtk(self):
        if self.on_export:
            self._update_status('正在导出VTK...')
            self.on_export()
            self._update_status('✅ VTK导出完成')
        else:
            self._update_status('⚠️ VTK导出未连接')
    
    def _save_model(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            fn = filedialog.asksaveasfilename(
                defaultextension='.json', filetypes=[('JSON', '*.json')])
            root.destroy()
        except Exception:
            fn = input("  文件名 [model.json]: ").strip() or "model.json"
        if fn:
            self.manager.save(fn)
            self._update_status(f'✅ 已保存: {fn}')
    
    def _load_model(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            fn = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
            root.destroy()
        except Exception:
            fn = input("  文件名: ").strip()
        if fn and self.manager.load(fn):
            self.history = CommandHistory()  # 重置历史
            self._redraw_all()
            self._update_status(f'✅ 已加载 {len(self.manager.bodies)} 个地质体')
    
    # =============================================================
    #  取消和清理
    # =============================================================
    
    def _cancel_current(self):
        self._clear_temp()
        self._clear_vertex_markers()
        self._temp_vertices = []
        self._temp_center = None
        self._drag_start = None
        self._drag_body_idx = None
        self._drag_vertex_idx = None
        self.mode = IDLE
        self._redraw_all()
        undo_hint = f" | Ctrl+Z撤销'{self.history.undo_description}'" if self.history.can_undo else ""
        self._update_status(f'就绪{undo_hint}')
    
    def _clear_temp(self):
        for a in self._temp_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._temp_artists = []
        self._rubber_band = None
        self.fig.canvas.draw_idle()
    
    def _clear_rubber_band(self):
        if self._rubber_band is not None:
            try:
                self._rubber_band.remove()
            except (ValueError, AttributeError):
                pass
            self._temp_artists = [a for a in self._temp_artists if a is not self._rubber_band]
            self._rubber_band = None
    
    def _clear_vertex_markers(self):
        for m in self._vertex_markers:
            try:
                m.remove()
            except (ValueError, AttributeError):
                pass
        self._vertex_markers = []
    
    def _refresh_temp_polygon(self):
        """多边形模式下删除点后刷新临时图形"""
        self._clear_temp()
        for i, (vx, vz) in enumerate(self._temp_vertices):
            dot, = self.ax.plot(vx, vz, 'ko', markersize=7, zorder=20)
            self._temp_artists.append(dot)
            if i > 0:
                px, pz = self._temp_vertices[i - 1]
                line, = self.ax.plot([px, vx], [pz, vz], 'k-', linewidth=1.5, zorder=19)
                self._temp_artists.append(line)
        self.fig.canvas.draw_idle()
    
    def _refresh_temp_layer(self):
        """地层线模式下删除点后刷新"""
        self._clear_temp()
        for i, (vx, vz) in enumerate(self._temp_vertices):
            dot, = self.ax.plot(vx, vz, 'D', color='brown', markersize=8, zorder=20)
            self._temp_artists.append(dot)
            if i > 0:
                px, pz = self._temp_vertices[i - 1]
                line, = self.ax.plot([px, vx], [pz, vz], '-', color='brown',
                                   linewidth=2, zorder=19)
                self._temp_artists.append(line)
        self.fig.canvas.draw_idle()
    
    # =============================================================
    #  重绘所有内容
    # =============================================================
    
    def _redraw_all(self):
        """重绘所有地质体和地层线"""
        # 清除旧图形
        for p in self._body_patches:
            try:
                p.remove()
            except (ValueError, AttributeError):
                pass
        self._body_patches = []
        
        for a in self._layer_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._layer_artists = []
        
        self._clear_vertex_markers()
        
        # 画地质体
        for i, body in enumerate(self.manager.bodies):
            is_selected = (self.mode in (SELECT, EDIT_VERTEX) and i == self.selected_idx)
            
            polygon = MplPolygon(
                body.vertices_xz, closed=True,
                facecolor=body.color, alpha=0.35,
                edgecolor='black' if is_selected else body.color,
                linewidth=3 if is_selected else 1.5,
                linestyle='--' if is_selected else '-',
                zorder=5
            )
            self.ax.add_patch(polygon)
            self._body_patches.append(polygon)
            
            cx, cz = body.centroid_2d()
            label = self.ax.text(
                cx, cz, f'{body.name}\nρ={body.density:.2f}',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=body.color, alpha=0.7),
                zorder=6
            )
            self._body_patches.append(label)
        
        # 画地层线
        if hasattr(self.manager, 'layers'):
            x_range = self.manager.domain['x_range']
            for layer in self.manager.layers:
                xs, zs = layer.get_dense_points(x_range)
                line, = self.ax.plot(xs, zs, '-', color=layer.color,
                                   linewidth=2.5, zorder=8)
                self._layer_artists.append(line)
                
                # 控制点标记
                for cp_x, cp_z in layer.control_points:
                    marker, = self.ax.plot(cp_x, cp_z, 'D', color=layer.color,
                                          markersize=6, zorder=9)
                    self._layer_artists.append(marker)
                
                # 标签
                mid_x = (x_range[0] + x_range[1]) / 2
                mid_z = layer.z_at_x(mid_x)
                lbl = self.ax.text(
                    mid_x, mid_z - 20,
                    f'{layer.name}\n↑ρ={layer.density_above} ↓ρ={layer.density_below}',
                    ha='center', va='top', fontsize=7,
                    color=layer.color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    zorder=9
                )
                self._layer_artists.append(lbl)
        
        self.fig.canvas.draw_idle()
    
    # =============================================================
    #  属性输入
    # =============================================================
    
    def _ask_properties(self):
        default_name = self.manager.get_next_name()
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
            name = simpledialog.askstring("名称", "地质体名称:", initialvalue=default_name, parent=root)
            if name is None: root.destroy(); return None
            density = simpledialog.askfloat("密度差", "密度差 (g/cm³):", initialvalue=0.5, parent=root)
            if density is None: density = 0.5
            y_min = simpledialog.askfloat("Y起始", "Y起始 (m):", initialvalue=300.0, parent=root)
            y_max = simpledialog.askfloat("Y结束", "Y结束 (m):", initialvalue=700.0, parent=root)
            root.destroy()
            return {'name': name, 'density': density, 'y_range': (min(y_min or 300, y_max or 700), max(y_min or 300, y_max or 700))}
        except Exception:
            return self._ask_properties_terminal(default_name)
    
    def _ask_properties_terminal(self, default_name):
        print(f"\n{'='*40}")
        name = input(f"  名称 [{default_name}]: ").strip() or default_name
        try:
            density = float(input("  密度差 [0.5]: ").strip() or "0.5")
        except ValueError:
            density = 0.5
        try:
            y_min = float(input("  Y起始 [300]: ").strip() or "300")
            y_max = float(input("  Y结束 [700]: ").strip() or "700")
        except ValueError:
            y_min, y_max = 300, 700
        print(f"{'='*40}\n")
        return {'name': name, 'density': density, 'y_range': (min(y_min, y_max), max(y_min, y_max))}
    
    def _ask_layer_properties(self):
        n_layers = len(getattr(self.manager, 'layers', []))
        default_name = f"layer_{n_layers + 1}"
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
            name = simpledialog.askstring("地层线名称", "名称:", initialvalue=default_name, parent=root)
            if name is None: root.destroy(); return None
            da = simpledialog.askfloat("上方密度差", "线上方密度差 (g/cm³):", initialvalue=0.0, parent=root)
            db = simpledialog.askfloat("下方密度差", "线下方密度差 (g/cm³):", initialvalue=0.1, parent=root)
            root.destroy()
            return {'name': name, 'density_above': da or 0.0, 'density_below': db or 0.0}
        except Exception:
            print(f"\n{'='*40}")
            name = input(f"  地层线名称 [{default_name}]: ").strip() or default_name
            try:
                da = float(input("  上方密度差 [0.0]: ").strip() or "0.0")
                db = float(input("  下方密度差 [0.1]: ").strip() or "0.1")
            except ValueError:
                da, db = 0.0, 0.1
            print(f"{'='*40}\n")
            return {'name': name, 'density_above': da, 'density_below': db}
    
    # =============================================================
    #  帮助
    # =============================================================
    
    def _toggle_help(self):
        if self._help_text is not None:
            try:
                self._help_text.remove()
            except (ValueError, AttributeError):
                pass
            self._help_text = None
            self.fig.canvas.draw_idle()
            return
        
        self._help_text = self.ax.text(
            0.02, 0.98,
            "━━ 快捷键帮助 ━━\n"
            "\n"
            "绘制:\n"
            "  P → 多边形\n"
            "  E → 椭圆\n"
            "  R → 矩形\n"
            "  L → 地层线\n"
            "\n"
            "编辑:\n"
            "  S     → 选择\n"
            "  M     → 拖拽移动\n"
            "  T     → 顶点编辑\n"
            "  双击  → 编辑物性\n"
            "  Del   → 删除/撤销点\n"
            "\n"
            "操作:\n"
            "  Ctrl+Z → 撤销\n"
            "  Ctrl+Y → 重做\n"
            "  Ctrl+C → 复制\n"
            "  Ctrl+V → 粘贴\n"
            "  Ctrl+S → 保存\n"
            "  Ctrl+L → 加载\n"
            "  G      → 网格吸附\n"
            "  Esc    → 取消\n"
            "\n"
            "功能:\n"
            "  C → 正演计算\n"
            "  V → 3D查看\n"
            "  X → VTK导出\n"
            "  I → 打印信息\n"
            "\n按 H 关闭帮助",
            transform=self.ax.transAxes,
            fontsize=8.5, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.95),
            zorder=100
        )
        self.fig.canvas.draw_idle()
    
    def _update_status(self, text):
        if self._status_text:
            self._status_text.set_text(text)
            self.fig.canvas.draw_idle()
    
    def _print_help(self):
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║              交互式地质建模工具 v2 — 操作指南                       ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  绘制:  P=多边形  E=椭圆  R=矩形  L=地层线                        ║
║  编辑:  S=选择  M=移动  T=顶点编辑  双击=改物性                    ║
║  操作:  Ctrl+Z=撤销  Ctrl+Y=重做  Ctrl+C/V=复制粘贴               ║
║  功能:  C=正演  V=3D  X=VTK  G=网格吸附  H=帮助                   ║
║  保存:  Ctrl+S=保存  Ctrl+L=加载                                  ║
║                                                                   ║
║  窗口中按 H 随时查看完整帮助                                       ║
╚═══════════════════════════════════════════════════════════════════╝
        """)