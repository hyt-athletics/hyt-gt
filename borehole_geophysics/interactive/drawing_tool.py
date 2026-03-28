"""
2D交互绘图工具 — 交互式建模的核心

用matplotlib实现：
  - 鼠标点击画多边形/椭圆/矩形
  - 实时预览（橡皮筋线条）
  - 选择/删除地质体
  - 物性设置

状态机：
  IDLE → 按P → DRAW_POLY → 右键完成 → 输入属性 → IDLE
  IDLE → 按E → DRAW_ELLIPSE_CENTER → 点击 → DRAW_ELLIPSE_DRAG → 点击 → IDLE
  IDLE → 按R → DRAW_RECT_1 → 点击 → DRAW_RECT_2 → 点击 → IDLE
  IDLE → 按S → SELECT → 点击 → (选中一个体) → 按Delete删除 → SELECT
"""

import numpy as np

# 设置中文字体（必须在导入pyplot之前）
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Ellipse as MplEllipse
from matplotlib.collections import PatchCollection

from interactive.body_manager import ModelManager, GeologicalBody, BODY_COLORS


# 禁用matplotlib默认快捷键（和我们的冲突）
_keys_to_clear = [
    'keymap.save', 'keymap.grid', 'keymap.grid_minor',
    'keymap.home', 'keymap.pan', 'keymap.zoom',
    'keymap.xscale', 'keymap.yscale',
]
for _k in _keys_to_clear:
    try:
        plt.rcParams[_k] = []
    except (KeyError, ValueError):
        pass
# 保留 Ctrl+S 保存, Ctrl+Q 退出
plt.rcParams['keymap.quit'] = ['ctrl+q']


# ===================================================================
#  状态常量
# ===================================================================
IDLE = 'idle'
DRAW_POLY = 'draw_poly'
DRAW_ELLIPSE_CENTER = 'draw_ellipse_center'
DRAW_ELLIPSE_DRAG = 'draw_ellipse_drag'
DRAW_RECT_1 = 'draw_rect_1'
DRAW_RECT_2 = 'draw_rect_2'
SELECT = 'select'


class InteractiveDrawingTool:
    """
    2D交互绘图工具
    
    使用方法:
        tool = InteractiveDrawingTool()
        tool.run()   # 打开窗口，开始交互
    """
    
    def __init__(self, model_manager=None):
        """
        参数:
            model_manager: ModelManager对象。如果不传，自动创建一个。
        """
        self.manager = model_manager or ModelManager()
        
        # 状态
        self.mode = IDLE
        self.selected_idx = None       # 当前选中的地质体索引
        
        # 临时绘图数据（画到一半的东西）
        self._temp_vertices = []       # 多边形模式下已点击的顶点
        self._temp_center = None       # 椭圆/矩形的起始点
        
        # matplotlib 对象
        self.fig = None
        self.ax = None
        self._temp_artists = []        # 临时绘图对象（画完或取消后清除）
        self._rubber_band = None       # 橡皮筋线条
        self._status_text = None       # 底部状态栏
        self._help_text = None         # 帮助信息显示
        self._body_patches = []        # 地质体的图形对象
    
    # =============================================================
    #  启动和初始化
    # =============================================================
    
    def run(self):
        """启动交互工具"""
        self._print_terminal_help()
        self._setup_figure()
        plt.show()
    
    def _setup_figure(self):
        """创建和配置matplotlib图形窗口"""
        self.fig, self.ax = plt.subplots(figsize=(13, 9))
        self.fig.subplots_adjust(bottom=0.13, top=0.93)
        
        # 标题
        self.fig.suptitle('Interactive Geological Modeler', 
                         fontsize=14, fontweight='bold')
        
        # 坐标轴设置
        x_range = self.manager.domain['x_range']
        z_range = self.manager.domain['z_range']
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(z_range[1], z_range[0])  # Z轴翻转（深度向下）
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Z (m, depth ↓)', fontsize=12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # 画钻孔
        well = self.manager.well_info
        self.ax.axvline(x=well['x'], color='red', linewidth=2.5,
                       linestyle='-', alpha=0.8, zorder=10)
        self.ax.plot(well['x'], well['z_top'], 'rv', markersize=10, zorder=11)
        self.ax.annotate('Borehole', xy=(well['x'], well['z_top']),
                        xytext=(well['x'] + 30, well['z_top'] + 30),
                        fontsize=9, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        # 底部状态栏（两行）
        self._status_text = self.fig.text(
            0.5, 0.06,
            '',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                     edgecolor='gray', alpha=0.9)
        )
        self._shortcut_text = self.fig.text(
            0.5, 0.02,
            '[P]多边形  [E]椭圆  [R]矩形  [S]选择  [V]3D查看  '
            '[G]生成网格  [X]导出VTK  [H]帮助  [Ctrl+S]保存  [Ctrl+L]加载',
            ha='center', va='center', fontsize=8.5,
            color='#555555'
        )
        
        self._update_status('就绪 — 按 P 开始画多边形，按 H 查看帮助')
        
        # 连接事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        
        # 画已有的地质体
        self._redraw_bodies()
    
    # =============================================================
    #  事件处理：键盘
    # =============================================================
    
    def _on_key(self, event):
        """处理键盘按键"""
        if event.key is None:
            return
        
        key = event.key.lower()
        
        # ---------- 模式切换 ----------
        if key == 'p':
            self._enter_polygon_mode()
        elif key == 'e':
            self._enter_ellipse_mode()
        elif key == 'r':
            self._enter_rect_mode()
        elif key == 's' and self.mode not in (DRAW_POLY, DRAW_ELLIPSE_DRAG, DRAW_RECT_2):
            self._enter_select_mode()
        
        # ---------- 操作 ----------
        elif key == 'delete' or key == 'backspace':
            if self.mode == SELECT and self.selected_idx is not None:
                self._delete_selected()
            elif self.mode == DRAW_POLY and len(self._temp_vertices) > 0:
                self._polygon_undo_vertex()
        
        elif key == 'escape':
            self._cancel_current()
        
        elif key == 'enter':
            if self.mode == DRAW_POLY and len(self._temp_vertices) >= 3:
                self._polygon_finish()
        
        # ---------- 功能 ----------
        elif key == 'v':
            self._view_3d()
        elif key == 'g':
            self._generate_mesh()
        elif key == 'x':
            self._export_vtk()
        elif key == 'h':
            self._show_help_on_figure()
        elif key == 'ctrl+s':
            self._save_model()
        elif key == 'ctrl+l':
            self._load_model()
        elif key == 'i':
            self.manager.print_summary()
    
    # =============================================================
    #  事件处理：鼠标点击
    # =============================================================
    
    def _on_click(self, event):
        """处理鼠标点击"""
        # 忽略在坐标轴外的点击
        if event.inaxes != self.ax:
            return
        
        x, z = event.xdata, event.ydata
        
        # ---------- 多边形模式 ----------
        if self.mode == DRAW_POLY:
            if event.button == 1:       # 左键：添加顶点
                self._polygon_add_vertex(x, z)
            elif event.button == 3:     # 右键：完成
                if len(self._temp_vertices) >= 3:
                    self._polygon_finish()
                else:
                    self._update_status('至少需要3个点！继续点击添加顶点')
        
        # ---------- 椭圆模式 ----------
        elif self.mode == DRAW_ELLIPSE_CENTER:
            if event.button == 1:
                self._ellipse_set_center(x, z)
        elif self.mode == DRAW_ELLIPSE_DRAG:
            if event.button == 1:
                self._ellipse_finish(x, z)
            elif event.button == 3:
                self._cancel_current()
        
        # ---------- 矩形模式 ----------
        elif self.mode == DRAW_RECT_1:
            if event.button == 1:
                self._rect_set_corner1(x, z)
        elif self.mode == DRAW_RECT_2:
            if event.button == 1:
                self._rect_finish(x, z)
            elif event.button == 3:
                self._cancel_current()
        
        # ---------- 选择模式 ----------
        elif self.mode == SELECT:
            if event.button == 1:
                self._select_body_at(x, z)
    
    # =============================================================
    #  事件处理：鼠标移动（橡皮筋预览）
    # =============================================================
    
    def _on_move(self, event):
        """处理鼠标移动 — 更新预览"""
        if event.inaxes != self.ax:
            return
        
        x, z = event.xdata, event.ydata
        
        # 多边形：从最后一个顶点到光标的虚线
        if self.mode == DRAW_POLY and len(self._temp_vertices) > 0:
            last_x, last_z = self._temp_vertices[-1]
            first_x, first_z = self._temp_vertices[0]
            
            if self._rubber_band is None:
                self._rubber_band, = self.ax.plot(
                    [last_x, x, first_x], [last_z, z, first_z],
                    'k--', linewidth=1, alpha=0.4)
                self._temp_artists.append(self._rubber_band)
            else:
                self._rubber_band.set_data(
                    [last_x, x, first_x], [last_z, z, first_z])
            
            self.fig.canvas.draw_idle()
        
        # 椭圆：实时预览椭圆形状
        elif self.mode == DRAW_ELLIPSE_DRAG and self._temp_center is not None:
            cx, cz = self._temp_center
            a = abs(x - cx)
            b = abs(z - cz)
            if a > 5 and b > 5:  # 最小尺寸限制
                self._update_ellipse_preview(cx, cz, a, b)
        
        # 矩形：实时预览矩形
        elif self.mode == DRAW_RECT_2 and self._temp_center is not None:
            x1, z1 = self._temp_center
            self._update_rect_preview(x1, z1, x, z)
    
    # =============================================================
    #  多边形绘制
    # =============================================================
    
    def _enter_polygon_mode(self):
        """进入多边形绘制模式"""
        self._cancel_current()  # 先取消之前的操作
        self.mode = DRAW_POLY
        self._temp_vertices = []
        self._update_status('多边形模式 — 左键添加顶点 | 右键或Enter完成 | Esc取消')
    
    def _polygon_add_vertex(self, x, z):
        """添加一个多边形顶点"""
        self._temp_vertices.append((x, z))
        
        # 画顶点标记
        dot, = self.ax.plot(x, z, 'ko', markersize=7, zorder=20)
        self._temp_artists.append(dot)
        
        # 画到前一个顶点的连线
        if len(self._temp_vertices) > 1:
            prev_x, prev_z = self._temp_vertices[-2]
            line, = self.ax.plot([prev_x, x], [prev_z, z],
                               'k-', linewidth=1.5, zorder=19)
            self._temp_artists.append(line)
        
        # 删掉旧的橡皮筋（会在_on_move里重建）
        if self._rubber_band is not None:
            try:
                self._rubber_band.remove()
            except ValueError:
                pass
            self._rubber_band = None
            # 从_temp_artists里也移除
            self._temp_artists = [a for a in self._temp_artists 
                                  if a is not self._rubber_band]
        
        n = len(self._temp_vertices)
        self._update_status(f'多边形模式 — 已添加 {n} 个顶点 | 右键或Enter完成 | Esc取消')
        self.fig.canvas.draw_idle()
    
    def _polygon_undo_vertex(self):
        """撤销最后一个顶点"""
        if len(self._temp_vertices) == 0:
            return
        
        self._temp_vertices.pop()
        
        # 重画临时图形
        self._clear_temp()
        for i, (vx, vz) in enumerate(self._temp_vertices):
            dot, = self.ax.plot(vx, vz, 'ko', markersize=7, zorder=20)
            self._temp_artists.append(dot)
            if i > 0:
                prev_x, prev_z = self._temp_vertices[i - 1]
                line, = self.ax.plot([prev_x, vx], [prev_z, vz],
                                   'k-', linewidth=1.5, zorder=19)
                self._temp_artists.append(line)
        
        n = len(self._temp_vertices)
        self._update_status(f'多边形模式 — {n} 个顶点 | Delete撤销顶点 | Esc取消')
        self.fig.canvas.draw_idle()
    
    def _polygon_finish(self):
        """完成多边形绘制"""
        if len(self._temp_vertices) < 3:
            return
        
        vertices = self._temp_vertices.copy()
        self._clear_temp()
        
        # 请求属性
        props = self._ask_properties()
        if props is None:
            self._update_status('已取消')
            self.mode = IDLE
            return
        
        # 创建地质体
        body = GeologicalBody(
            name=props['name'],
            vertices_xz=vertices,
            density=props['density'],
            y_range=props['y_range'],
            color=self.manager.get_next_color(),
        )
        self.manager.add_body(body)
        
        self._redraw_bodies()
        self.mode = IDLE
        self._update_status(
            f'✅ 已创建 "{body.name}" (密度={body.density}) | 继续画或按V查看3D')
    
    # =============================================================
    #  椭圆绘制
    # =============================================================
    
    def _enter_ellipse_mode(self):
        """进入椭圆绘制模式"""
        self._cancel_current()
        self.mode = DRAW_ELLIPSE_CENTER
        self._update_status('椭圆模式 — 左键点击设置中心')
    
    def _ellipse_set_center(self, x, z):
        """设置椭圆中心"""
        self._temp_center = (x, z)
        
        dot, = self.ax.plot(x, z, 'k+', markersize=15, markeredgewidth=2, zorder=20)
        self._temp_artists.append(dot)
        self.fig.canvas.draw_idle()
        
        self.mode = DRAW_ELLIPSE_DRAG
        self._update_status('椭圆模式 — 移动鼠标调整大小，左键确认 | Esc取消')
    
    def _update_ellipse_preview(self, cx, cz, a, b):
        """更新椭圆预览"""
        # 移除旧预览
        if self._rubber_band is not None:
            try:
                self._rubber_band.remove()
            except ValueError:
                pass
            self._temp_artists = [art for art in self._temp_artists
                                  if art is not self._rubber_band]
        
        # 画椭圆预览
        angles = np.linspace(0, 2 * np.pi, 50)
        ex = cx + a * np.cos(angles)
        ez = cz + b * np.sin(angles)
        self._rubber_band, = self.ax.plot(ex, ez, 'k--', linewidth=1, alpha=0.5)
        self._temp_artists.append(self._rubber_band)
        self.fig.canvas.draw_idle()
    
    def _ellipse_finish(self, x, z):
        """完成椭圆绘制"""
        cx, cz = self._temp_center
        a = abs(x - cx)
        b = abs(z - cz)
        
        if a < 5 or b < 5:
            self._update_status('椭圆太小了！请拖远一点')
            return
        
        # 把椭圆转成多边形（36个顶点）
        n_pts = 36
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        vertices = [(cx + a * np.cos(t), cz + b * np.sin(t)) for t in angles]
        
        self._clear_temp()
        
        props = self._ask_properties()
        if props is None:
            self.mode = IDLE
            return
        
        body = GeologicalBody(
            name=props['name'],
            vertices_xz=vertices,
            density=props['density'],
            y_range=props['y_range'],
            color=self.manager.get_next_color(),
        )
        self.manager.add_body(body)
        
        self._redraw_bodies()
        self.mode = IDLE
        self._update_status(f'✅ 已创建椭圆 "{body.name}" | 继续画或按V查看3D')
    
    # =============================================================
    #  矩形绘制
    # =============================================================
    
    def _enter_rect_mode(self):
        """进入矩形绘制模式"""
        self._cancel_current()
        self.mode = DRAW_RECT_1
        self._update_status('矩形模式 — 左键点击设置第一个角')
    
    def _rect_set_corner1(self, x, z):
        """设置矩形第一个角"""
        self._temp_center = (x, z)
        
        dot, = self.ax.plot(x, z, 'k+', markersize=15, markeredgewidth=2, zorder=20)
        self._temp_artists.append(dot)
        self.fig.canvas.draw_idle()
        
        self.mode = DRAW_RECT_2
        self._update_status('矩形模式 — 移动鼠标，左键点击对角位置 | Esc取消')
    
    def _update_rect_preview(self, x1, z1, x2, z2):
        """更新矩形预览"""
        if self._rubber_band is not None:
            try:
                self._rubber_band.remove()
            except ValueError:
                pass
            self._temp_artists = [a for a in self._temp_artists
                                  if a is not self._rubber_band]
        
        rect_x = [x1, x2, x2, x1, x1]
        rect_z = [z1, z1, z2, z2, z1]
        self._rubber_band, = self.ax.plot(rect_x, rect_z, 'k--', linewidth=1, alpha=0.5)
        self._temp_artists.append(self._rubber_band)
        self.fig.canvas.draw_idle()
    
    def _rect_finish(self, x2, z2):
        """完成矩形绘制"""
        x1, z1 = self._temp_center
        
        if abs(x2 - x1) < 5 or abs(z2 - z1) < 5:
            self._update_status('矩形太小了！')
            return
        
        xmin, xmax = min(x1, x2), max(x1, x2)
        zmin, zmax = min(z1, z2), max(z1, z2)
        vertices = [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)]
        
        self._clear_temp()
        
        props = self._ask_properties()
        if props is None:
            self.mode = IDLE
            return
        
        body = GeologicalBody(
            name=props['name'],
            vertices_xz=vertices,
            density=props['density'],
            y_range=props['y_range'],
            color=self.manager.get_next_color(),
        )
        self.manager.add_body(body)
        
        self._redraw_bodies()
        self.mode = IDLE
        self._update_status(f'✅ 已创建矩形 "{body.name}" | 继续画或按V查看3D')
    
    # =============================================================
    #  选择和删除
    # =============================================================
    
    def _enter_select_mode(self):
        """进入选择模式"""
        self._cancel_current()
        self.mode = SELECT
        self.selected_idx = None
        self._redraw_bodies()
        self._update_status('选择模式 — 左键点击地质体选中 | Delete删除 | Esc退出')
    
    def _select_body_at(self, x, z):
        """选择点击位置的地质体"""
        idx = self.manager.find_body_at(x, z)
        self.selected_idx = idx
        self._redraw_bodies()
        
        if idx is not None:
            body = self.manager.bodies[idx]
            self._update_status(
                f'已选中 [{idx}] "{body.name}" (密度={body.density}) | '
                f'Delete删除 | D修改密度')
        else:
            self._update_status('未选中任何地质体 — 请点击在某个体内部')
    
    def _delete_selected(self):
        """删除选中的地质体"""
        if self.selected_idx is None:
            return
        
        body = self.manager.remove_body(self.selected_idx)
        if body:
            self.selected_idx = None
            self._redraw_bodies()
            self._update_status(f'🗑️ 已删除 "{body.name}"')
    
    # =============================================================
    #  通用工具方法
    # =============================================================
    
    def _cancel_current(self):
        """取消当前操作，回到IDLE"""
        self._clear_temp()
        self._temp_vertices = []
        self._temp_center = None
        self.mode = IDLE
        self._update_status('就绪 — 按 P/E/R 开始绘制，按 S 选择，按 H 帮助')
    
    def _clear_temp(self):
        """清除所有临时绘图对象"""
        for artist in self._temp_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._temp_artists = []
        self._rubber_band = None
        self.fig.canvas.draw_idle()
    
    def _update_status(self, text):
        """更新底部状态栏"""
        if self._status_text:
            self._status_text.set_text(text)
            self.fig.canvas.draw_idle()
    
    def _redraw_bodies(self):
        """重新绘制所有地质体"""
        # 移除旧图形
        for patch in self._body_patches:
            try:
                patch.remove()
            except (ValueError, AttributeError):
                pass
        self._body_patches = []
        
        # 画每个地质体
        for i, body in enumerate(self.manager.bodies):
            is_selected = (self.mode == SELECT and i == self.selected_idx)
            
            # 填充多边形
            polygon = MplPolygon(
                body.vertices_xz, closed=True,
                facecolor=body.color, alpha=0.35,
                edgecolor=body.color if not is_selected else 'black',
                linewidth=1.5 if not is_selected else 3,
                linestyle='-' if not is_selected else '--',
                zorder=5
            )
            self.ax.add_patch(polygon)
            self._body_patches.append(polygon)
            
            # 标签（名字+密度）
            cx, cz = body.centroid_2d()
            label = self.ax.text(
                cx, cz,
                f'{body.name}\nρ={body.density:.2f}',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2',
                         facecolor=body.color, alpha=0.7),
                zorder=6
            )
            self._body_patches.append(label)
        
        self.fig.canvas.draw_idle()
    
    # =============================================================
    #  属性输入对话框
    # =============================================================
    
    def _ask_properties(self):
        """
        弹出对话框，获取地质体属性
        
        尝试用tkinter对话框，失败则用终端输入
        """
        default_name = self.manager.get_next_name()
        
        try:
            return self._ask_properties_tkinter(default_name)
        except Exception:
            return self._ask_properties_terminal(default_name)
    
    def _ask_properties_tkinter(self, default_name):
        """用tkinter对话框获取属性"""
        import tkinter as tk
        from tkinter import simpledialog
        
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        
        name = simpledialog.askstring(
            "地质体名称",
            "请输入名称：",
            initialvalue=default_name,
            parent=root
        )
        if name is None:
            root.destroy()
            return None
        
        density = simpledialog.askfloat(
            "密度差",
            "请输入密度差 (g/cm³):\n"
            "正值 = 高密度（矿体）\n"
            "负值 = 低密度（空洞）",
            initialvalue=0.5,
            parent=root
        )
        if density is None:
            density = 0.5
        
        y_min = simpledialog.askfloat(
            "Y方向起始",
            "Y方向起始位置 (m):\n（沿钻孔方向的延伸范围）",
            initialvalue=300.0,
            parent=root
        )
        if y_min is None:
            y_min = 300.0
        
        y_max = simpledialog.askfloat(
            "Y方向结束",
            "Y方向结束位置 (m):",
            initialvalue=700.0,
            parent=root
        )
        if y_max is None:
            y_max = 700.0
        
        root.destroy()
        
        return {
            'name': name,
            'density': density,
            'y_range': (min(y_min, y_max), max(y_min, y_max)),
        }
    
    def _ask_properties_terminal(self, default_name):
        """用终端输入获取属性（备用方案）"""
        print(f"\n{'='*45}")
        print("  请在终端中输入地质体属性")
        print(f"{'='*45}")
        
        name = input(f"  名称 [{default_name}]: ").strip()
        if not name:
            name = default_name
        
        try:
            density = float(input("  密度差 g/cm³ [0.5]: ").strip() or "0.5")
        except ValueError:
            density = 0.5
        
        try:
            y_min = float(input("  Y起始 m [300]: ").strip() or "300")
        except ValueError:
            y_min = 300.0
        
        try:
            y_max = float(input("  Y结束 m [700]: ").strip() or "700")
        except ValueError:
            y_max = 700.0
        
        print(f"  ✅ {name}: 密度={density}, Y={y_min}~{y_max}m")
        print(f"{'='*45}\n")
        
        return {
            'name': name,
            'density': density,
            'y_range': (min(y_min, y_max), max(y_min, y_max)),
        }
    
    # =============================================================
    #  功能按钮：3D查看、网格生成、保存加载
    # =============================================================
    
    def _view_3d(self):
        """在PyVista中查看3D模型"""
        if len(self.manager.bodies) == 0:
            self._update_status('⚠️ 没有地质体，请先画一个！')
            return
        
        self._update_status('正在打开3D查看器...')
        
        from interactive.model_viewer import show_model_3d
        from borehole.well import VerticalWell
        
        well = VerticalWell(**self.manager.well_info)
        show_model_3d(self.manager.bodies, well, self.manager.domain)
        
        self._update_status('3D查看器已关闭 — 继续编辑')
    
    def _generate_mesh(self):
        """生成网格"""
        if len(self.manager.bodies) == 0:
            self._update_status('⚠️ 没有地质体，请先画一个！')
            return
        
        self._update_status('正在生成网格...')
        
        from interactive.mesh_integration import generate_mesh_from_model
        from borehole.well import VerticalWell
        
        well = VerticalWell(**self.manager.well_info)
        generate_mesh_from_model(self.manager, well)
        
        self._update_status('网格生成完成！')
    
    def _export_vtk(self):
        """导出VTK文件"""
        if len(self.manager.bodies) == 0:
            self._update_status('⚠️ 没有地质体，请先画一个！')
            return
        
        self._update_status('正在导出VTK文件...')
        
        from export.vtk_export import VTKExporter
        from borehole.well import VerticalWell
        from mesh.octree import OctreeMesh
        
        well = VerticalWell(**self.manager.well_info)
        domain = self.manager.domain
        
        # 生成网格
        mesh = OctreeMesh(
            x_range=domain['x_range'],
            y_range=domain['y_range'],
            z_range=domain['z_range'],
            max_level=5
        )
        mesh.refine_along_borehole(well, levels_and_radii=[
            (5, 50), (4, 150), (3, 300),
        ])
        for body in self.manager.bodies:
            mesh.refine_at_boundary(body.contains_point_3d, target_level=4)
        mesh.balance()
        for body in self.manager.bodies:
            mesh.assign_property(body.contains_point_3d, 'density', body.density)
        
        # 导出
        exporter = VTKExporter(output_dir='vtk_output')
        exporter.export_all(
            octree_mesh=mesh,
            well=well,
            bodies=self.manager.bodies,
        )
        
        self._update_status('✅ VTK文件已导出到 vtk_output/ 文件夹')
    
    def _save_model(self):
        """保存模型"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.asksaveasfilename(
                defaultextension='.json',
                filetypes=[('JSON', '*.json')],
                title='保存模型'
            )
            root.destroy()
        except Exception:
            filename = input("  输入保存文件名 [model.json]: ").strip() or "model.json"
        
        if filename:
            self.manager.save(filename)
            self._update_status(f'✅ 模型已保存: {filename}')
    
    def _load_model(self):
        """加载模型"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(
                filetypes=[('JSON', '*.json')],
                title='加载模型'
            )
            root.destroy()
        except Exception:
            filename = input("  输入文件名: ").strip()
        
        if filename and self.manager.load(filename):
            self._redraw_bodies()
            self._update_status(f'✅ 模型已加载，{len(self.manager.bodies)} 个地质体')
    
    # =============================================================
    #  帮助
    # =============================================================
    
    def _show_help_on_figure(self):
        """在图上显示帮助"""
        if self._help_text is not None:
            try:
                self._help_text.remove()
            except (ValueError, AttributeError):
                pass
            self._help_text = None
            self.fig.canvas.draw_idle()
            return
        
        help_msg = (
            "━━━ 快捷键帮助 ━━━\n"
            "\n"
            "绘制模式:\n"
            "  P → 画多边形（最常用）\n"
            "  E → 画椭圆\n"
            "  R → 画矩形\n"
            "\n"
            "编辑:\n"
            "  S     → 选择模式\n"
            "  Delete→ 删除选中体\n"
            "  Esc   → 取消/退出\n"
            "\n"
            "功能:\n"
            "  V      → 3D查看\n"
            "  G      → 生成网格\n"
            "  X      → 导出VTK\n"
            "  I      → 打印模型信息\n"
            "  Ctrl+S → 保存模型\n"
            "  Ctrl+L → 加载模型\n"
            "  H      → 开/关帮助\n"
            "\n"
            "鼠标:\n"
            "  左键 → 放置点/选择\n"
            "  右键 → 完成绘制\n"
            "\n"
            "按 H 关闭此帮助"
        )
        
        self._help_text = self.ax.text(
            0.02, 0.98, help_msg,
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                     edgecolor='gray', alpha=0.95),
            zorder=100
        )
        self.fig.canvas.draw_idle()
    
    def _print_terminal_help(self):
        """在终端打印帮助"""
        print("""
╔════════════════════════════════════════════════════════════════╗
║              交互式地质建模工具 — 操作指南                       ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  绘制地质体:                                                    ║
║    P → 多边形模式（左键加点，右键完成）                            ║
║    E → 椭圆模式（左键定中心，移动鼠标，左键确认大小）              ║
║    R → 矩形模式（左键定角，移动鼠标，左键定对角）                  ║
║                                                                ║
║  编辑:                                                          ║
║    S      → 进入选择模式，点击选中地质体                          ║
║    Delete → 删除选中的地质体                                     ║
║    Esc    → 取消当前操作                                        ║
║                                                                ║
║  功能:                                                          ║
║    V      → 3D查看（弹出PyVista窗口）                           ║
║    G      → 生成网格                                            ║
║    X      → 导出VTK文件                                         ║
║    I      → 终端打印模型信息                                    ║
║    Ctrl+S → 保存模型到文件                                      ║
║    Ctrl+L → 从文件加载模型                                      ║
║    H      → 在图上显示/隐藏帮助                                  ║
║                                                                ║
║  提示: 窗口中按 H 可以随时查看帮助                                ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
        """)