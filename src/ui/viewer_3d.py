"""PyVista 三维可视化引擎。

参考 EMIGMA / Voxler / Petrel 的场景管理模式：
- SceneObject 场景对象注册表（网格、钻孔、地形、剖面、标注各为独立对象）
- 每个对象可独立控制可见性、透明度、色标
- 支持 clip（裁切）而非仅 slice（切片）
- 坐标网格边框 + 北向标注 + 比例尺
- 多道测井沿孔展示

需要安装 3D 可选依赖：uv pip install geophys-tool[3d]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

_HAS_PYVISTA = False
try:
    import pyvista
    from pyvistaqt import QtInteractor
    _HAS_PYVISTA = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
# 场景对象模型（EMIGMA / Voxler 的 scene graph 理念）
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SceneObject:
    """场景中一个可管理的渲染对象。"""
    name: str
    category: str          # "mesh" | "borehole" | "surface" | "slice" | "annotation"
    visible: bool = True
    opacity: float = 1.0
    cmap: str = "jet_r"
    log_scale: bool = False
    actors: list[Any] = field(default_factory=list)
    vtk_data: Any = None   # 原始 VTK 数据（用于重新渲染）
    scalars: str = ""
    metadata: dict = field(default_factory=dict)


class Viewer3D(QWidget):
    """井中地球物理三维可视化器（场景图管理）。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._objects: dict[str, SceneObject] = {}
        self._primary_dataset: Any = None
        self._scalar_names: list[str] = []
        self._current_scalars: str = ""

        if not _HAS_PYVISTA:
            placeholder = QLabel(
                "三维可视化需要安装 3D 依赖：\n\n"
                "  uv pip install geophys-tool[3d]\n\n"
                "安装后重启应用即可使用。"
            )
            placeholder.setWordWrap(True)
            placeholder.setStyleSheet(
                "color: #888; font-size: 13px; padding: 40px; background: #f0f0f0;"
            )
            layout.addWidget(placeholder)
            self._plotter = None
            return

        self._plotter = QtInteractor(self)
        layout.addWidget(self._plotter.interactor)
        self._plotter.set_background("#f7f9fc")
        self._setup_lighting()

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._plotter is not None

    @property
    def scalar_names(self) -> list[str]:
        return list(self._scalar_names)

    @property
    def bounds(self) -> tuple[float, ...] | None:
        if self._primary_dataset is not None:
            return tuple(self._primary_dataset.bounds)
        return None

    @property
    def scene_objects(self) -> dict[str, SceneObject]:
        return dict(self._objects)

    # ── 场景管理 ──────────────────────────────────────────────────────

    def clear(self) -> None:
        if self._plotter:
            self._plotter.clear()
        self._objects.clear()
        self._primary_dataset = None
        self._scalar_names.clear()
        self._current_scalars = ""
        if self._plotter:
            self._add_bounds_axes()

    def set_object_visible(self, name: str, visible: bool) -> None:
        """EMIGMA 式对象可见性切换。"""
        obj = self._objects.get(name)
        if not obj:
            return
        obj.visible = visible
        for actor in obj.actors:
            actor.SetVisibility(visible)

    def remove_object(self, name: str) -> None:
        obj = self._objects.pop(name, None)
        if obj and self._plotter:
            for actor in obj.actors:
                self._plotter.remove_actor(actor)

    def rebuild_scene(self) -> None:
        """完整重建场景（属性/色标/透明度变化后调用）。"""
        if not self._plotter:
            return
        self._plotter.clear()
        self._add_bounds_axes()
        for obj in self._objects.values():
            if not obj.visible:
                continue
            self._render_object(obj)

    # ── 网格 ──────────────────────────────────────────────────────────

    def load_mesh(
        self,
        mesh: Any,
        name: str = "模型网格",
        scalars: str = "",
        values: np.ndarray | None = None,
        cmap: str = "jet_r",
        opacity: float = 0.85,
        log_scale: bool = False,
        clim: tuple[float, float] | None = None,
        exclude_air: bool = True,
    ) -> SceneObject:
        """加载网格到场景。支持 discretize mesh 和 PyVista dataset。"""
        if not self._plotter:
            return SceneObject(name=name, category="mesh")

        if hasattr(mesh, "to_vtk"):
            models = {scalars: values} if values is not None and scalars else {}
            vtk_obj = mesh.to_vtk(models=models)
        else:
            vtk_obj = mesh
            if values is not None and scalars:
                vtk_obj[scalars] = values

        # 排除空气层（lithology < 0）
        if exclude_air and "lithology" in (list(vtk_obj.cell_data.keys()) + list(vtk_obj.point_data.keys())):
            active = np.asarray(vtk_obj["lithology"]) >= 0
            vtk_obj = vtk_obj.extract_cells(np.where(active)[0])

        self._primary_dataset = vtk_obj
        self._scalar_names = list(vtk_obj.cell_data.keys()) + list(vtk_obj.point_data.keys())
        if not scalars and self._scalar_names:
            scalars = self._scalar_names[0]
        self._current_scalars = scalars

        obj = SceneObject(
            name=name, category="mesh", opacity=opacity,
            cmap=cmap, log_scale=log_scale, vtk_data=vtk_obj, scalars=scalars,
            metadata={"clim": clim},
        )
        self._objects[name] = obj
        self._render_mesh(obj)
        self._add_bounds_axes()
        self._plotter.reset_camera()
        return obj

    def load_file(self, path: str | Path) -> SceneObject:
        """从 VTK/VTU 文件加载网格。"""
        if not self._plotter:
            return SceneObject(name=str(path), category="mesh")
        dataset = pyvista.read(str(path))
        return self.load_mesh(dataset, name=Path(path).stem, exclude_air=True)

    def update_mesh_display(
        self,
        name: str = "",
        scalars: str = "",
        cmap: str = "jet_r",
        opacity: float = 0.85,
        log_scale: bool = False,
        clim: tuple[float, float] | None = None,
    ) -> None:
        """更新网格显示参数（不重新加载数据）。"""
        target = name or next(
            (n for n, o in self._objects.items() if o.category == "mesh"), ""
        )
        obj = self._objects.get(target)
        if not obj:
            return
        obj.scalars = scalars or obj.scalars
        obj.cmap = cmap
        obj.opacity = opacity
        obj.log_scale = log_scale
        obj.metadata["clim"] = clim
        self._current_scalars = obj.scalars
        self.rebuild_scene()

    # ── 钻孔 ──────────────────────────────────────────────────────────

    def add_borehole(
        self,
        trajectory: np.ndarray,
        name: str = "BH",
        radius: float = 3.0,
        color: str = "#2196F3",
        depth_interval: float = 100.0,
        logs: dict[str, np.ndarray] | None = None,
        log_cmap: str = "jet_r",
        log_radius: float = 6.0,
        formations: list[tuple[float, float, str, str]] | None = None,
    ) -> SceneObject:
        """添加钻孔到场景。

        EMIGMA 式多道测井 + 深度标注 + 地层分段。

        Args:
            logs: {"GR": values, "COND": values, ...} 多道测井数据，
                  沿轨迹着色管显示（每道偏移一定半径）
            formations: [(top, bot, name, color), ...] 地层分段着色
        """
        if not self._plotter:
            return SceneObject(name=name, category="borehole")

        obj = SceneObject(
            name=name, category="borehole", opacity=1.0,
            metadata={
                "trajectory": trajectory, "radius": radius,
                "color": color, "depth_interval": depth_interval,
                "logs": logs, "log_cmap": log_cmap, "log_radius": log_radius,
                "formations": formations,
            },
        )
        self._objects[name] = obj
        self._render_borehole(obj)
        return obj

    # ── 地形面 ────────────────────────────────────────────────────────

    def add_topography(
        self,
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        name: str = "地形面",
        opacity: float = 0.4,
        cmap: str = "terrain",
    ) -> SceneObject:
        if not self._plotter:
            return SceneObject(name=name, category="surface")
        grid = pyvista.StructuredGrid(x, y, z)
        obj = SceneObject(
            name=name, category="surface", opacity=opacity,
            cmap=cmap, vtk_data=grid,
        )
        self._objects[name] = obj
        actor = self._plotter.add_mesh(
            grid, cmap=cmap, opacity=opacity, show_edges=False, lighting=True,
        )
        obj.actors.append(actor)
        return obj

    # ── 剖面 ──────────────────────────────────────────────────────────

    def add_clip_plane(
        self,
        normal: str = "y",
        position: float | None = None,
        name: str = "",
        cmap: str = "jet_r",
        log_scale: bool = False,
    ) -> SceneObject | None:
        """添加裁切剖面。"""
        if not self._plotter or self._primary_dataset is None:
            return None
        if not name:
            name = f"剖面-{normal.upper()}"

        ds = self._primary_dataset
        center = list(ds.center)
        axis_map = {"x": 0, "y": 1, "z": 2}
        if position is not None:
            center[axis_map.get(normal, 1)] = position
        sliced = ds.slice(normal=normal, origin=tuple(center))

        scalars = self._current_scalars
        obj = SceneObject(
            name=name, category="slice", cmap=cmap,
            log_scale=log_scale, vtk_data=sliced, scalars=scalars,
            metadata={"normal": normal, "position": position or center[axis_map.get(normal, 1)]},
        )
        # 移除同方向旧剖面
        old = [n for n, o in self._objects.items() if o.category == "slice" and o.metadata.get("normal") == normal]
        for n in old:
            self.remove_object(n)
        self._objects[name] = obj
        actor = self._plotter.add_mesh(sliced, scalars=scalars, cmap=cmap, log_scale=log_scale)
        obj.actors.append(actor)
        return obj

    def update_clip_position(self, normal: str, position: float) -> None:
        self.add_clip_plane(normal=normal, position=position)

    # ── 标注 ──────────────────────────────────────────────────────────

    def add_north_arrow(self, scale: float = 50.0) -> None:
        if not self._plotter or self._primary_dataset is None:
            return
        self.remove_object("北向箭头")
        bounds = self._primary_dataset.bounds
        x_pos = bounds[1] + scale * 0.5
        y_pos, z_pos = bounds[2], bounds[5]

        arrow = pyvista.Arrow(start=(x_pos, y_pos, z_pos), direction=(0, 1, 0), scale=scale)
        obj = SceneObject(name="北向箭头", category="annotation")
        a1 = self._plotter.add_mesh(arrow, color="black")
        obj.actors.append(a1)
        self._plotter.add_point_labels(
            [(x_pos, y_pos + scale * 1.1, z_pos)], ["N"],
            font_size=16, text_color="black", bold=True, shape_opacity=0.0,
        )
        self._objects["北向箭头"] = obj

    def add_scale_bar(self, length: float = 200.0) -> None:
        if not self._plotter or self._primary_dataset is None:
            return
        self.remove_object("比例尺")
        bounds = self._primary_dataset.bounds
        x0 = bounds[0]
        y_pos = bounds[2] - length * 0.3
        z_pos = bounds[4]

        obj = SceneObject(name="比例尺", category="annotation")
        line = pyvista.Line((x0, y_pos, z_pos), (x0 + length, y_pos, z_pos))
        a = self._plotter.add_mesh(line, color="black", line_width=3)
        obj.actors.append(a)
        for frac, label in [(0, "0"), (0.5, f"{length/2:.0f}m"), (1.0, f"{length:.0f}m")]:
            self._plotter.add_point_labels(
                [(x0 + length * frac, y_pos, z_pos)], [label],
                font_size=9, text_color="black", shape_opacity=0.0,
            )
        self._objects["比例尺"] = obj

    # ── 投影模式 ──────────────────────────────────────────────────────

    def set_parallel_projection(self, enabled: bool) -> None:
        """正交/透视投影切换（Voxler 模式）。"""
        if self._plotter:
            self._plotter.camera.parallel_projection = enabled

    # ── 内部渲染 ──────────────────────────────────────────────────────

    def _setup_lighting(self) -> None:
        """Voxler 式三点光照。"""
        if not self._plotter:
            return
        self._plotter.remove_all_lights()
        key = pyvista.Light(position=(1, 1, 1), intensity=0.8, light_type="scenelight")
        fill = pyvista.Light(position=(-1, 0.5, 0.5), intensity=0.3, light_type="scenelight")
        back = pyvista.Light(position=(0, -1, -0.5), intensity=0.15, light_type="scenelight")
        for light in (key, fill, back):
            self._plotter.add_light(light)

    def _add_bounds_axes(self) -> None:
        """Voxler 式坐标网格边框。"""
        if not self._plotter:
            return
        if self._primary_dataset is not None:
            self._plotter.show_bounds(
                mesh=self._primary_dataset,
                grid="back", location="outer",
                font_size=8, color="#555555",
                xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
            )
        else:
            self._plotter.add_axes()

    def _render_object(self, obj: SceneObject) -> None:
        """按类别分发渲染。"""
        obj.actors.clear()
        if obj.category == "mesh":
            self._render_mesh(obj)
        elif obj.category == "borehole":
            self._render_borehole(obj)
        elif obj.category == "surface":
            if obj.vtk_data is not None:
                a = self._plotter.add_mesh(
                    obj.vtk_data, cmap=obj.cmap, opacity=obj.opacity,
                    show_edges=False, lighting=True,
                )
                obj.actors.append(a)
        elif obj.category == "slice":
            if obj.vtk_data is not None:
                a = self._plotter.add_mesh(
                    obj.vtk_data, scalars=obj.scalars,
                    cmap=obj.cmap, log_scale=obj.log_scale,
                )
                obj.actors.append(a)
        elif obj.category == "annotation":
            pass  # 标注在 rebuild 时重新添加

    def _render_mesh(self, obj: SceneObject) -> None:
        if not self._plotter or obj.vtk_data is None:
            return
        clim = obj.metadata.get("clim")
        actor = self._plotter.add_mesh(
            obj.vtk_data, scalars=obj.scalars, cmap=obj.cmap,
            show_edges=False, opacity=obj.opacity, log_scale=obj.log_scale,
            clim=clim,
            scalar_bar_args={"title": obj.scalars, "fmt": "%.2g", "n_labels": 6},
        )
        obj.actors.append(actor)

    def _render_borehole(self, obj: SceneObject) -> None:
        if not self._plotter:
            return
        m = obj.metadata
        traj = m["trajectory"]
        radius = m["radius"]
        color = m["color"]
        logs = m.get("logs")
        formations = m.get("formations")

        n_pts = max(len(traj), 150)

        # ── 主管（地层分段 或 单色） ──
        if formations:
            self._render_formations(obj, traj, radius, formations)
        else:
            line = pyvista.Spline(traj, n_points=n_pts)
            tube = line.tube(radius=radius)
            a = self._plotter.add_mesh(tube, color=color)
            obj.actors.append(a)

        # ── EMIGMA 式多道测井着色管 ──
        if logs:
            n_logs = len(logs)
            for i, (log_name, log_vals) in enumerate(logs.items()):
                offset_angle = 2 * np.pi * i / max(n_logs, 1)
                log_r = m.get("log_radius", 6.0)
                dx = log_r * 2.5 * np.cos(offset_angle)
                dy = log_r * 2.5 * np.sin(offset_angle)
                offset_traj = traj.copy()
                offset_traj[:, 0] += dx
                offset_traj[:, 1] += dy

                line = pyvista.Spline(offset_traj, n_points=n_pts)
                interp = np.interp(
                    np.linspace(0, 1, n_pts),
                    np.linspace(0, 1, len(log_vals)),
                    log_vals,
                )
                line[log_name] = interp
                tube = line.tube(radius=log_r)
                a = self._plotter.add_mesh(
                    tube, scalars=log_name, cmap=m.get("log_cmap", "jet_r"),
                    scalar_bar_args={"title": log_name, "fmt": "%.1f"},
                )
                obj.actors.append(a)

        # ── 孔口标签 ──
        self._plotter.add_point_labels(
            [traj[0]], [obj.name], font_size=14,
            point_color=color, text_color="black",
            bold=True, shape_opacity=0.6,
        )

        # ── 深度标注 ──
        interval = m.get("depth_interval", 0)
        if interval > 0:
            self._add_depth_markers(obj, traj, interval, color)

    def _render_formations(self, obj, traj, radius, formations):
        collar_z = traj[0, 2]
        depths = collar_z - traj[:, 2]
        total = depths[-1]
        for top, bot, fm_name, fm_color in formations:
            if top >= total or bot <= 0:
                continue
            idx = np.where((depths >= top) & (depths <= bot))[0]
            if len(idx) < 2:
                continue
            seg = traj[idx[0]:idx[-1]+1]
            line = pyvista.Spline(seg, n_points=max(len(seg), 20))
            tube = line.tube(radius=radius)
            a = self._plotter.add_mesh(tube, color=fm_color)
            obj.actors.append(a)
            # 地层标注文字
            mid_idx = idx[len(idx) // 2]
            self._plotter.add_point_labels(
                [traj[mid_idx] + [radius * 3, 0, 0]], [fm_name],
                font_size=8, text_color="#444444", shape_opacity=0.3,
            )

    def _add_depth_markers(self, obj, traj, interval, color):
        collar_z = traj[0, 2]
        depths = collar_z - traj[:, 2]
        pts, labels = [], []
        for d in np.arange(interval, depths[-1], interval):
            idx = np.argmin(np.abs(depths - d))
            pts.append(traj[idx])
            labels.append(f"{d:.0f}m")
        if pts:
            arr = np.array(pts)
            spheres = pyvista.PolyData(arr)
            a = self._plotter.add_mesh(spheres, color=color, point_size=5, render_points_as_spheres=True)
            obj.actors.append(a)
            self._plotter.add_point_labels(
                arr, labels, font_size=8, text_color="#333333",
                shape_opacity=0.3, always_visible=True,
            )
