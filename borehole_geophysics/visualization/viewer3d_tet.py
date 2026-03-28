"""
3D四面体网格可视化

用PyVista画四面体网格
"""

import numpy as np
import pyvista as pv


def show_tet_mesh(mesh_3d, property_name=None, property_values=None,
                  well=None, show_edges=True, opacity=0.3,
                  show_surface_only=True):
    """
    可视化3D四面体网格
    
    参数:
        mesh_3d: Delaunay3D对象
        property_name: 物性名称（用于标题）
        property_values: 每个四面体的物性值数组
        well: 钻孔对象
        show_surface_only: True=只显示外表面（推荐，否则太密了看不清）
    """
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # 构建PyVista非结构化网格
    grid = _build_pyvista_tet_grid(mesh_3d, property_values, property_name)
    
    if show_surface_only:
        # 只画外表面（推荐）
        surface = grid.extract_surface()
        if property_values is not None and property_name:
            plotter.add_mesh(surface, scalars=property_name,
                           cmap='RdYlBu_r', show_edges=show_edges,
                           opacity=opacity, edge_color='gray',
                           scalar_bar_args={'title': property_name or ''})
        else:
            plotter.add_mesh(surface, color='lightblue',
                           show_edges=show_edges, opacity=opacity,
                           edge_color='gray')
    else:
        # 画所有四面体（可能很密）
        if property_values is not None and property_name:
            plotter.add_mesh(grid, scalars=property_name,
                           cmap='RdYlBu_r', show_edges=show_edges,
                           opacity=opacity)
        else:
            plotter.add_mesh(grid, color='lightblue',
                           show_edges=show_edges, opacity=opacity)
    
    # 画钻孔
    if well is not None:
        _add_well_to_plotter(plotter, well)
    
    plotter.add_axes()
    n_tets = len(mesh_3d.tetrahedra)
    n_pts = len(mesh_3d.points)
    plotter.add_text(f"Tet mesh: {n_tets} tets, {n_pts} points",
                    position='upper_left', font_size=10)
    plotter.show()


def show_tet_cross_section(mesh_3d, normal='y', origin=None,
                            property_values=None, property_name=None,
                            well=None):
    """
    切一刀看内部！
    
    参数:
        normal: 切面法线方向 'x', 'y', 或 'z'
        origin: 切面经过的点 (x,y,z)，默认是模型中心
    """
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    grid = _build_pyvista_tet_grid(mesh_3d, property_values, property_name)
    
    if origin is None:
        origin = mesh_3d.points.mean(axis=0)
    
    # 切面
    normal_vec = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[normal]
    
    clipped = grid.clip(normal=normal_vec, origin=origin)
    
    if property_values is not None and property_name:
        plotter.add_mesh(clipped, scalars=property_name,
                        cmap='RdYlBu_r', show_edges=True,
                        edge_color='gray', line_width=0.5,
                        scalar_bar_args={'title': property_name})
    else:
        plotter.add_mesh(clipped, color='lightblue',
                        show_edges=True, edge_color='gray')
    
    # 边框
    plotter.add_mesh(grid.outline(), color='black', line_width=2)
    
    if well is not None:
        _add_well_to_plotter(plotter, well)
    
    plotter.add_axes()
    plotter.add_text(f"Cross section: {normal}={origin[{'x':0,'y':1,'z':2}[normal]]:.0f}m",
                    position='upper_left', font_size=10)
    plotter.show()


def show_tet_quality(mesh_3d, quality_data):
    """
    用颜色显示四面体质量分布
    """
    plotter = pv.Plotter(shape=(1, 2))
    
    # 左：质量值
    plotter.subplot(0, 0)
    plotter.set_background('white')
    grid1 = _build_pyvista_tet_grid(mesh_3d, quality_data['qualities'], 'quality')
    clipped1 = grid1.clip(normal=[0, 1, 0], origin=mesh_3d.points.mean(axis=0))
    plotter.add_mesh(clipped1, scalars='quality', cmap='RdYlGn',
                    clim=[0, 1], show_edges=True, edge_color='gray',
                    scalar_bar_args={'title': '质量 (0=差, 1=好)'})
    plotter.add_text("质量分布", position='upper_left')
    plotter.add_axes()
    
    # 右：体积
    plotter.subplot(0, 1)
    plotter.set_background('white')
    grid2 = _build_pyvista_tet_grid(mesh_3d, np.log10(quality_data['volumes'] + 1), 'log_volume')
    clipped2 = grid2.clip(normal=[0, 1, 0], origin=mesh_3d.points.mean(axis=0))
    plotter.add_mesh(clipped2, scalars='log_volume', cmap='viridis',
                    show_edges=True, edge_color='gray',
                    scalar_bar_args={'title': '体积 (log10)'})
    plotter.add_text("体积分布", position='upper_left')
    plotter.add_axes()
    
    plotter.show()


def show_tet_anomaly_body(mesh_3d, property_values, property_name='density',
                           threshold=0.01, well=None):
    """
    只显示异常体（过滤掉背景）
    
    最适合查看赋值后的效果
    """
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    grid = _build_pyvista_tet_grid(mesh_3d, property_values, property_name)
    
    # 过滤：只显示绝对值大于阈值的四面体
    threshed = grid.threshold(value=threshold, scalars=property_name)
    
    if threshed.n_cells > 0:
        plotter.add_mesh(threshed, scalars=property_name,
                        cmap='RdYlBu_r', show_edges=True,
                        opacity=0.8, edge_color='gray',
                        scalar_bar_args={'title': property_name})
    
    # 半透明边框
    plotter.add_mesh(grid.outline(), color='gray', line_width=2)
    
    if well is not None:
        _add_well_to_plotter(plotter, well)
    
    plotter.add_axes()
    n_anomaly = threshed.n_cells if threshed.n_cells > 0 else 0
    plotter.add_text(f"异常体: {n_anomaly} tets", position='upper_left')
    plotter.show()


# ==========================================================
#  内部辅助函数
# ==========================================================

def _build_pyvista_tet_grid(mesh_3d, values=None, name=None):
    """把我们的四面体网格转成PyVista格式"""
    points = mesh_3d.points
    tets = mesh_3d.tetrahedra
    
    if len(tets) == 0:
        raise ValueError("四面体网格为空，无法可视化")
    
    cells = []
    celltypes = []
    for tet in tets:
        cells.extend([4, tet[0], tet[1], tet[2], tet[3]])
        celltypes.append(10)  # VTK_TETRA = 10
    
    grid = pv.UnstructuredGrid(
        np.array(cells),
        np.array(celltypes),
        np.array(points)
    )
    
    if values is not None and name is not None:
        grid.cell_data[name] = np.array(values)
    
    return grid


def _add_well_to_plotter(plotter, well):
    """在plotter中添加钻孔"""
    path = well.get_path_points()
    line = pv.lines_from_points(path)
    plotter.add_mesh(line, color='red', line_width=5)
    
    pts = pv.PolyData(well.stations)
    plotter.add_mesh(pts, color='red', point_size=10,
                    render_points_as_spheres=True)