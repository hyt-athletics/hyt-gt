"""
2D截面可视化

用 matplotlib 画2D三角网格
比3D简单，适合调试
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# 设置中文字体，解决中文显示为方框的问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def show_triangulation(triangulation, title="2D Delaunay三角剖分",
                       show_points=True, show_circumcircles=False,
                       highlight_boundary=None, property_values=None,
                       figsize=(12, 10)):
    """
    显示2D三角网格
    
    参数:
        triangulation: DelaunayTriangulation2D对象
        title: 图标题
        show_points: 是否显示点
        show_circumcircles: 是否显示外接圆（仅前几个，用于教学）
        highlight_boundary: 多边形顶点列表（高亮显示异常体边界）
        property_values: 每个三角形的物性值（用颜色显示）
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    points = triangulation.points
    triangles = np.array(triangulation.triangles)
    
    if property_values is not None:
        # 用颜色显示物性
        triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        tpc = ax.tripcolor(triang, property_values, cmap='RdYlBu_r', 
                           edgecolors='gray', linewidth=0.5)
        plt.colorbar(tpc, ax=ax, label='Property value')
    else:
        # 只画线框
        triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        ax.triplot(triang, 'b-', linewidth=0.5)
    
    if show_points:
        ax.plot(points[:, 0], points[:, 1], 'k.', markersize=3)
    
    if highlight_boundary is not None:
        poly = MplPolygon(highlight_boundary, fill=False, 
                         edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(poly)
    
    if show_circumcircles:
        # 显示前5个三角形的外接圆（教学用）
        for i in range(min(5, len(triangles))):
            _draw_circumcircle(ax, points, triangles[i])
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m, depth)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.invert_yaxis()   # Z轴向下（深度）
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def show_quality_map(triangulation, quality_data, figsize=(14, 5)):
    """
    显示网格质量分布
    
    画三张图：质量值、最小角度、面积
    """
    points = triangulation.points
    triangles = np.array(triangulation.triangles)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 图1：质量值
    tpc1 = axes[0].tripcolor(triang, quality_data['qualities'], 
                              cmap='RdYlGn', vmin=0, vmax=1,
                              edgecolors='gray', linewidth=0.3)
    plt.colorbar(tpc1, ax=axes[0])
    axes[0].set_title('质量 (0=差, 1=好)')
    
    # 图2：最小角度
    tpc2 = axes[1].tripcolor(triang, quality_data['min_angles'],
                              cmap='RdYlGn', vmin=0, vmax=60,
                              edgecolors='gray', linewidth=0.3)
    plt.colorbar(tpc2, ax=axes[1])
    axes[1].set_title('最小角度 (°)')
    
    # 图3：面积
    tpc3 = axes[2].tripcolor(triang, np.log10(quality_data['areas'] + 1),
                              cmap='viridis',
                              edgecolors='gray', linewidth=0.3)
    plt.colorbar(tpc3, ax=axes[2])
    axes[2].set_title('面积 (log10, m²)')
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
    
    plt.suptitle('网格质量评估', fontsize=14)
    plt.tight_layout()
    plt.show()


def show_mesh_comparison(mesh_coarse, mesh_fine, titles=None, figsize=(14, 6)):
    """
    对比两个网格（如加密前后）
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if titles is None:
        titles = ['加密前', '加密后']
    
    for i, (mesh, title) in enumerate(zip([mesh_coarse, mesh_fine], titles)):
        points = mesh.points
        triangles = np.array(mesh.triangles)
        triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        
        axes[i].triplot(triang, 'b-', linewidth=0.5)
        axes[i].plot(points[:, 0], points[:, 1], 'k.', markersize=2)
        axes[i].set_title(f'{title}\n{len(triangles)}个三角形, {len(points)}个点')
        axes[i].set_aspect('equal')
        axes[i].invert_yaxis()
        axes[i].set_xlabel('X (m)')
        axes[i].set_ylabel('Z (m)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _draw_circumcircle(ax, points, triangle):
    """画一个三角形的外接圆（教学用）"""
    ax_pt, ay_pt = points[triangle[0]]
    bx, by = points[triangle[1]]
    cx, cy = points[triangle[2]]
    
    d = 2 * (ax_pt * (by - cy) + bx * (cy - ay_pt) + cx * (ay_pt - by))
    if abs(d) < 1e-10:
        return
    
    ux = ((ax_pt**2 + ay_pt**2) * (by - cy) + (bx**2 + by**2) * (cy - ay_pt) + 
          (cx**2 + cy**2) * (ay_pt - by)) / d
    uy = ((ax_pt**2 + ay_pt**2) * (cx - bx) + (bx**2 + by**2) * (ax_pt - cx) + 
          (cx**2 + cy**2) * (bx - ax_pt)) / d
    
    r = np.sqrt((ax_pt - ux)**2 + (ay_pt - uy)**2)
    
    circle = plt.Circle((ux, uy), r, fill=False, color='green', 
                        linewidth=0.5, linestyle='--', alpha=0.5)
    ax.add_patch(circle)


def show_25d_mesh(prism_mesh, property_name='density', 
                   well=None, threshold=None, opacity=0.7):
    """
    用PyVista可视化2.5D三棱柱网格
    """
    import pyvista as pv
    
    nodes = prism_mesh.nodes
    prisms = prism_mesh.prisms
    prop_data = getattr(prism_mesh, property_name)
    
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # PyVista需要特定格式的单元数组
    # 三棱柱 = VTK_WEDGE, type_id = 13
    n_prisms = len(prisms)
    cells = []
    celltypes = []
    
    for prism in prisms:
        cells.append(6)  # 6个节点
        cells.extend(prism)
        celltypes.append(13)  # VTK_WEDGE
    
    cells = np.array(cells)
    celltypes = np.array(celltypes)
    
    grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    grid.cell_data[property_name] = prop_data
    
    if threshold is not None:
        display = grid.threshold(threshold, scalars=property_name)
    else:
        display = grid
    
    plotter.add_mesh(display, scalars=property_name,
                    cmap='RdYlBu_r', show_edges=True,
                    opacity=opacity,
                    scalar_bar_args={'title': property_name})
    
    # 画模型域边框
    plotter.add_mesh(grid.outline(), color='gray', line_width=2)
    
    # 画钻孔
    if well is not None:
        path = well.get_path_points()
        line = pv.lines_from_points(path)
        plotter.add_mesh(line, color='red', line_width=5)
        
        pts = pv.PolyData(well.stations)
        plotter.add_mesh(pts, color='red', point_size=10,
                        render_points_as_spheres=True)
    
    plotter.add_axes()
    plotter.show()


def show_constrained_triangulation(cdt, title="约束Delaunay三角剖分",
                                    highlight_constraints=True,
                                    show_labels=False, figsize=(12, 10)):
    
    """
    显示约束Delaunay三角网格
    
    特别标注约束边（用粗红线）
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    points = cdt.points
    triangles = np.array(cdt.triangles)
    
    # 画三角形
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    ax.triplot(triang, 'b-', linewidth=0.5, alpha=0.6)
    
    # 高亮约束边
    if highlight_constraints:
        for (i, j) in cdt.constraints:
            pi = points[i]
            pj = points[j]
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], 
                   'r-', linewidth=2.5, label='约束边' if (i,j) == cdt.constraints[0] else "")
    
    # 画点
    ax.plot(points[:, 0], points[:, 1], 'k.', markersize=4)
    
    # 显示索引标签
    if show_labels:
        for idx, (x, z) in enumerate(points):
            ax.text(x, z, str(idx), fontsize=7, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m, depth)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    if highlight_constraints:
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def compare_delaunay_vs_constrained(points, constraints, figsize=(16, 6)):
    """
    对比普通Delaunay和约束Delaunay
    """
    from mesh.delaunay2d import DelaunayTriangulation2D
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：普通Delaunay
    dt = DelaunayTriangulation2D()
    dt.triangulate(points)
    
    triangles_dt = np.array(dt.triangles)
    triang_dt = mtri.Triangulation(dt.points[:, 0], dt.points[:, 1], triangles_dt)
    axes[0].triplot(triang_dt, 'b-', linewidth=0.5)
    axes[0].plot(dt.points[:, 0], dt.points[:, 1], 'k.', markersize=4)
    
    # 画约束边（可能会被穿过）
    for (i, j) in constraints:
        pi = dt.points[i]
        pj = dt.points[j]
        axes[0].plot([pi[0], pj[0]], [pi[1], pj[1]], 'r--', linewidth=2, alpha=0.7)
    
    axes[0].set_title('普通Delaunay\n（约束边可能被穿过）')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：约束Delaunay
    from mesh.constrained_delaunay import ConstrainedDelaunayTriangulation
    cdt = ConstrainedDelaunayTriangulation()
    cdt.triangulate(points, constraints)
    
    triangles_cdt = np.array(cdt.triangles)
    triang_cdt = mtri.Triangulation(cdt.points[:, 0], cdt.points[:, 1], triangles_cdt)
    axes[1].triplot(triang_cdt, 'b-', linewidth=0.5, alpha=0.6)
    axes[1].plot(cdt.points[:, 0], cdt.points[:, 1], 'k.', markersize=4)
    
    # 画约束边（一定存在）
    for (i, j) in cdt.constraints:
        pi = cdt.points[i]
        pj = cdt.points[j]
        axes[1].plot([pi[0], pj[0]], [pi[1], pj[1]], 'r-', linewidth=2.5)
    
    axes[1].set_title('约束Delaunay\n（约束边强制存在）')
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
    
    plt.suptitle('普通Delaunay vs 约束Delaunay 对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
