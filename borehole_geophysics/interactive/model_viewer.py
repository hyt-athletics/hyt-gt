"""
3D模型查看器

把2D截面上画的地质体拉伸成3D，用PyVista显示
"""

import numpy as np
import pyvista as pv


def create_3d_body_mesh(body):
    """
    把一个2D地质体拉伸成3D网格
    
    原理：
      底面 = 多边形在 y = y_min 处
      顶面 = 多边形在 y = y_max 处
      侧面 = 连接底面和顶面的矩形
    
        顶面
       ╱────╲     y = y_max
      ╱ body ╲
     ╱────────╲
     │  body  │    侧面
     │        │
     ╲────────╱
      ╲ body ╱
       ╲────╱     y = y_min
        底面
    """
    verts = body.vertices_xz  # [(x,z), ...]
    y_min, y_max = body.y_range
    n = len(verts)
    
    # 底面和顶面的3D坐标
    bottom_pts = np.array([[x, y_min, z] for x, z in verts])
    top_pts = np.array([[x, y_max, z] for x, z in verts])
    all_points = np.vstack([bottom_pts, top_pts])  # 索引 0~n-1 底面, n~2n-1 顶面
    
    # 构建面
    faces = []
    
    # 底面（一个n边形）
    faces.append(n)  # 顶点数
    faces.extend(range(n - 1, -1, -1))  # 反向（法线向外=向下）
    
    # 顶面（一个n边形）
    faces.append(n)
    faces.extend(range(n, 2 * n))  # 正向（法线向外=向上）
    
    # 侧面（n个四边形）
    for i in range(n):
        j = (i + 1) % n
        faces.extend([4, i, j, j + n, i + n])  # 四边形
    
    mesh = pv.PolyData(all_points, faces=faces)
    return mesh


def show_model_3d(bodies, well, domain):
    """
    3D显示所有地质体 + 钻孔
    
    参数:
        bodies: GeologicalBody列表
        well: VerticalWell对象
        domain: 域信息字典
    """
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # 模型域边框
    x0, x1 = domain['x_range']
    y0, y1 = domain['y_range']
    z0, z1 = domain['z_range']
    bounds = [x0, x1, y0, y1, z0, z1]
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color='gray', line_width=2)
    
    # 地面（半透明）
    ground = pv.Plane(
        center=((x0 + x1) / 2, (y0 + y1) / 2, z0),
        direction=(0, 0, 1),
        i_size=x1 - x0, j_size=y1 - y0,
        i_resolution=1, j_resolution=1,
    )
    plotter.add_mesh(ground, color='tan', opacity=0.15)
    
    # 地质体
    for body in bodies:
        try:
            mesh_3d = create_3d_body_mesh(body)
            plotter.add_mesh(
                mesh_3d,
                color=body.color,
                opacity=0.6,
                show_edges=True,
                edge_color='gray',
                label=f'{body.name} (ρ={body.density:.2f})'
            )
        except Exception as e:
            print(f"  警告：地质体 '{body.name}' 3D显示失败: {e}")
    
    # 钻孔
    if well is not None:
        path = well.get_path_points()
        line = pv.lines_from_points(path)
        plotter.add_mesh(line, color='red', line_width=6, label='Borehole')
        
        stations = pv.PolyData(well.stations)
        plotter.add_mesh(stations, color='red', point_size=8,
                        render_points_as_spheres=True)
    
    plotter.add_legend(bcolor='white', border=True)
    plotter.add_axes()
    
    plotter.add_text(
        f"3D Model: {len(bodies)} bodies",
        position='upper_left', font_size=10
    )
    
    plotter.show()


def show_model_3d_with_mesh(bodies, well, domain, mesh_data=None):
    """
    同时显示地质体和网格（生成网格后使用）
    """
    plotter = pv.Plotter(shape=(1, 2))
    
    # 左：地质体
    plotter.subplot(0, 0)
    plotter.set_background('white')
    plotter.add_text("Geological Bodies", position='upper_left')
    
    x0, x1 = domain['x_range']
    y0, y1 = domain['y_range']
    z0, z1 = domain['z_range']
    
    for body in bodies:
        try:
            mesh_3d = create_3d_body_mesh(body)
            plotter.add_mesh(mesh_3d, color=body.color, opacity=0.6,
                           show_edges=True, edge_color='gray')
        except Exception:
            pass
    
    if well:
        path = well.get_path_points()
        plotter.add_mesh(pv.lines_from_points(path), color='red', line_width=5)
    
    plotter.add_axes()
    
    # 右：网格
    plotter.subplot(0, 1)
    plotter.set_background('white')
    plotter.add_text("Mesh", position='upper_left')
    
    if mesh_data is not None:
        # 假设是八叉树网格的to_arrays()结果
        centers = mesh_data['centers']
        sizes = mesh_data['sizes']
        density = mesh_data['density']
        
        for i in range(len(centers)):
            if abs(density[i]) > 0.001:  # 只画有异常的
                cx, cy, cz = centers[i]
                dx, dy, dz = sizes[i]
                box = pv.Box(bounds=[
                    cx - dx/2, cx + dx/2,
                    cy - dy/2, cy + dy/2,
                    cz - dz/2, cz + dz/2
                ])
                color = '#e74c3c' if density[i] > 0 else '#3498db'
                plotter.add_mesh(box, color=color, opacity=0.5, show_edges=True)
    
    if well:
        path = well.get_path_points()
        plotter.add_mesh(pv.lines_from_points(path), color='red', line_width=5)
    
    plotter.add_axes()
    plotter.show()