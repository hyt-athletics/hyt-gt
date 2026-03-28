"""
将交互式建模的结果连接到网格剖分模块

流程:
  1. 从ModelManager获取地质体
  2. 创建八叉树网格（或2.5D网格）
  3. 沿钻孔加密
  4. 沿地质体边界加密
  5. 赋物性
  6. 可视化
"""

import numpy as np


def generate_mesh_from_model(model_manager, well, mesh_type='octree',
                              max_level=5, view_result=True):
    """
    从交互式模型生成网格
    
    参数:
        model_manager: ModelManager 对象
        well: VerticalWell 对象
        mesh_type: 'octree' 或 '25d'
        max_level: 八叉树最大层级
        view_result: 是否自动显示结果
    """
    print("\n" + "=" * 50)
    print("  从交互模型生成网格")
    print("=" * 50)
    
    if mesh_type == 'octree':
        mesh_data = _generate_octree_mesh(model_manager, well, max_level)
    else:
        mesh_data = _generate_25d_mesh(model_manager, well)
    
    if view_result and mesh_data is not None:
        from interactive.model_viewer import show_model_3d_with_mesh
        show_model_3d_with_mesh(
            model_manager.bodies, well,
            model_manager.domain, mesh_data
        )
    
    return mesh_data


def _generate_octree_mesh(model_manager, well, max_level):
    """用八叉树生成网格"""
    from mesh.octree import OctreeMesh
    
    domain = model_manager.domain
    
    # 创建网格
    mesh = OctreeMesh(
        x_range=domain['x_range'],
        y_range=domain['y_range'],
        z_range=domain['z_range'],
        max_level=max_level
    )
    
    print(f"  初始: {mesh.n_cells} 个单元")
    
    # 沿钻孔加密
    print("  沿钻孔加密...")
    mesh.refine_along_borehole(well, levels_and_radii=[
        (max_level, 30),
        (max_level - 1, 80),
        (max_level - 2, 200),
    ])
    print(f"  钻孔加密后: {mesh.n_cells} 个单元")
    
    # 沿地质体边界加密
    for body in model_manager.bodies:
        print(f"  加密 '{body.name}' 边界...")
        mesh.refine_at_boundary(body.contains_point_3d, target_level=max_level - 1)
    print(f"  边界加密后: {mesh.n_cells} 个单元")
    
    # 平衡
    mesh.balance()
    print(f"  平衡后: {mesh.n_cells} 个单元")
    
    # 赋物性
    for body in model_manager.bodies:
        mesh.assign_property(body.contains_point_3d, 'density', body.density)
    
    mesh.get_info()
    
    return mesh.to_arrays()


def _generate_25d_mesh(model_manager, well):
    """用2.5D三棱柱生成网格"""
    from mesh.adaptive2d import AdaptiveTriangulation2D
    from mesh.prism25d import PrismMesh25D
    
    domain = model_manager.domain
    x_range = domain['x_range']
    z_range = domain['z_range']
    
    # 2D自适应三角化
    print("  生成2D自适应三角网格...")
    adaptive = AdaptiveTriangulation2D()
    adaptive.set_rectangular_domain(x_range=x_range, z_range=z_range, n_boundary=12)
    adaptive.generate_background_points(x_range=x_range, z_range=z_range, spacing=100)
    
    # 钻孔加密
    adaptive.refine_near_line_graded(
        start=(well.x, well.z_top),
        end=(well.x, well.z_bottom),
        min_spacing=20, max_spacing=80,
        influence_radius=200, n_layers=4
    )
    adaptive.refine_along_line(
        start=(well.x, well.z_top),
        end=(well.x, well.z_bottom),
        spacing=20
    )
    
    # 地质体边界加密
    for body in model_manager.bodies:
        adaptive.refine_at_polygon_boundary(body.vertices_xz, spacing=15)
    
    triangulation = adaptive.build()
    
    # Y方向分层
    y_min, y_max = domain['y_range']
    y_center = (well.y if well else (y_min + y_max) / 2)
    
    y_layers = np.sort(np.unique(np.concatenate([
        np.arange(y_center - 100, y_center + 101, 20),
        np.arange(y_center - 300, y_center - 100, 50),
        np.arange(y_center + 100, y_center + 301, 50),
        np.array([y_min, y_min + 100, y_max - 100, y_max]),
    ])))
    y_layers = y_layers[(y_layers >= y_min) & (y_layers <= y_max)]
    
    # 生成2.5D网格
    print("  生成2.5D三棱柱网格...")
    prism_mesh = PrismMesh25D()
    prism_mesh.set_triangulation(triangulation)
    prism_mesh.set_y_layers(y_layers)
    prism_mesh.build()
    
    # 赋物性
    for body in model_manager.bodies:
        prism_mesh.assign_property(body.contains_point_3d, 'density', body.density)
    
    prism_mesh.get_info()
    
    return prism_mesh.to_arrays()