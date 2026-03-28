"""
3D可视化模块

用 PyVista 把网格和钻孔画出来
"""

import numpy as np
import pyvista as pv


def show_structured_mesh(mesh, property_name='density', 
                          show_edges=True, threshold=None,
                          well=None, opacity=0.7):
    """
    可视化结构化网格
    
    参数:
        mesh: StructuredMesh 对象
        property_name: 要显示的物性，如 'density'
        show_edges: 是否显示网格线
        threshold: 只显示物性值大于此值的格子（去掉背景，只看异常体）
        well: VerticalWell 对象（如果有就画出来）
        opacity: 透明度 0~1
    """
    # 获取物性数据
    prop_data = getattr(mesh, property_name)
    
    # 创建 PyVista 的 RectilinearGrid（结构化网格专用）
    grid = pv.RectilinearGrid(mesh.x_edges, mesh.y_edges, mesh.z_edges)
    
    # 把物性数据附加到网格上
    # 注意：PyVista 的 cell 数据排列顺序和我们的可能不同，需要 reshape
    values = prop_data.reshape((mesh.nx, mesh.ny, mesh.nz), order='C')
    grid.cell_data[property_name] = values.ravel(order='F')  # PyVista用Fortran顺序
    
    # 创建画布
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    if threshold is not None:
        # 只显示超过阈值的格子（比如只看有异常的区域）
        threshed = grid.threshold(threshold, scalars=property_name)
        plotter.add_mesh(threshed, scalars=property_name, 
                        cmap='RdYlBu_r', show_edges=show_edges,
                        opacity=opacity, scalar_bar_args={'title': property_name})
        
        # 同时画一个半透明的边框表示整个模型域
        outline = grid.outline()
        plotter.add_mesh(outline, color='gray', line_width=2)
    else:
        # 显示所有格子
        plotter.add_mesh(grid, scalars=property_name,
                        cmap='RdYlBu_r', show_edges=show_edges,
                        opacity=opacity, scalar_bar_args={'title': property_name})
    
    # 画钻孔
    if well is not None:
        # 画钻孔路径（一条线）
        path = well.get_path_points()
        line = pv.lines_from_points(path)
        plotter.add_mesh(line, color='red', line_width=5, label='Borehole')
        
        # 画观测点（小球）
        points = pv.PolyData(well.stations)
        plotter.add_mesh(points, color='red', point_size=10, 
                        render_points_as_spheres=True, label='Stations')
    
    # 设置坐标轴
    plotter.add_axes()
    plotter.add_text(f"Property: {property_name}", position='upper_left', font_size=12)
    
    # Z轴翻转（深度向下为正，但显示时向下）
    plotter.camera_position = 'iso'
    
    plotter.show()


def show_octree_mesh(octree_mesh, property_name='density',
                      well=None, show_edges=True, threshold=None):
    """
    可视化八叉树网格
    
    把每个叶节点(最终格子)画成一个小方块
    """
    # 获取所有叶节点
    leaves = octree_mesh.root.get_leaves()
    
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # 收集数据
    prop_values = np.array([getattr(leaf, property_name) for leaf in leaves])
    
    # 用 MultiBlock 把所有小方块合并
    blocks = pv.MultiBlock()
    for i, leaf in enumerate(leaves):
        box = pv.Box(bounds=[
            leaf.x_min, leaf.x_max,
            leaf.y_min, leaf.y_max,
            leaf.z_min, leaf.z_max
        ])
        # 给所有单元格赋相同的值
        n_cells = box.n_cells
        box.cell_data[property_name] = [prop_values[i]] * n_cells
        blocks.append(box)
    
    merged = blocks.combine()
    
    if threshold is not None:
        merged = merged.threshold(threshold, scalars=property_name)
    
    plotter.add_mesh(merged, scalars=property_name,
                    cmap='RdYlBu_r', show_edges=show_edges,
                    scalar_bar_args={'title': property_name})
    
    # 画模型域边框
    domain = pv.Box(bounds=[
        octree_mesh.root.x_min, octree_mesh.root.x_max,
        octree_mesh.root.y_min, octree_mesh.root.y_max,
        octree_mesh.root.z_min, octree_mesh.root.z_max
    ])
    plotter.add_mesh(domain.outline(), color='gray', line_width=2)
    
    # 画钻孔
    if well is not None:
        path = well.get_path_points()
        line = pv.lines_from_points(path)
        plotter.add_mesh(line, color='red', line_width=5)
        
        points = pv.PolyData(well.stations)
        plotter.add_mesh(points, color='red', point_size=10,
                        render_points_as_spheres=True)
    
    plotter.add_axes()
    plotter.show()


def show_octree_wireframe(octree_mesh, well=None, color_by_level=True):
    """
    只画网格线框（不填充颜色）
    
    适合查看网格的疏密分布
    """
    leaves = octree_mesh.root.get_leaves()
    
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    if color_by_level:
        # 按细分层级着色：越细的格子颜色越深
        max_level = max(leaf.level for leaf in leaves)
        colors = ['#e0e0e0', '#b0b0b0', '#808080', '#505050', 
                  '#303030', '#101010', '#000000']
    
    for leaf in leaves:
        box = pv.Box(bounds=[
            leaf.x_min, leaf.x_max,
            leaf.y_min, leaf.y_max,
            leaf.z_min, leaf.z_max
        ])
        if color_by_level:
            c = colors[min(leaf.level, len(colors) - 1)]
        else:
            c = 'gray'
        plotter.add_mesh(box, style='wireframe', color=c, line_width=1)
    
    if well is not None:
        path = well.get_path_points()
        line = pv.lines_from_points(path)
        plotter.add_mesh(line, color='red', line_width=5)
    
    plotter.add_axes()
    plotter.add_text(f"Total cells: {len(leaves)}", position='upper_left')
    plotter.show()