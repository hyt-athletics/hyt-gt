"""
用PyVista导出VTK（更简单，但需要PyVista）

如果PyVista已安装，可以用这个更简洁的方式导出
"""

import numpy as np


def export_octree_pyvista(octree_mesh, filename):
    """用PyVista导出八叉树网格"""
    import pyvista as pv
    
    leaves = octree_mesh.root.get_leaves()
    
    blocks = pv.MultiBlock()
    for leaf in leaves:
        box = pv.Box(bounds=[
            leaf.x_min, leaf.x_max,
            leaf.y_min, leaf.y_max,
            leaf.z_min, leaf.z_max
        ])
        box.cell_data['density'] = [leaf.density]
        box.cell_data['resistivity'] = [leaf.resistivity]
        box.cell_data['level'] = [leaf.level]
        blocks.append(box)
    
    merged = blocks.combine()
    merged.save(filename)
    print(f"  ✅ PyVista导出: {filename}")


def export_tet_pyvista(tet_mesh, filename, cell_data=None):
    """用PyVista导出四面体网格"""
    import pyvista as pv
    
    points = tet_mesh.points
    tets = tet_mesh.tetrahedra
    
    cells = []
    celltypes = []
    for tet in tets:
        cells.extend([4, tet[0], tet[1], tet[2], tet[3]])
        celltypes.append(10)
    
    grid = pv.UnstructuredGrid(
        np.array(cells), np.array(celltypes), np.array(points)
    )
    
    if cell_data:
        for name, values in cell_data.items():
            grid.cell_data[name] = values
    
    grid.save(filename)
    print(f"  ✅ PyVista导出: {filename}")


def export_well_pyvista(well, filename):
    """用PyVista导出钻孔"""
    import pyvista as pv
    
    path = well.get_path_points()
    line = pv.lines_from_points(path)
    line.save(filename)
    print(f"  ✅ PyVista导出: {filename}")