"""
高层VTK导出接口

一键把模型的所有内容导出为VTK文件
"""

import numpy as np
import os

from export.vtk_writer import (
    write_vtu, write_vtp, write_vtm, write_csv_with_header,
    VTK_TETRA, VTK_HEXAHEDRON, VTK_WEDGE, VTK_TRIANGLE, VTK_LINE, VTK_VERTEX
)


class VTKExporter:
    """
    VTK导出器
    
    使用方法:
        exporter = VTKExporter('output_folder')
        exporter.export_octree_mesh(mesh)
        exporter.export_borehole(well)
        exporter.export_geological_bodies(bodies)
        exporter.export_gravity_result(result)
        exporter.write_master_file()   # 生成总文件
    
    导出结果:
        output_folder/
        ├── model.vtm              ← 用ParaView打开这个！
        ├── mesh.vtu               ← 网格
        ├── borehole.vtp           ← 钻孔
        ├── borehole_stations.vtp  ← 观测点
        ├── body_矿体1.vtp         ← 地质体表面
        ├── body_矿体2.vtp
        ├── gravity_result.csv     ← 正演结果
        └── gravity_curve.vtp      ← 正演结果（3D曲线）
    """
    
    def __init__(self, output_dir='vtk_output'):
        """
        参数:
            output_dir: 输出文件夹路径
        """
        self.output_dir = output_dir
        self.blocks = []  # vtm文件的块列表
        
        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n  VTK导出文件夹: {os.path.abspath(output_dir)}/")
    
    def _path(self, filename):
        """获取完整路径"""
        return os.path.join(self.output_dir, filename)
    
    # ==========================================================
    #  导出八叉树网格
    # ==========================================================
    
    def export_octree_mesh(self, octree_mesh, name='mesh'):
        """
        导出八叉树网格为VTU文件
        
        每个叶节点(格子)导出为一个六面体单元
        """
        print(f"\n  导出八叉树网格...")
        
        leaves = octree_mesh.root.get_leaves()
        n_cells = len(leaves)
        
        # 收集所有唯一节点和单元
        # 每个六面体有8个顶点
        all_points = []
        cells = []
        cell_types = []
        
        # 物性数据
        density_data = []
        susceptibility_data = []
        resistivity_data = []
        velocity_data = []
        level_data = []
        cell_size_data = []
        
        point_index = 0
        
        for leaf in leaves:
            # 六面体8个顶点
            #    6────7
            #   /|   /|
            #  4────5 |      VTK_HEXAHEDRON顶点顺序
            #  | 2──|-3
            #  |/   |/
            #  0────1
            
            x0, x1 = leaf.x_min, leaf.x_max
            y0, y1 = leaf.y_min, leaf.y_max
            z0, z1 = leaf.z_min, leaf.z_max
            
            corners = [
                [x0, y0, z0],  # 0
                [x1, y0, z0],  # 1
                [x1, y1, z0],  # 2
                [x0, y1, z0],  # 3
                [x0, y0, z1],  # 4
                [x1, y0, z1],  # 5
                [x1, y1, z1],  # 6
                [x0, y1, z1],  # 7
            ]
            
            all_points.extend(corners)
            
            cell = [8]  # 8个顶点
            cell.extend(range(point_index, point_index + 8))
            cells.append(cell)
            cell_types.append(VTK_HEXAHEDRON)
            
            point_index += 8
            
            # 物性
            density_data.append(leaf.density)
            susceptibility_data.append(leaf.susceptibility)
            resistivity_data.append(leaf.resistivity)
            velocity_data.append(leaf.velocity)
            level_data.append(leaf.level)
            cell_size_data.append(leaf.min_edge)
        
        all_points = np.array(all_points)
        
        filename = f'{name}.vtu'
        write_vtu(
            self._path(filename),
            points=all_points,
            cells=cells,
            cell_types=cell_types,
            cell_data={
                'density': np.array(density_data),
                'susceptibility': np.array(susceptibility_data),
                'resistivity': np.array(resistivity_data),
                'velocity': np.array(velocity_data),
                'octree_level': np.array(level_data),
                'cell_size': np.array(cell_size_data),
            }
        )
        
        self.blocks.append((name, filename))
    
    # ==========================================================
    #  导出八叉树（从to_arrays格式）
    # ==========================================================
    
    def export_mesh_from_arrays(self, mesh_data, name='mesh'):
        """
        从网格数据字典导出（centers + sizes格式）
        
        参数:
            mesh_data: {'centers': (N,3), 'sizes': (N,3), 'density': (N,), ...}
        """
        print(f"\n  导出网格数据...")
        
        centers = mesh_data['centers']
        sizes = mesh_data['sizes']
        n_cells = len(centers)
        
        all_points = []
        cells = []
        cell_types = []
        point_index = 0
        
        for i in range(n_cells):
            cx, cy, cz = centers[i]
            dx, dy, dz = sizes[i]
            
            x0, x1 = cx - dx/2, cx + dx/2
            y0, y1 = cy - dy/2, cy + dy/2
            z0, z1 = cz - dz/2, cz + dz/2
            
            corners = [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ]
            all_points.extend(corners)
            
            cell = [8] + list(range(point_index, point_index + 8))
            cells.append(cell)
            cell_types.append(VTK_HEXAHEDRON)
            point_index += 8
        
        # 收集物性数据
        cell_data = {}
        for key in ['density', 'susceptibility', 'resistivity', 'velocity']:
            if key in mesh_data:
                cell_data[key] = mesh_data[key]
        if 'volumes' in mesh_data:
            cell_data['volume'] = mesh_data['volumes']
        
        filename = f'{name}.vtu'
        write_vtu(
            self._path(filename),
            points=np.array(all_points),
            cells=cells,
            cell_types=cell_types,
            cell_data=cell_data
        )
        
        self.blocks.append((name, filename))
    
    # ==========================================================
    #  导出2.5D三棱柱网格
    # ==========================================================
    
    def export_prism_mesh(self, prism_mesh, name='prism_mesh'):
        """
        导出2.5D三棱柱网格
        """
        print(f"\n  导出2.5D三棱柱网格...")
        
        nodes = prism_mesh.nodes
        prisms = prism_mesh.prisms
        n_cells = len(prisms)
        
        cells = []
        cell_types = []
        for prism in prisms:
            cells.append([6] + list(prism))  # 三棱柱有6个节点
            cell_types.append(VTK_WEDGE)
        
        cell_data = {
            'density': prism_mesh.density,
            'susceptibility': prism_mesh.susceptibility,
            'resistivity': prism_mesh.resistivity,
            'velocity': prism_mesh.velocity,
        }
        if prism_mesh.prism_volumes is not None:
            cell_data['volume'] = prism_mesh.prism_volumes
        
        filename = f'{name}.vtu'
        write_vtu(
            self._path(filename),
            points=nodes,
            cells=cells,
            cell_types=cell_types,
            cell_data=cell_data
        )
        
        self.blocks.append((name, filename))
    
    # ==========================================================
    #  导出3D四面体网格
    # ==========================================================
    
    def export_tet_mesh(self, tet_mesh, cell_properties=None, name='tet_mesh'):
        """
        导出3D四面体网格
        
        参数:
            tet_mesh: DelaunayTriangulation3D 对象
            cell_properties: 字典 {'density': array, ...}
        """
        print(f"\n  导出3D四面体网格...")
        
        points = tet_mesh.points
        tets = tet_mesh.tetrahedra
        
        cells = []
        cell_types = []
        for tet in tets:
            cells.append([4, tet[0], tet[1], tet[2], tet[3]])
            cell_types.append(VTK_TETRA)
        
        cell_data = cell_properties if cell_properties else {}
        
        # 自动计算体积
        volumes = tet_mesh.get_all_volumes()
        cell_data['volume'] = volumes
        
        filename = f'{name}.vtu'
        write_vtu(
            self._path(filename),
            points=points,
            cells=cells,
            cell_types=cell_types,
            cell_data=cell_data
        )
        
        self.blocks.append((name, filename))
    
    # ==========================================================
    #  导出2D三角网格
    # ==========================================================
    
    def export_triangulation_2d(self, triangulation, cell_properties=None,
                                 name='triangulation_2d'):
        """
        导出2D三角网格（XZ平面，Y=0）
        """
        print(f"\n  导出2D三角网格...")
        
        points_2d = triangulation.points
        # 添加Y=0，变成3D
        points_3d = np.column_stack([
            points_2d[:, 0],
            np.zeros(len(points_2d)),
            points_2d[:, 1]
        ])
        
        cells = []
        cell_types = []
        for tri in triangulation.triangles:
            cells.append([3, tri[0], tri[1], tri[2]])
            cell_types.append(VTK_TRIANGLE)
        
        filename = f'{name}.vtu'
        write_vtu(
            self._path(filename),
            points=points_3d,
            cells=cells,
            cell_types=cell_types,
            cell_data=cell_properties
        )
        
        self.blocks.append((name, filename))
    
    # ==========================================================
    #  导出钻孔
    # ==========================================================
    
    def export_borehole(self, well, name='borehole'):
        """
        导出钻孔路径和观测点
        
        钻孔路径 → 线段(.vtp)
        观测点 → 点(.vtp)
        """
        print(f"\n  导出钻孔...")
        
        # 钻孔路径（一条折线）
        path_points = well.get_path_points(n_points=100)
        n = len(path_points)
        
        # 线段连接关系
        lines = []
        for i in range(n - 1):
            lines.append((i, i + 1))
        
        filename_path = f'{name}_path.vtp'
        write_vtp(
            self._path(filename_path),
            points=path_points,
            polygons=lines,
        )
        self.blocks.append((f'{name}_path', filename_path))
        
        # 观测点（每个点是一个顶点）
        stations = well.stations
        n_stations = len(stations)
        
        # 导出为VTU（点数据）
        filename_stations = f'{name}_stations.vtu'
        
        cells = []
        cell_types = []
        for i in range(n_stations):
            cells.append([1, i])
            cell_types.append(VTK_VERTEX)
        
        point_data = {
            'depth': stations[:, 2],
            'station_id': np.arange(n_stations, dtype=float),
        }
        
        write_vtu(
            self._path(filename_stations),
            points=stations,
            cells=cells,
            cell_types=cell_types,
            point_data=point_data,
        )
        self.blocks.append((f'{name}_stations', filename_stations))
    
    # ==========================================================
    #  导出地质体表面
    # ==========================================================
    
    def export_geological_bodies(self, bodies, name_prefix='body'):
        """
        导出地质体的3D表面
        
        把2D多边形截面拉伸成3D，导出外表面
        """
        print(f"\n  导出地质体表面...")
        
        for i, body in enumerate(bodies):
            verts = body.vertices_xz
            y_min, y_max = body.y_range
            n = len(verts)
            
            # 底面和顶面的3D坐标
            bottom = [[x, y_min, z] for x, z in verts]
            top = [[x, y_max, z] for x, z in verts]
            all_points = np.array(bottom + top)
            
            polygons = []
            
            # 底面
            polygons.append(list(range(n - 1, -1, -1)))
            
            # 顶面
            polygons.append(list(range(n, 2 * n)))
            
            # 侧面
            for j in range(n):
                k = (j + 1) % n
                polygons.append([j, k, k + n, j + n])
            
            # 面数据
            n_polys = len(polygons)
            cell_data = {
                'density': np.full(n_polys, body.density),
            }
            
            safe_name = body.name.replace(' ', '_').replace('/', '_')
            filename = f'{name_prefix}_{safe_name}.vtp'
            
            write_vtp(
                self._path(filename),
                points=all_points,
                polygons=polygons,
                cell_data=cell_data,
            )
            self.blocks.append((f'{body.name}', filename))
    
    # ==========================================================
    #  导出正演结果
    # ==========================================================
    
    def export_gravity_result(self, result, name='gravity'):
        """
        导出正演结果
        
        1. CSV文件（方便用Excel打开）
        2. VTP 3D曲线（在ParaView中显示）
        """
        print(f"\n  导出正演结果...")
        
        # CSV
        filename_csv = f'{name}_result.csv'
        write_csv_with_header(
            self._path(filename_csv),
            columns={
                'depth_m': result.depths,
                'gz_mGal': result.gz,
                'x_m': np.full_like(result.depths, result.well.x),
                'y_m': np.full_like(result.depths, result.well.y),
            },
            header_lines=[
                f'Borehole gravity forward modeling result',
                f'Well position: ({result.well.x}, {result.well.y})',
                f'Depth range: {result.depths.min():.0f} ~ {result.depths.max():.0f} m',
                f'gz range: {result.gz.min():.4f} ~ {result.gz.max():.4f} mGal',
            ]
        )
        
        # 3D曲线：把重力值映射成水平偏移
        # 这样在ParaView里能直接看到曲线
        gz_normalized = result.gz.copy()
        if np.ptp(gz_normalized) > 0:
            scale = 100.0 / max(abs(gz_normalized).max(), 1e-10)  # 最大偏移100m
        else:
            scale = 1.0
        
        curve_points = np.column_stack([
            result.well.x + gz_normalized * scale,  # X方向偏移
            np.full_like(result.depths, result.well.y),
            result.depths
        ])
        
        n = len(curve_points)
        lines = [(i, i + 1) for i in range(n - 1)]
        
        filename_curve = f'{name}_curve.vtp'
        write_vtp(
            self._path(filename_curve),
            points=curve_points,
            polygons=lines,
            point_data={
                'gz_mGal': result.gz,
                'depth_m': result.depths,
            }
        )
        self.blocks.append(('gravity_curve', filename_curve))
        
        # 也把原始钻孔位置的点导出，带gz值
        filename_points = f'{name}_at_stations.vtu'
        cells = [[1, i] for i in range(len(result.depths))]
        cell_types = [VTK_VERTEX] * len(result.depths)
        
        write_vtu(
            self._path(filename_points),
            points=result.well.stations,
            cells=cells,
            cell_types=cell_types,
            point_data={
                'gz_mGal': result.gz,
                'depth_m': result.depths,
            }
        )
        self.blocks.append(('gravity_at_stations', filename_points))
    
    # ==========================================================
    #  生成总文件
    # ==========================================================
    
    def write_master_file(self, name='model'):
        """
        生成 .vtm 主文件
        
        在ParaView中打开这一个文件就能看到所有内容
        """
        print(f"\n  生成主文件...")
        
        filename = f'{name}.vtm'
        write_vtm(self._path(filename), self.blocks)
        
        full_path = os.path.abspath(self._path(filename))
        
        print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║                VTK 导出完成！                         ║
    ╠══════════════════════════════════════════════════════╣
    ║                                                      ║
    ║  输出文件夹: {self.output_dir}/
    ║  主文件: {filename}
    ║  包含 {len(self.blocks)} 个数据块:
    ║""")
        for bname, bfile in self.blocks:
            print(f"    ║    • {bname} → {bfile}")
        print(f"""    ║
    ║  使用方法:
    ║    1. 打开 ParaView
    ║    2. File → Open → 选择 {filename}
    ║    3. 点击 Apply
    ║    4. 在左侧 Pipeline Browser 中
    ║       勾选/取消各个块的可见性
    ║                                                      ║
    ║  文件完整路径:
    ║    {full_path}
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
        """)
    
    # ==========================================================
    #  一键导出全部
    # ==========================================================
    
    def export_all(self, octree_mesh=None, prism_mesh=None, tet_mesh=None,
                   well=None, bodies=None, gravity_result=None,
                   mesh_data=None, tet_properties=None):
        """
        一键导出所有可用的数据
        
        传什么就导出什么，没有的跳过
        """
        print("\n" + "=" * 55)
        print("  一键导出所有数据到VTK")
        print("=" * 55)
        
        if octree_mesh is not None:
            self.export_octree_mesh(octree_mesh)
        
        if mesh_data is not None:
            self.export_mesh_from_arrays(mesh_data)
        
        if prism_mesh is not None:
            self.export_prism_mesh(prism_mesh)
        
        if tet_mesh is not None:
            self.export_tet_mesh(tet_mesh, cell_properties=tet_properties)
        
        if well is not None:
            self.export_borehole(well)
        
        if bodies is not None and len(bodies) > 0:
            self.export_geological_bodies(bodies)
        
        if gravity_result is not None:
            self.export_gravity_result(gravity_result)
        
        self.write_master_file()