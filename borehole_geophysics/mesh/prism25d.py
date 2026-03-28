"""
2.5D三棱柱网格

原理：
    1. 在XZ截面上做2D三角化（用上面写的Delaunay）
    2. 沿Y方向拉伸，把三角形变成三棱柱
    3. Y方向可以不等间距（近处密、远处疏）
"""

import numpy as np


class PrismMesh25D:
    """
    2.5D三棱柱网格
    
    使用流程:
        mesh = PrismMesh25D()
        mesh.set_triangulation(triangulation)   # 设置2D三角化
        mesh.set_y_layers(y_values)              # 设置Y方向分层
        mesh.build()                             # 生成3D网格
        mesh.assign_property(...)                # 赋物性
    """
    
    def __init__(self):
        self.triangulation = None   # 2D三角化结果
        self.y_layers = None        # Y方向的层位置
        
        # 3D结果
        self.nodes = None           # 3D节点坐标 (N_nodes, 3)
        self.prisms = None          # 三棱柱连接关系 (N_prisms, 6)
        self.n_prisms = 0
        
        # 物性
        self.density = None
        self.susceptibility = None
        self.resistivity = None
        self.velocity = None
        
        # 辅助信息
        self.prism_centers = None
        self.prism_volumes = None
    
    def set_triangulation(self, triangulation):
        """
        设置2D三角剖分结果
        
        参数:
            triangulation: DelaunayTriangulation2D 对象
        """
        self.triangulation = triangulation
        print(f"  2D截面: {len(triangulation.points)} 个点, "
              f"{len(triangulation.triangles)} 个三角形")
        return self
    
    def set_y_layers(self, y_values):
        """
        设置Y方向的分层
        
        参数:
            y_values: Y坐标列表，如 [0, 5, 10, 20, 50, 100, 200, 500, 1000]
                      可以不等间距（近处密、远处疏）
        """
        self.y_layers = np.array(sorted(y_values), dtype=float)
        
        # 打印Y方向信息
        spacings = np.diff(self.y_layers)
        print(f"  Y方向: {len(self.y_layers)} 层, "
              f"{len(self.y_layers)-1} 段")
        print(f"    范围: {self.y_layers[0]:.0f} ~ {self.y_layers[-1]:.0f} m")
        print(f"    最小间距: {spacings.min():.1f} m")
        print(f"    最大间距: {spacings.max():.1f} m")
        return self
    
    def build(self):
        """
        生成3D三棱柱网格
        
        过程:
            1. 把2D节点在每个Y层复制一份 → 3D节点
            2. 把2D三角形在每个Y段拉伸成三棱柱 → 3D单元
        """
        if self.triangulation is None:
            raise ValueError("请先调用 set_triangulation()")
        if self.y_layers is None:
            raise ValueError("请先调用 set_y_layers()")
        
        points_2d = self.triangulation.points     # (n_pts_2d, 2) → [x, z]
        tris_2d = self.triangulation.triangles     # [(i, j, k), ...]
        
        n_pts_2d = len(points_2d)
        n_tris = len(tris_2d)
        n_y_layers = len(self.y_layers)
        n_y_segments = n_y_layers - 1
        
        # ==========================================
        # 生成3D节点
        # 每个2D点 × 每个Y层 = 一个3D节点
        # ==========================================
        nodes = []
        for iy in range(n_y_layers):
            y_val = self.y_layers[iy]
            for i2d in range(n_pts_2d):
                x, z = points_2d[i2d]
                nodes.append([x, y_val, z])
        
        self.nodes = np.array(nodes)
        
        # ==========================================
        # 生成三棱柱单元
        # 每个2D三角形 × 每个Y段 = 一个三棱柱
        # ==========================================
        prisms = []
        for iy in range(n_y_segments):
            # 底面节点偏移量（第iy层的节点起始索引）
            offset_bottom = iy * n_pts_2d
            # 顶面节点偏移量（第iy+1层的节点起始索引）
            offset_top = (iy + 1) * n_pts_2d
            
            for tri in tris_2d:
                # 三棱柱的6个节点: 底面3个 + 顶面3个
                prism = [
                    tri[0] + offset_bottom,
                    tri[1] + offset_bottom,
                    tri[2] + offset_bottom,
                    tri[0] + offset_top,
                    tri[1] + offset_top,
                    tri[2] + offset_top
                ]
                prisms.append(prism)
        
        self.prisms = np.array(prisms)
        self.n_prisms = len(self.prisms)
        
        # ==========================================
        # 初始化物性
        # ==========================================
        self.density = np.zeros(self.n_prisms)
        self.susceptibility = np.zeros(self.n_prisms)
        self.resistivity = np.ones(self.n_prisms) * 100.0
        self.velocity = np.ones(self.n_prisms) * 3000.0
        
        # ==========================================
        # 计算三棱柱中心和体积
        # ==========================================
        self._compute_prism_info()
        
        print(f"\n  ✅ 2.5D网格生成完成:")
        print(f"     3D节点: {len(self.nodes)}")
        print(f"     三棱柱: {self.n_prisms}")
        print(f"     = {n_tris} 个三角形 × {n_y_segments} 个Y段")
        
        return self
    
    def _compute_prism_info(self):
        """计算每个三棱柱的中心坐标和体积"""
        centers = []
        volumes = []
        
        for i in range(self.n_prisms):
            node_coords = self.nodes[self.prisms[i]]  # (6, 3)
            
            # 中心 = 6个顶点的平均
            center = node_coords.mean(axis=0)
            centers.append(center)
            
            # 体积 = 底面三角形面积 × 高(Y方向厚度)
            # 底面三角形的3个点
            p0, p1, p2 = node_coords[0], node_coords[1], node_coords[2]
            
            # 三角形面积（叉积，XZ平面）
            v01 = p1 - p0
            v02 = p2 - p0
            tri_area = 0.5 * abs(v01[0] * v02[2] - v02[0] * v01[2])
            
            # Y方向高度
            height = abs(node_coords[3, 1] - node_coords[0, 1])
            
            volumes.append(tri_area * height)
        
        self.prism_centers = np.array(centers)
        self.prism_volumes = np.array(volumes)
    
    def assign_property(self, geometry_func, prop_name, value):
        """
        给几何体内的三棱柱赋物性值
        
        参数:
            geometry_func: 函数(x,y,z) → True/False
            prop_name: 'density', 'susceptibility', 'resistivity', 'velocity'
            value: 物性值
        """
        prop_array = getattr(self, prop_name)
        count = 0
        for i in range(self.n_prisms):
            x, y, z = self.prism_centers[i]
            if geometry_func(x, y, z):
                prop_array[i] = value
                count += 1
        print(f"  赋值: {count} 个三棱柱的 {prop_name} = {value}")
    
    def get_info(self):
        """打印网格信息"""
        info = f"""
        ╔══════════════════════════════════════════╗
        ║         2.5D 三棱柱网格信息                ║
        ╠══════════════════════════════════════════╣
        ║ 3D节点数: {len(self.nodes)}
        ║ 三棱柱数: {self.n_prisms}
        ║ 
        ║ XZ截面:
        ║   三角形数: {len(self.triangulation.triangles)}
        ║   点数: {len(self.triangulation.points)}
        ║
        ║ Y方向:
        ║   层数: {len(self.y_layers)}
        ║   范围: {self.y_layers[0]:.0f} ~ {self.y_layers[-1]:.0f} m
        ║
        ║ 体积:
        ║   最小: {self.prism_volumes.min():.1f} m³
        ║   最大: {self.prism_volumes.max():.1f} m³
        ║   总体积: {self.prism_volumes.sum():.0f} m³
        ╚══════════════════════════════════════════╝
        """
        print(info)
    
    def to_arrays(self):
        """导出为数组格式供正演使用"""
        return {
            'nodes': self.nodes,
            'prisms': self.prisms,
            'centers': self.prism_centers,
            'volumes': self.prism_volumes,
            'density': self.density,
            'susceptibility': self.susceptibility,
            'resistivity': self.resistivity,
            'velocity': self.velocity,
            'n_cells': self.n_prisms
        }