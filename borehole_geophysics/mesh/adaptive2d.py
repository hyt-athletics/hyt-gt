"""
2D自适应网格加密

在Delaunay三角剖分的基础上，在需要精细的地方插入新点
"""

import numpy as np
from mesh.delaunay2d import DelaunayTriangulation2D


class AdaptiveTriangulation2D:
    """
    2D自适应三角化
    
    工作流程:
        1. 先定义边界点和初始内部点
        2. 根据加密规则，自动在需要的地方插入新点
        3. 对所有点做Delaunay三角化
    
    比直接调用 DelaunayTriangulation2D 多了一步：
        "智能地决定在哪里加点"
    """
    
    def __init__(self):
        self.boundary_points = []   # 边界点
        self.interior_points = []   # 内部点（加密后会增加）
        self.triangulation = None   # 三角化结果
    
    def set_rectangular_domain(self, x_range, z_range, n_boundary=20):
        """
        设置矩形域的边界
        
        参数:
            x_range: (x_min, x_max) 水平范围
            z_range: (z_min, z_max) 深度范围
            n_boundary: 每条边上的点数
        """
        x_min, x_max = x_range
        z_min, z_max = z_range
        
        # 四条边上的点（不重复角点）
        # 底边
        bottom = [(x, z_min) for x in np.linspace(x_min, x_max, n_boundary)]
        # 右边（去掉底边角点）
        right = [(x_max, z) for z in np.linspace(z_min, z_max, n_boundary)[1:]]
        # 顶边（去掉右边角点，反向）
        top = [(x, z_max) for x in np.linspace(x_max, x_min, n_boundary)[1:]]
        # 左边（去掉顶边角点和底边角点）
        left = [(x_min, z) for z in np.linspace(z_max, z_min, n_boundary)[1:-1]]
        
        self.boundary_points = bottom + right + top + left
        
        print(f"  矩形域边界: {len(self.boundary_points)} 个边界点")
        return self
    
    def add_refinement_point(self, x, z):
        """手动添加一个内部加密点"""
        self.interior_points.append((x, z))
    
    def refine_around_point(self, center, radii_and_counts):
        """
        在某个点周围添加同心圆上的加密点
        
        参数:
            center: (x, z) 加密中心（如钻孔位置）
            radii_and_counts: [(半径, 该圈上的点数), ...]
                              从内到外排列
        
        效果:
                   ·  ·
                ·   ·   ·
              ·    ●    ·      ● = center
                ·   ·   ·      · = 加密点
                   ·  ·         同心圆分布
        """
        cx, cz = center
        
        for radius, n_points in radii_and_counts:
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            for angle in angles:
                x = cx + radius * np.cos(angle)
                z = cz + radius * np.sin(angle)
                self.interior_points.append((x, z))
        
        print(f"  点加密: 在 ({cx:.0f}, {cz:.0f}) 周围添加了 "
              f"{sum(c for _, c in radii_and_counts)} 个点")
        return self
    
    def refine_along_line(self, start, end, spacing):
        """
        沿一条线段加密（如沿钻孔轨迹）
        
        参数:
            start: (x, z) 线段起点
            end: (x, z) 线段终点
            spacing: 加密点间距 (m)
        """
        sx, sz = start
        ex, ez = end
        length = np.sqrt((ex - sx)**2 + (ez - sz)**2)
        n = max(int(length / spacing), 2)
        
        for i in range(1, n):  # 跳过首尾点
            t = i / n
            x = sx + t * (ex - sx)
            z = sz + t * (ez - sz)
            self.interior_points.append((x, z))
        
        print(f"  线加密: 沿 ({sx:.0f},{sz:.0f})→({ex:.0f},{ez:.0f}) "
              f"添加了 {n-1} 个点，间距 {spacing:.1f}m")
        return self
    
    def refine_near_line_graded(self, start, end, 
                                 min_spacing, max_spacing, 
                                 influence_radius, n_layers=5):
        """
        沿线段渐变加密：近处密、远处疏
        
        这是钻孔加密最常用的方法!
        
        参数:
            start, end: 线段的起点和终点（钻孔顶部和底部在截面上的位置）
            min_spacing: 最近处的点间距
            max_spacing: 最远处的点间距
            influence_radius: 影响半径
            n_layers: 加密层数
        
        效果:
            ·  ·  ·  ·  ·  ·  ·  ·       ← 远处，稀疏
             ·   ·   ·   ·   ·   ·
              ·  ·  ·  ·  ·  ·            ← 中间，中等
               · · · · · · ·
                ·····●·····               ← 近处，密集
               · · · · · · ·              ● = 钻孔
              ·  ·  ·  ·  ·  ·
             ·   ·   ·   ·   ·   ·
            ·  ·  ·  ·  ·  ·  ·  ·       ← 远处，稀疏
        """
        sx, sz = start
        ex, ez = end
        line_length = np.sqrt((ex - sx)**2 + (ez - sz)**2)
        
        # 线段方向和法线方向
        dx = ex - sx
        dz = ez - sz
        length = np.sqrt(dx*dx + dz*dz)
        dir_x, dir_z = dx / length, dz / length     # 线段方向
        norm_x, norm_z = -dir_z, dir_x               # 法线方向
        
        count = 0
        for layer in range(1, n_layers + 1):
            # 这一层离线段的距离
            t = layer / n_layers
            distance = t * influence_radius
            
            # 这一层的点间距（线性插值）
            spacing = min_spacing + t * (max_spacing - min_spacing)
            
            # 沿线段方向的点数
            n_along = max(int(line_length / spacing), 3)
            
            for i in range(n_along + 1):
                s = i / n_along
                # 线段上的基准点
                base_x = sx + s * dx
                base_z = sz + s * dz
                
                # 法线方向两侧各加一个点
                for sign in [1, -1]:
                    px = base_x + sign * distance * norm_x
                    pz = base_z + sign * distance * norm_z
                    self.interior_points.append((px, pz))
                    count += 1
        
        print(f"  渐变加密: 共添加 {count} 个点，"
              f"间距 {min_spacing:.0f}~{max_spacing:.0f}m，"
              f"影响半径 {influence_radius:.0f}m")
        return self
    
    def refine_at_polygon_boundary(self, polygon_points, spacing):
        """
        沿多边形边界加密（用于不规则异常体）
        
        参数:
            polygon_points: [(x1,z1), (x2,z2), ...] 多边形顶点（闭合）
            spacing: 边界上的加密点间距
        """
        count = 0
        n = len(polygon_points)
        for i in range(n):
            p1 = np.array(polygon_points[i])
            p2 = np.array(polygon_points[(i + 1) % n])
            
            edge_length = np.linalg.norm(p2 - p1)
            n_seg = max(int(edge_length / spacing), 1)
            
            for j in range(1, n_seg):
                t = j / n_seg
                pt = p1 + t * (p2 - p1)
                self.interior_points.append(tuple(pt))
                count += 1
        
        print(f"  多边形边界加密: 添加 {count} 个点，间距 {spacing:.0f}m")
        return self
    
    def generate_background_points(self, x_range, z_range, spacing):
        """
        生成均匀背景点（避免远处没有点导致三角形太大）
        
        参数:
            x_range: (x_min, x_max)
            z_range: (z_min, z_max)
            spacing: 背景点间距
        """
        x_min, x_max = x_range
        z_min, z_max = z_range
        
        count = 0
        x = x_min + spacing / 2
        while x < x_max:
            z = z_min + spacing / 2
            while z < z_max:
                # 检查是否离已有的点太近（避免重复）
                too_close = False
                for px, pz in self.boundary_points + self.interior_points:
                    if (x - px)**2 + (z - pz)**2 < (spacing / 3)**2:
                        too_close = True
                        break
                
                if not too_close:
                    self.interior_points.append((x, z))
                    count += 1
                
                z += spacing
            x += spacing
        
        print(f"  背景点: 添加 {count} 个，间距 {spacing:.0f}m")
        return self
    
    def build(self):
        """
        执行三角化
        
        把所有点（边界+内部）合并，调用Delaunay三角化
        """
        all_points = self.boundary_points + self.interior_points
        print(f"\n  总点数: {len(all_points)} "
              f"(边界:{len(self.boundary_points)}, "
              f"内部:{len(self.interior_points)})")
        
        self.triangulation = DelaunayTriangulation2D()
        self.triangulation.triangulate(all_points)
        
        return self.triangulation