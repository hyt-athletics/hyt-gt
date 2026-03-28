"""
3D自适应网格点生成

核心思路：
    不是直接操作四面体（太复杂）
    而是"智能地撒点"——在需要精细的地方多撒，远处少撒
    然后把所有点扔给Delaunay去连接

    就像种花：
      花园中心（钻孔旁）种得密
      花园边缘种得疏
      花坛边缘（矿体边界）种一圈

    Delaunay会自动把这些点连成好的四面体
"""

import numpy as np


class AdaptivePointGenerator3D:
    """
    3D自适应点生成器
    
    使用流程:
        gen = AdaptivePointGenerator3D()
        gen.set_domain(...)
        gen.add_background_points(...)
        gen.refine_along_borehole(...)
        gen.refine_at_boundary(...)
        points = gen.get_points()
    """
    
    def __init__(self):
        self.domain = None       # (x_range, y_range, z_range)
        self.points = []         # 所有点 [(x,y,z), ...]
        self._labels = []        # 每个点的标签（用于调试）
    
    def set_domain(self, x_range, y_range, z_range):
        """设置模型域"""
        self.domain = (x_range, y_range, z_range)
        print(f"  域: X={x_range}, Y={y_range}, Z={z_range}")
        return self
    
    def _in_domain(self, x, y, z, margin=0):
        """判断点是否在域内"""
        (x0, x1), (y0, y1), (z0, z1) = self.domain
        return (x0 + margin <= x <= x1 - margin and
                y0 + margin <= y <= y1 - margin and
                z0 + margin <= z <= z1 - margin)
    
    def _add_point(self, x, y, z, label=""):
        """添加一个点（自动检查是否在域内）"""
        if self._in_domain(x, y, z):
            self.points.append((x, y, z))
            self._labels.append(label)
    
    def add_domain_boundary_points(self, spacing):
        """
        在域的六个面上撒点
        
        参数:
            spacing: 面上的点间距
        """
        (x0, x1), (y0, y1), (z0, z1) = self.domain
        count = 0
        
        xs = np.arange(x0, x1 + spacing / 2, spacing)
        ys = np.arange(y0, y1 + spacing / 2, spacing)
        zs = np.arange(z0, z1 + spacing / 2, spacing)
        
        # 6个面
        for y in ys:
            for z in zs:
                self._add_point(x0, y, z, "boundary"); count += 1
                self._add_point(x1, y, z, "boundary"); count += 1
        for x in xs:
            for z in zs:
                self._add_point(x, y0, z, "boundary"); count += 1
                self._add_point(x, y1, z, "boundary"); count += 1
        for x in xs:
            for y in ys:
                self._add_point(x, y, z0, "boundary"); count += 1
                self._add_point(x, y, z1, "boundary"); count += 1
        
        print(f"  域边界点: {count} 个, 间距 {spacing}m")
        return self
    
    def add_background_points(self, spacing):
        """
        在域内均匀撒背景点
        
        参数:
            spacing: 点间距 (m)
        """
        (x0, x1), (y0, y1), (z0, z1) = self.domain
        
        # 偏移半个间距，避免和边界点重合
        half = spacing / 2
        xs = np.arange(x0 + half, x1, spacing)
        ys = np.arange(y0 + half, y1, spacing)
        zs = np.arange(z0 + half, z1, spacing)
        
        count = 0
        for x in xs:
            for y in ys:
                for z in zs:
                    self._add_point(x, y, z, "background")
                    count += 1
        
        print(f"  背景点: {count} 个, 间距 {spacing}m")
        return self
    
    def refine_along_borehole(self, well, min_spacing, max_spacing,
                                influence_radius, n_layers=5):
        """
        沿钻孔渐变加密（最重要的加密方法！）
        
        在钻孔周围形成同心"管状"加密区：
        
        俯视图（XY平面）:
             ·   ·   ·          ← 远处，稀疏
           ·   ·   ·   ·
          ·  · · · · ·  ·
          · · ···●··· · ·       ● = 钻孔
          ·  · · · · ·  ·
           ·   ·   ·   ·
             ·   ·   ·          ← 远处，稀疏
        
        侧视图（XZ平面）:同样的渐变
        
        参数:
            well: VerticalWell 对象
            min_spacing: 最近处的点间距 (m)
            max_spacing: 最远处的点间距 (m)
            influence_radius: 影响半径 (m)
            n_layers: 加密层数
        """
        path_points = well.get_path_points(n_points=50)
        
        count = 0
        
        for layer in range(1, n_layers + 1):
            # 这一层的距离和间距
            t = layer / n_layers
            distance = t * influence_radius
            spacing = min_spacing + t * (max_spacing - min_spacing)
            
            # 沿钻孔路径的间距
            n_along = max(int(well.z_bottom - well.z_top) // int(spacing), 3)
            z_positions = np.linspace(well.z_top, well.z_bottom, n_along)
            
            # 在每个深度，沿圆周撒点
            n_around = max(int(2 * np.pi * distance / spacing), 6)
            angles = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
            
            for z in z_positions:
                for angle in angles:
                    x = well.x + distance * np.cos(angle)
                    y = well.y + distance * np.sin(angle)
                    self._add_point(x, y, z, f"borehole_L{layer}")
                    count += 1
        
        # 钻孔轴线上的点（最密集）
        n_axis = max(int((well.z_bottom - well.z_top) / min_spacing), 10)
        for z in np.linspace(well.z_top, well.z_bottom, n_axis):
            self._add_point(well.x, well.y, z, "borehole_axis")
            count += 1
        
        print(f"  钻孔加密: {count} 个点, "
              f"间距 {min_spacing}~{max_spacing}m, "
              f"影响半径 {influence_radius}m")
        return self
    
    def refine_around_point(self, center, radii_and_counts):
        """
        在某个点周围球面加密
        
        参数:
            center: (x, y, z)
            radii_and_counts: [(半径, 每球面上的点数), ...]
        """
        cx, cy, cz = center
        count = 0
        
        for radius, n_pts in radii_and_counts:
            # 在球面上均匀分布点（斐波那契球面）
            pts = self._fibonacci_sphere(n_pts)
            for px, py, pz in pts:
                x = cx + radius * px
                y = cy + radius * py
                z = cz + radius * pz
                self._add_point(x, y, z, "point_refine")
                count += 1
        
        self._add_point(cx, cy, cz, "point_refine_center")
        count += 1
        
        print(f"  点加密: 在 ({cx:.0f},{cy:.0f},{cz:.0f}) 周围 {count} 个点")
        return self
    
    def refine_at_surface(self, geometry_func, spacing, n_samples=20):
        """
        在几何体表面加密
        
        原理：在域内均匀采样，找到几何体内外交界的位置，
              在那里加密撒点
        
        参数:
            geometry_func: 函数(x,y,z) → True/False
            spacing: 表面附近的加密间距
            n_samples: 每个方向的采样数
        """
        (x0, x1), (y0, y1), (z0, z1) = self.domain
        
        xs = np.linspace(x0, x1, n_samples)
        ys = np.linspace(y0, y1, n_samples)
        zs = np.linspace(z0, z1, n_samples)
        
        count = 0
        
        # 扫描每条线，找到从"内"变"外"或反过来的位置
        # X方向扫描
        for y in ys:
            for z in zs:
                prev_inside = None
                for x in np.linspace(x0, x1, n_samples * 3):
                    inside = geometry_func(x, y, z)
                    if prev_inside is not None and inside != prev_inside:
                        # 找到边界！在这里撒点
                        for dx in np.linspace(-spacing, spacing, 3):
                            for dy in np.linspace(-spacing, spacing, 3):
                                for dz in np.linspace(-spacing, spacing, 3):
                                    if dx*dx + dy*dy + dz*dz <= spacing*spacing:
                                        self._add_point(x+dx, y+dy, z+dz, "surface")
                                        count += 1
                    prev_inside = inside
        
        # Y方向和Z方向类似（为简洁省略重复代码，但实际也要做）
        for x in xs:
            for z in zs:
                prev_inside = None
                for y in np.linspace(y0, y1, n_samples * 3):
                    inside = geometry_func(x, y, z)
                    if prev_inside is not None and inside != prev_inside:
                        for dx in np.linspace(-spacing, spacing, 3):
                            for dy in np.linspace(-spacing, spacing, 3):
                                for dz in np.linspace(-spacing, spacing, 3):
                                    if dx*dx + dy*dy + dz*dz <= spacing*spacing:
                                        self._add_point(x+dx, y+dy, z+dz, "surface")
                                        count += 1
                    prev_inside = inside
        
        for x in xs:
            for y in ys:
                prev_inside = None
                for z in np.linspace(z0, z1, n_samples * 3):
                    inside = geometry_func(x, y, z)
                    if prev_inside is not None and inside != prev_inside:
                        for dx in np.linspace(-spacing, spacing, 3):
                            for dy in np.linspace(-spacing, spacing, 3):
                                for dz in np.linspace(-spacing, spacing, 3):
                                    if dx*dx + dy*dy + dz*dz <= spacing*spacing:
                                        self._add_point(x+dx, y+dy, z+dz, "surface")
                                        count += 1
                    prev_inside = inside
        
        # 去除重复点
        self._remove_duplicates(min_dist=spacing / 3)
        
        print(f"  表面加密: 约 {count} 个点, 间距 {spacing}m")
        return self
    
    def _fibonacci_sphere(self, n_points):
        """
        斐波那契球面点分布
        
        在球面上近似均匀地分布n个点
        比随机分布均匀得多
        """
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n_points):
            theta = np.arccos(1 - 2 * (i + 0.5) / n_points)
            phi = 2 * np.pi * i / golden_ratio
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append((x, y, z))
        
        return points
    
    def _remove_duplicates(self, min_dist):
        """
        移除距离太近的点
        
        简化版本：格子法，不完美但够用
        """
        if len(self.points) == 0:
            return
        
        pts = np.array(self.points)
        keep = np.ones(len(pts), dtype=bool)
        
        # 简单方法：按顺序扫描，距离太近就删后面的
        # 对于大量点可能慢，但对于我们的规模够用
        for i in range(len(pts)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(pts)):
                if not keep[j]:
                    continue
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < min_dist:
                    keep[j] = False
        
        old_count = len(self.points)
        self.points = [self.points[i] for i in range(len(self.points)) if keep[i]]
        self._labels = [self._labels[i] for i in range(len(self._labels)) if keep[i]]
        
        removed = old_count - len(self.points)
        if removed > 0:
            print(f"    去重: 移除 {removed} 个过近的点")
    
    def get_points(self):
        """获取所有点"""
        self._remove_duplicates(min_dist=5.0)  # 至少间隔5m
        return np.array(self.points)
    
    def get_info(self):
        """打印点分布统计"""
        labels = self._labels
        label_counts = {}
        for label in labels:
            base_label = label.split("_L")[0] if "_L" in label else label
            label_counts[base_label] = label_counts.get(base_label, 0) + 1
        
        print(f"\n  点分布统计 (共 {len(self.points)} 个点):")
        for label, count in sorted(label_counts.items()):
            print(f"    {label}: {count} 个")