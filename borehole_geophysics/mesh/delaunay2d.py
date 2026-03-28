"""
自研2D Delaunay三角剖分

算法：Bowyer-Watson 增量插入法

用大白话说：
    1. 先画一个超大的三角形把所有点罩住
    2. 一个一个把点扔进去
    3. 每扔一个点，找到"不高兴"的三角形（外接圆包含新点的）
    4. 删掉不高兴的三角形，用新点重新连线
    5. 全部点扔完后，删掉最开始的超大三角形
"""

import numpy as np
import math


class DelaunayTriangulation2D:
    """
    自研2D Delaunay三角剖分
    
    使用方法：
        tri = DelaunayTriangulation2D()
        tri.triangulate(points)
        
        # 结果
        tri.points      → 所有点的坐标
        tri.triangles    → 所有三角形（每个是3个点的索引）
    """
    
    def __init__(self):
        self.points = None           # 所有点的坐标 [(x,z), ...]
        self.triangles = []          # 三角形列表 [(i,j,k), ...]
        self._super_indices = []     # 超级三角形的点索引（最后要删）
    
    def triangulate(self, points):
        """
        对一组2D点进行Delaunay三角剖分
        
        参数:
            points: 点的坐标列表 [(x1,z1), (x2,z2), ...]
                    或 numpy数组 shape=(N,2)
        
        返回:
            self（可以链式调用）
        
        结果存储在:
            self.points    → numpy数组 (N, 2)
            self.triangles → 列表 [(i,j,k), ...]
        """
        # 转成numpy数组
        pts = np.array(points, dtype=float)
        n_original = len(pts)
        
        print(f"  开始三角剖分，共 {n_original} 个点...")
        
        # ============================================
        # 第1步：创建超级三角形
        # ============================================
        super_pts = self._make_super_triangle(pts)
        
        # 把超级三角形的3个点加到点集末尾
        all_pts = np.vstack([pts, super_pts])
        self.points = all_pts
        
        # 超级三角形的索引
        si = n_original      # super index 0
        sj = n_original + 1  # super index 1
        sk = n_original + 2  # super index 2
        self._super_indices = [si, sj, sk]
        
        # 初始三角形列表：只有超级三角形
        self.triangles = [(si, sj, sk)]
        
        # ============================================
        # 第2步：逐点插入
        # ============================================
        for idx in range(n_original):
            if idx % 100 == 0 and idx > 0:
                print(f"    已插入 {idx}/{n_original} 个点...")
            self._insert_point(idx)
        
        # ============================================
        # 第3步：删除与超级三角形相关的三角形
        # ============================================
        self.triangles = [
            tri for tri in self.triangles
            if not any(v in self._super_indices for v in tri)
        ]
        
        # 只保留原始点
        self.points = pts
        
        print(f"  三角剖分完成：{len(self.triangles)} 个三角形")
        return self
    
    def _make_super_triangle(self, pts):
        """
        创建超级三角形：必须大到把所有点都包在里面
        
               p2
              ╱  ╲
             ╱    ╲
            ╱  所有 ╲
           ╱  点都在 ╲
          ╱   这里面  ╲
         ╱              ╲
        p0 ────────────── p1
        
        技巧：把超级三角形做得足够大（10倍于点集范围）
        """
        x_min, z_min = pts.min(axis=0)
        x_max, z_max = pts.max(axis=0)
        
        dx = x_max - x_min
        dz = z_max - z_min
        d = max(dx, dz)
        
        # 中心点
        cx = (x_min + x_max) / 2
        cz = (z_min + z_max) / 2
        
        # 超级三角形的3个顶点（足够大）
        margin = d * 10
        p0 = np.array([cx - margin, cz - margin])
        p1 = np.array([cx + margin, cz - margin])
        p2 = np.array([cx, cz + margin])
        
        return np.array([p0, p1, p2])
    
    def _insert_point(self, point_idx):
        """
        插入一个点（Bowyer-Watson核心步骤）
        
        步骤：
        1. 找到所有"坏三角形"（外接圆包含新点的）
        2. 找到坏三角形围成的"空洞"的边界边
        3. 删除坏三角形
        4. 用新点和边界边组成新三角形
        """
        point = self.points[point_idx]
        
        # ------ 第1步：找坏三角形 ------
        bad_triangles = []
        for tri in self.triangles:
            if self._point_in_circumcircle(point, tri):
                bad_triangles.append(tri)
        
        # ------ 第2步：找空洞的边界边 ------
        boundary_edges = []
        for tri in bad_triangles:
            # 三角形的3条边
            edges = [
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[2], tri[0])
            ]
            for edge in edges:
                # 检查这条边是不是只被一个坏三角形使用
                # 如果两个坏三角形共享这条边，那它不是边界
                is_boundary = True
                reversed_edge = (edge[1], edge[0])
                
                for other_tri in bad_triangles:
                    if other_tri is tri:
                        continue
                    other_edges = [
                        (other_tri[0], other_tri[1]),
                        (other_tri[1], other_tri[2]),
                        (other_tri[2], other_tri[0])
                    ]
                    if reversed_edge in other_edges or edge in other_edges:
                        is_boundary = False
                        break
                
                if is_boundary:
                    boundary_edges.append(edge)
        
        # ------ 第3步：删除坏三角形 ------
        for tri in bad_triangles:
            self.triangles.remove(tri)
        
        # ------ 第4步：用新点和边界边创建新三角形 ------
        for edge in boundary_edges:
            new_tri = (edge[0], edge[1], point_idx)
            self.triangles.append(new_tri)
    
    def _point_in_circumcircle(self, point, triangle):
        """
        判断一个点是否在三角形的外接圆内
        
        数学方法：用行列式判断
        
        如果行列式 > 0 → 点在外接圆内
        如果行列式 = 0 → 点在外接圆上
        如果行列式 < 0 → 点在外接圆外
        """
        ax, ay = self.points[triangle[0]]
        bx, by = self.points[triangle[1]]
        cx, cy = self.points[triangle[2]]
        dx, dy = point
        
        # 确保三角形是逆时针方向（行列式符号才正确）
        # 计算有向面积
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if area < 0:
            # 顺时针，交换两个点使其逆时针
            bx, by, cx, cy = cx, cy, bx, by
        
        # 行列式判断
        #  | ax-dx  ay-dy  (ax-dx)²+(ay-dy)² |
        #  | bx-dx  by-dy  (bx-dx)²+(by-dy)² |  > 0 → 点在外接圆内
        #  | cx-dx  cy-dy  (cx-dx)²+(cy-dy)² |
        
        a11 = ax - dx
        a12 = ay - dy
        a13 = a11 * a11 + a12 * a12
        
        a21 = bx - dx
        a22 = by - dy
        a23 = a21 * a21 + a22 * a22
        
        a31 = cx - dx
        a32 = cy - dy
        a33 = a31 * a31 + a32 * a32
        
        det = (a11 * (a22 * a33 - a23 * a32)
             - a12 * (a21 * a33 - a23 * a31)
             + a13 * (a21 * a32 - a22 * a31))
        
        return det > 0
    
    def get_triangle_points(self, tri_idx):
        """获取某个三角形的三个顶点坐标"""
        tri = self.triangles[tri_idx]
        return self.points[list(tri)]
    
    def get_edges(self):
        """获取所有不重复的边"""
        edge_set = set()
        for tri in self.triangles:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edge_set.add(edge)
        return list(edge_set)
    
    def triangle_area(self, tri_idx):
        """计算三角形面积"""
        tri = self.triangles[tri_idx]
        p0 = self.points[tri[0]]
        p1 = self.points[tri[1]]
        p2 = self.points[tri[2]]
        
        # 叉积公式
        area = 0.5 * abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1]) -
            (p2[0] - p0[0]) * (p1[1] - p0[1])
        )
        return area
    
    def triangle_center(self, tri_idx):
        """计算三角形重心"""
        pts = self.get_triangle_points(tri_idx)
        return pts.mean(axis=0)
    
    def get_info(self):
        """打印信息"""
        areas = [self.triangle_area(i) for i in range(len(self.triangles))]
        info = f"""
        ╔══════════════════════════════════════╗
        ║      2D Delaunay 三角剖分信息         ║
        ╠══════════════════════════════════════╣
        ║ 点数: {len(self.points)}
        ║ 三角形数: {len(self.triangles)}
        ║ 边数: {len(self.get_edges())}
        ║ 最小面积: {min(areas):.2f} m²
        ║ 最大面积: {max(areas):.2f} m²
        ║ 平均面积: {np.mean(areas):.2f} m²
        ╚══════════════════════════════════════╝
        """
        print(info)