"""
约束Delaunay三角化（Constrained Delaunay Triangulation）

在普通Delaunay的基础上，强制某些边出现在最终三角网格中

算法：边恢复算法（Edge Insertion Algorithm）
  1. 先做普通Delaunay
  2. 对每条约束边：
     a. 检查这条边是否已存在
     b. 如果不存在，找到所有穿过它的三角形
     c. 删掉这些三角形，形成多边形空洞
     d. 在空洞内重新三角化，强制包含这条边
"""

import numpy as np
from mesh.delaunay2d import DelaunayTriangulation2D


class ConstrainedDelaunayTriangulation:
    """
    约束Delaunay三角化
    
    使用方法:
        cdt = ConstrainedDelaunayTriangulation()
        cdt.triangulate(points, constraints)
        
    参数:
        points: 所有点坐标 [(x,z), ...]
        constraints: 约束边列表 [(i,j), (k,l), ...]
                     每个约束边是两个点索引的元组
    """
    
    def __init__(self):
        self.points = None
        self.triangles = []
        self.constraints = []      # 已成功插入的约束边
        self._original_points = None
    
    def triangulate(self, points, constraints):
        """
        执行约束Delaunay三角化
        
        参数:
            points: 点坐标列表 [(x1,z1), (x2,z2), ...]
            constraints: 约束边列表 [(i, j), ...]
                        注意：i,j 是点的索引，不是坐标
        
        返回:
            self
        """
        print(f"开始约束Delaunay三角化...")
        print(f"  点数: {len(points)}")
        print(f"  约束边数: {len(constraints)}")
        
        self._original_points = np.array(points, dtype=float)
        
        # ============================================
        # 第1步：普通Delaunay三角化
        # ============================================
        print("  第1步：执行普通Delaunay...")
        dt = DelaunayTriangulation2D()
        dt.triangulate(points)
        
        self.points = dt.points.copy()
        self.triangles = dt.triangles.copy()
        print(f"    初始三角形: {len(self.triangles)} 个")
        
        # ============================================
        # 第2步：逐条插入约束边
        # ============================================
        print("  第2步：插入约束边...")
        self.constraints = []
        
        for idx, (i, j) in enumerate(constraints):
            if idx % 10 == 0 and idx > 0:
                print(f"    已插入 {idx}/{len(constraints)} 条约束边...")
            
            success = self._insert_constraint_edge(i, j)
            if success:
                self.constraints.append((i, j))
            else:
                print(f"    警告：约束边 ({i}, {j}) 插入失败")
        
        print(f"  完成！成功插入 {len(self.constraints)}/{len(constraints)} 条约束边")
        print(f"  最终三角形: {len(self.triangles)} 个")
        
        return self
    
    def _insert_constraint_edge(self, i, j):
        """
        插入一条约束边
        
        如果这条边已经存在，直接返回True
        如果不存在，执行边恢复算法
        """
        # 检查边是否已经存在
        if self._edge_exists(i, j):
            return True
        
        # 找到所有穿过这条边的三角形
        intersecting_triangles = self._find_intersecting_triangles(i, j)
        
        if len(intersecting_triangles) == 0:
            # 没有三角形穿过这条边，直接添加
            return True
        
        # 构建受影响区域的边界（多边形空洞）
        polygon = self._build_polygon_from_triangles(intersecting_triangles)
        
        # 从多边形中移除边(i,j)，得到两个子多边形
        poly1, poly2 = self._split_polygon_by_edge(polygon, i, j)
        
        # 删除旧三角形
        for tri in intersecting_triangles:
            if tri in self.triangles:
                self.triangles.remove(tri)
        
        # 对两个子多边形分别三角化
        self._triangulate_polygon(poly1)
        self._triangulate_polygon(poly2)
        
        return True
    
    def _edge_exists(self, i, j):
        """检查边(i,j)是否已经存在于某个三角形中"""
        for tri in self.triangles:
            edges = [
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[2], tri[0])
            ]
            for (a, b) in edges:
                if (a == i and b == j) or (a == j and b == i):
                    return True
        return False
    
    def _find_intersecting_triangles(self, i, j):
        """
        找到所有与线段(i,j)相交的三角形
        
        一个三角形与线段相交的条件：
          线段穿过三角形的内部（不包括端点）
        """
        p_i = self.points[i]
        p_j = self.points[j]
        intersecting = []
        
        for tri in self.triangles:
            # 如果三角形包含i或j，跳过（端点接触不算穿过）
            if i in tri or j in tri:
                continue
            
            # 检查线段是否与三角形相交
            if self._segment_intersects_triangle(p_i, p_j, tri):
                intersecting.append(tri)
        
        return intersecting
    
    def _segment_intersects_triangle(self, p1, p2, triangle):
        """
        判断线段是否与三角形相交
        
        方法：检查线段是否与三角形的任一条边相交
        """
        tri_pts = [self.points[idx] for idx in triangle]
        
        # 三角形的3条边
        edges = [
            (tri_pts[0], tri_pts[1]),
            (tri_pts[1], tri_pts[2]),
            (tri_pts[2], tri_pts[0])
        ]
        
        for e1, e2 in edges:
            if self._segments_intersect(p1, p2, e1, e2):
                return True
        
        return False
    
    def _segments_intersect(self, a1, a2, b1, b2):
        """
        判断两条线段是否相交
        
        使用定向面积法
        """
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        def on_segment(p, a, b):
            return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                    min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))
        
        d1 = cross(a1, a2, b1)
        d2 = cross(a1, a2, b2)
        d3 = cross(b1, b2, a1)
        d4 = cross(b1, b2, a2)
        
        # 一般情况
        if d1 * d2 < 0 and d3 * d4 < 0:
            return True
        
        # 特殊情况：端点在另一条线段上
        if d1 == 0 and on_segment(b1, a1, a2):
            return True
        if d2 == 0 and on_segment(b2, a1, a2):
            return True
        if d3 == 0 and on_segment(a1, b1, b2):
            return True
        if d4 == 0 and on_segment(a2, b1, b2):
            return True
        
        return False
    
    def _build_polygon_from_triangles(self, triangles):
        """
        从一组三角形构建外边界多边形
        
        算法：
          1. 收集所有边
          2. 找出只出现一次的边（边界边）
          3. 把边界边按顺序连接成多边形
        """
        # 统计每条边出现的次数
        edge_count = {}
        for tri in triangles:
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # 边界边 = 只出现一次的边
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        if len(boundary_edges) == 0:
            return []
        
        # 把边界边连接成多边形
        polygon = [boundary_edges[0][0], boundary_edges[0][1]]
        boundary_edges.pop(0)
        
        while boundary_edges:
            last = polygon[-1]
            found = False
            for idx, (a, b) in enumerate(boundary_edges):
                if a == last:
                    polygon.append(b)
                    boundary_edges.pop(idx)
                    found = True
                    break
                elif b == last:
                    polygon.append(a)
                    boundary_edges.pop(idx)
                    found = True
                    break
            if not found:
                break
        
        return polygon
    
    def _split_polygon_by_edge(self, polygon, i, j):
        """
        用边(i,j)把多边形分成两部分
        
        前提：i和j都在polygon中
        
        返回: poly1, poly2 两个子多边形
        """
        if not polygon or i not in polygon or j not in polygon:
            return [], []
        
        idx_i = polygon.index(i)
        idx_j = polygon.index(j)
        
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i
        
        # 子多边形1: 从i到j
        poly1 = polygon[idx_i:idx_j + 1]
        
        # 子多边形2: 从j到末尾+开头到i
        poly2 = polygon[idx_j:] + polygon[:idx_i + 1]
        
        return poly1, poly2
    
    def _triangulate_polygon(self, polygon):
        """
        对一个多边形进行三角化（耳切法）
        
        耳切法（Ear Clipping）：
          "耳朵" = 多边形的三个连续顶点，形成一个三角形且不包含其他点
          反复切掉耳朵，直到剩下最后一个三角形
        """
        if len(polygon) < 3:
            return
        
        # 简化：使用Delaunay三角化这些点，然后只保留在多边形内的三角形
        poly_points = [self.points[idx] for idx in polygon]
        
        dt = DelaunayTriangulation2D()
        dt.triangulate(poly_points)
        
        # 过滤：只保留在原多边形内的三角形
        from geometry.polygon import point_in_polygon_2d
        
        for tri in dt.triangles:
            # 计算三角形重心
            p0, p1, p2 = [poly_points[i] for i in tri]
            cx = (p0[0] + p1[0] + p2[0]) / 3
            cz = (p0[1] + p1[1] + p2[1]) / 3
            
            # 检查重心是否在多边形内
            if point_in_polygon_2d(cx, cz, poly_points):
                # 把局部索引映射回全局索引
                global_tri = (polygon[tri[0]], polygon[tri[1]], polygon[tri[2]])
                self.triangles.append(global_tri)
    
    def get_info(self):
        """打印信息"""
        info = f"""
        ╔══════════════════════════════════════════╗
        ║      约束Delaunay三角剖分信息              ║
        ╠══════════════════════════════════════════╣
        ║ 点数: {len(self.points)}
        ║ 三角形数: {len(self.triangles)}
        ║ 约束边数: {len(self.constraints)}
        ╚══════════════════════════════════════════╝
        """
        print(info)