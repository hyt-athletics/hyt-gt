r"""
边翻转算法 (Edge Flip)

用于约束Delaunay三角化中恢复约束边

原理：
    翻转前:                    翻转后:
        d                         d
       /|\                       / \
      / | \                     /   \
     /  |  \                   /     \
    a---|---b      ->          a-------b
     \  |  /                   \     /
      \ | /                     \   /
       \|/                       \ /
        c                         c

    两个三角形共享边 -> 翻转对角线

算法：
    1. 单次翻转：检查凸性 -> 翻转对角线
    2. 链式翻转：为约束边找到翻转序列，逐步翻转直到约束边出现

参考：
    - Shewchuk, J.R. "Triangle: Engineering a 2D Quality Mesh Generator"
    - CGAL Triangulation_2 documentation
"""

import numpy as np


class EdgeFlipper:
    """
    边翻转器
    
    用于约束Delaunay三角化中恢复约束边
    
    使用方法:
        flipper = EdgeFlipper(points, triangles)
        flipper.flip_chain_for_constraint(i, j)
        triangles = flipper.triangles
    """
    
    def __init__(self, points, triangles):
        """
        初始化边翻转器
        
        参数:
            points: 点坐标数组 (N, 2)
            triangles: 三角形列表 [(i,j,k), ...]
        """
        self.points = np.array(points, dtype=float)
        self.triangles = list(triangles)
        self._build_edge_map()
    
    def _build_edge_map(self):
        """构建边到三角形的映射，加速查找"""
        self._edge_to_tris = {}
        for tri_idx, tri in enumerate(self.triangles):
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted([a, b]))
                if edge not in self._edge_to_tris:
                    self._edge_to_tris[edge] = []
                self._edge_to_tris[edge].append(tri_idx)
    
    def _update_edge_map(self, tri_idx, old_tri, new_tri):
        """更新边映射"""
        for i in range(3):
            a, b = old_tri[i], old_tri[(i + 1) % 3]
            edge = tuple(sorted([a, b]))
            if edge in self._edge_to_tris and tri_idx in self._edge_to_tris[edge]:
                self._edge_to_tris[edge].remove(tri_idx)
        
        for i in range(3):
            a, b = new_tri[i], new_tri[(i + 1) % 3]
            edge = tuple(sorted([a, b]))
            if edge not in self._edge_to_tris:
                self._edge_to_tris[edge] = []
            self._edge_to_tris[edge].append(tri_idx)
    
    def _get_opposite_vertices(self, tri, a, b):
        """获取三角形中边之外的两个顶点"""
        vertices = set(tri)
        vertices.discard(a)
        vertices.discard(b)
        return list(vertices)
    
    def _find_adjacent_triangle(self, tri_idx, edge):
        """找到共享指定边的相邻三角形"""
        edge_sorted = tuple(sorted(edge))
        if edge_sorted not in self._edge_to_tris:
            return None
        for adj_idx in self._edge_to_tris[edge_sorted]:
            if adj_idx != tri_idx:
                return adj_idx
        return None
    
    def _cross2d(self, o, a, b):
        """二维叉积: (a-o) × (b-o)"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def _point_on_left(self, p, a, b):
        """判断点p是否在有向边的左侧"""
        return self._cross2d(a, b, p) > 0
    
    def _point_on_right(self, p, a, b):
        """判断点p是否在有向边的右侧"""
        return self._cross2d(a, b, p) < 0
    
    def is_convex_quad(self, tri1, tri2, edge):
        """
        检查两个三角形形成的四边形是否为凸四边形
        
        凸四边形才能进行边翻转
        
        参数:
            tri1, tri2: 两个相邻三角形
            edge: 共享边
        
        返回:
            True 如果是凸四边形
        """
        a, b = edge
        c = self._get_opposite_vertices(tri1, a, b)[0]
        d = self._get_opposite_vertices(tri2, a, b)[0]
        
        pa, pb = self.points[a], self.points[b]
        pc, pd = self.points[c], self.points[d]
        
        cross_c = self._cross2d(pa, pb, pc)
        cross_d = self._cross2d(pa, pb, pd)
        
        if cross_c * cross_d >= 0:
            return False
        
        cross_a = self._cross2d(pc, pd, pa)
        cross_b = self._cross2d(pc, pd, pb)
        
        if cross_a * cross_b >= 0:
            return False
        
        return True
    
    def can_flip(self, tri_idx, edge):
        """
        检查边是否可以翻转
        
        条件:
            1. 边必须是两个三角形的公共边
            2. 四边形必须是凸的
            3. 翻转后不产生重复三角形
        
        参数:
            tri_idx: 三角形索引
            edge: 要翻转的边
        
        返回:
            (can_flip, adj_tri_idx) 或
        """
        adj_idx = self._find_adjacent_triangle(tri_idx, edge)
        if adj_idx is None:
            return False, None
        
        tri1 = self.triangles[tri_idx]
        tri2 = self.triangles[adj_idx]
        
        if not self.is_convex_quad(tri1, tri2, edge):
            return False, adj_idx
        
        return True, adj_idx
    
    def flip_edge(self, tri_idx, edge):
        """
        执行单次边翻转
        
        参数:
            tri_idx: 三角形索引
            edge: 要翻转的边
        
        返回:
            True 如果翻转成功
        """
        can_flip, adj_idx = self.can_flip(tri_idx, edge)
        if not can_flip:
            return False
        
        tri1 = self.triangles[tri_idx]
        tri2 = self.triangles[adj_idx]
        a, b = edge
        
        c = self._get_opposite_vertices(tri1, a, b)[0]
        d = self._get_opposite_vertices(tri2, a, b)[0]
        
        new_tri1 = (c, d, a)
        new_tri2 = (d, c, b)
        
        self._update_edge_map(tri_idx, tri1, new_tri1)
        self._update_edge_map(adj_idx, tri2, new_tri2)
        
        self.triangles[tri_idx] = new_tri1
        self.triangles[adj_idx] = new_tri2
        
        return True
    
    def _segments_intersect(self, p1, p2, q1, q2):
        """判断两条线段是否相交（不包括端点）"""
        def sign(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0
        
        d1 = sign(self._cross2d(q1, q2, p1))
        d2 = sign(self._cross2d(q1, q2, p2))
        d3 = sign(self._cross2d(p1, p2, q1))
        d4 = sign(self._cross2d(p1, p2, q2))
        
        if d1 * d2 < 0 and d3 * d4 < 0:
            return True
        
        return False
    
    def _edge_intersects_constraint(self, edge, constraint):
        """检查边是否与约束边相交"""
        a, b = edge
        i, j = constraint
        
        if a == i or a == j or b == i or b == j:
            return False
        
        pa, pb = self.points[a], self.points[b]
        pi, pj = self.points[i], self.points[j]
        
        return self._segments_intersect(pa, pb, pi, pj)
    
    def _find_intersecting_edge(self, constraint):
        """找到与约束边相交的一条边"""
        i, j = constraint
        
        for tri_idx, tri in enumerate(self.triangles):
            for k in range(3):
                a, b = tri[k], tri[(k + 1) % 3]
                edge = (a, b)
                
                if self._edge_intersects_constraint(edge, constraint):
                    return tri_idx, edge
        
        return None, None
    
    def flip_chain_for_constraint(self, i, j):
        """
        通过链式翻转恢复约束边
        
        算法:
            1. 找到与约束边相交的边
            2. 翻转该边
            3. 重复直到约束边出现
        
        参数:
            i, j: 约束边的两个端点索引
        
        返回:
            True 如果成功恢复约束边
        """
        constraint = (i, j)
        max_iterations = len(self.triangles) * 3
        iterations = 0
        
        while iterations < max_iterations:
            if self._edge_exists(i, j):
                return True
            
            tri_idx, edge = self._find_intersecting_edge(constraint)
            if tri_idx is None:
                break
            
            if not self.flip_edge(tri_idx, edge):
                break
            
            iterations += 1
        
        return self._edge_exists(i, j)
    
    def _edge_exists(self, i, j):
        """检查边是否已存在"""
        edge = tuple(sorted([i, j]))
        return edge in self._edge_to_tris and len(self._edge_to_tris[edge]) > 0
    
    def flip_to_delaunay(self, max_iterations=None):
        """
        通过边翻转恢复Delaunay性质
        
        算法:
            对每条边，如果翻转能改善Delaunay条件，则翻转
        
        参数:
            max_iterations: 最大迭代次数
        
        返回:
            翻转次数
        """
        if max_iterations is None:
            max_iterations = len(self.triangles) * 10
        
        flip_count = 0
        improved = True
        
        while improved and flip_count < max_iterations:
            improved = False
            
            for tri_idx, tri in enumerate(self.triangles):
                for k in range(3):
                    edge = (tri[k], tri[(k + 1) % 3])
                    
                    if self._should_flip_for_delaunay(tri_idx, edge):
                        if self.flip_edge(tri_idx, edge):
                            flip_count += 1
                            improved = True
                            break
                
                if improved:
                    break
        
        return flip_count
    
    def _should_flip_for_delaunay(self, tri_idx, edge):
        """检查是否应该翻转以改善Delaunay条件"""
        adj_idx = self._find_adjacent_triangle(tri_idx, edge)
        if adj_idx is None:
            return False
        
        tri1 = self.triangles[tri_idx]
        tri2 = self.triangles[adj_idx]
        
        if not self.is_convex_quad(tri1, tri2, edge):
            return False
        
        a, b = edge
        c = self._get_opposite_vertices(tri1, a, b)[0]
        d = self._get_opposite_vertices(tri2, a, b)[0]
        
        if self._point_in_circumcircle(d, tri1):
            return True
        if self._point_in_circumcircle(c, tri2):
            return True
        
        return False
    
    def _point_in_circumcircle(self, point_idx, triangle):
        """检查点是否在三角形的外接圆内"""
        p = self.points[point_idx]
        tri_pts = [self.points[i] for i in triangle]
        
        ax, ay = tri_pts[0]
        bx, by = tri_pts[1]
        cx, cy = tri_pts[2]
        dx, dy = p
        
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
    
    def get_flipped_triangles(self):
        """获取翻转后的三角形列表"""
        return self.triangles.copy()


def flip_edges_for_constraints(points, triangles, constraints):
    """
    便捷函数：为多条约束边执行边翻转
    
    参数:
        points: 点坐标数组 (N, 2)
        triangles: 三角形列表 [(i,j,k), ...]
        constraints: 约束边列表 [(i,j), ...]
    
    返回:
        triangles: 更新后的三角形列表
        success_count: 成功恢复的约束边数量
    """
    flipper = EdgeFlipper(points, triangles)
    success_count = 0
    
    for i, j in constraints:
        if flipper.flip_chain_for_constraint(i, j):
            success_count += 1
    
    return flipper.get_flipped_triangles(), success_count


def demo():
    """演示边翻转算法"""
    print("╔══════════════════════════════════════════╗")
    print("║       边翻转算法演示                      ║")
    print("╚══════════════════════════════════════════╝")
    
    points = np.array([
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
    ])
    
    triangles = [
        (0, 1, 2),
        (0, 2, 3),
    ]
    
    print("\n初始三角形:")
    for i, tri in enumerate(triangles):
        print(f"  三角形 {i}: {tri}")
    print("  共享边: (0, 2)")
    
    flipper = EdgeFlipper(points, triangles)
    
    print("\n检查共享边 (0, 2) 是否可翻转...")
    can_flip, adj = flipper.can_flip(0, (0, 2))
    print(f"  可翻转: {can_flip}, 相邻三角形索引: {adj}")
    
    if can_flip:
        print("\n执行翻转...")
        success = flipper.flip_edge(0, (0, 2))
        print(f"  翻转成功: {success}")
        
        print("\n翻转后三角形:")
        for i, tri in enumerate(flipper.triangles):
            print(f"  三角形 {i}: {tri}")
        print("  新共享边: (1, 3)")
    
    print("\n" + "=" * 50)
    print("  测试链式翻转恢复约束边...")
    print("=" * 50)
    
    points2 = np.array([
        [0, 0],
        [2, 0],
        [3, 1],
        [3, 3],
        [1, 3],
        [0, 2],
    ])
    
    triangles2 = [
        (0, 1, 5),
        (1, 2, 5),
        (2, 3, 5),
        (3, 4, 5),
        (4, 0, 5),
    ]
    
    print("\n初始三角形:")
    for i, tri in enumerate(triangles2):
        print(f"  三角形 {i}: {tri}")
    
    print("\n尝试恢复约束边 (0, 3)...")
    flipper2 = EdgeFlipper(points2, triangles2)
    success = flipper2.flip_chain_for_constraint(0, 3)
    print(f"  成功: {success}")
    
    if success:
        print("\n翻转后三角形:")
        for i, tri in enumerate(flipper2.triangles):
            print(f"  三角形 {i}: {tri}")
        print("  约束边 (0, 3) 已恢复!")
    else:
        print("  约束边恢复失败（可能需要更多翻转或网格不支持）")
    
    print("\n✅ 演示完成")


if __name__ == '__main__':
    demo()
