"""
自研3D Delaunay四面体剖分

算法：Bowyer-Watson 增量插入法（3D版本）

和2D版的区别：
  2D: 三角形 → 外接圆 → 边界是线段
  3D: 四面体 → 外接球 → 边界是三角面

性能说明：
  纯Python实现，适合 500~3000 个点
  超过3000个点会比较慢（几十秒到几分钟）
  如果需要大规模网格，可以切换到scipy后端
"""

import numpy as np
import time


class DelaunayTriangulation3D:
    """
    自研3D Delaunay四面体剖分
    
    使用方法:
        dt3d = DelaunayTriangulation3D()
        dt3d.triangulate(points)
        
    结果:
        dt3d.points       → 点坐标数组 (N, 3)
        dt3d.tetrahedra   → 四面体列表 [(i,j,k,l), ...]
    """
    
    def __init__(self):
        self.points = None          # 所有点的坐标
        self.tetrahedra = []        # 四面体列表，每个是4个点索引的元组
        self._super_indices = []    # 超级四面体的4个点索引
    
    def triangulate(self, points):
        """
        对一组3D点进行Delaunay四面体剖分
        
        参数:
            points: 点坐标 [(x,y,z), ...] 或 numpy数组 (N, 3)
        
        返回:
            self
        """
        pts = np.array(points, dtype=float)
        n_original = len(pts)
        
        print(f"  开始3D四面体剖分，共 {n_original} 个点...")
        t_start = time.time()
        
        # 添加微小扰动，防止共面/共球退化
        pts = pts + np.random.randn(*pts.shape) * 1e-10 * np.ptp(pts)
        
        # ============================================
        # 第1步：创建超级四面体
        # ============================================
        super_pts = self._make_super_tetrahedron(pts)
        all_pts = np.vstack([pts, super_pts])
        self.points = all_pts
        
        si = n_original  # 超级四面体顶点的起始索引
        self._super_indices = [si, si + 1, si + 2, si + 3]
        
        # 确保超级四面体正向
        super_tet = self._ensure_orientation(si, si + 1, si + 2, si + 3)
        self.tetrahedra = [super_tet]
        
        # ============================================
        # 第2步：逐点插入
        # ============================================
        report_interval = max(n_original // 10, 1)
        
        for idx in range(n_original):
            if idx % report_interval == 0 and idx > 0:
                elapsed = time.time() - t_start
                print(f"    已插入 {idx}/{n_original} 个点... "
                      f"({elapsed:.1f}秒, {len(self.tetrahedra)}个四面体)")
            self._insert_point(idx)
        
        # ============================================
        # 第3步：删除超级四面体相关的四面体
        # ============================================
        self.tetrahedra = [
            tet for tet in self.tetrahedra
            if not any(v in self._super_indices for v in tet)
        ]
        
        # 只保留原始点
        self.points = pts
        
        elapsed = time.time() - t_start
        print(f"  3D四面体剖分完成：{len(self.tetrahedra)} 个四面体 ({elapsed:.1f}秒)")
        
        return self
    
    def _make_super_tetrahedron(self, pts):
        """
        创建超级四面体：包围所有点的巨大四面体
        
              p3
             ╱│╲
            ╱ │ ╲           所有输入点
           ╱  │  ╲          都在这个超级四面体内部
          ╱   │   ╲
         ╱    │    ╲
        p0────│─────p2
         ╲    │    ╱
          ╲   │   ╱
           ╲  │  ╱
            ╲ │ ╱
              p1
        """
        # 找到点集的包围盒
        mins = pts.min(axis=0) - 1
        maxs = pts.max(axis=0) + 1
        
        # 中心和最大跨度
        center = (mins + maxs) / 2
        d = max(maxs - mins) * 10  # 放大10倍，确保足够大
        
        # 超级四面体的4个顶点
        p0 = center + np.array([-d, -d, -d])
        p1 = center + np.array([3 * d, -d, -d])
        p2 = center + np.array([-d, 3 * d, -d])
        p3 = center + np.array([-d, -d, 3 * d])
        
        return np.array([p0, p1, p2, p3])
    
    def _insert_point(self, point_idx):
        """
        插入一个点（3D Bowyer-Watson核心）
        
        和2D一模一样的逻辑：
        1. 找到外接球包含新点的四面体（坏四面体）
        2. 找到坏四面体的边界面
        3. 删除坏四面体
        4. 用新点和边界面构成新四面体
        """
        point = self.points[point_idx]
        
        # ------ 第1步：找坏四面体 ------
        bad_tets = []
        for tet in self.tetrahedra:
            if self._point_in_circumsphere(point, tet):
                bad_tets.append(tet)
        
        if len(bad_tets) == 0:
            return  # 点可能在凸包外（不影响结果）
        
        # ------ 第2步：找边界面 ------
        # 统计每个面出现了几次
        # 出现1次 = 边界面（一边是坏四面体，另一边是好四面体或外部）
        # 出现2次 = 内部面（两边都是坏四面体）
        face_count = {}
        face_ordered = {}
        
        for tet in bad_tets:
            faces = self._get_faces(tet)
            for face in faces:
                key = frozenset(face)
                if key in face_count:
                    face_count[key] += 1
                else:
                    face_count[key] = 1
                    face_ordered[key] = face
        
        boundary_faces = [
            face_ordered[key]
            for key, count in face_count.items()
            if count == 1
        ]
        
        # ------ 第3步：删除坏四面体 ------
        for tet in bad_tets:
            self.tetrahedra.remove(tet)
        
        # ------ 第4步：用新点和边界面构成新四面体 ------
        for face in boundary_faces:
            new_tet = self._ensure_orientation(
                face[0], face[1], face[2], point_idx
            )
            self.tetrahedra.append(new_tet)
    
    def _get_faces(self, tet):
        """
        获取四面体的4个面
        
        四面体 (v0, v1, v2, v3) 有4个三角形面：
          面0: (v0, v1, v2)  — v3对面
          面1: (v0, v1, v3)  — v2对面
          面2: (v0, v2, v3)  — v1对面
          面3: (v1, v2, v3)  — v0对面
        """
        v0, v1, v2, v3 = tet
        return [
            (v0, v1, v2),
            (v0, v1, v3),
            (v0, v2, v3),
            (v1, v2, v3)
        ]
    
    def _point_in_circumsphere(self, point, tet):
        """
        判断点是否在四面体的外接球内
        
        外接球 = 穿过四面体4个顶点的球
        
        数学方法：4×4行列式判断
        
        ┌                                                    ┐
        │ ax-px  ay-py  az-pz  (ax-px)²+(ay-py)²+(az-pz)²  │
        │ bx-px  by-py  bz-pz  (bx-px)²+(by-py)²+(bz-pz)²  │ > 0
        │ cx-px  cy-py  cz-pz  (cx-px)²+(cy-py)²+(cz-pz)²  │
        │ dx-px  dy-py  dz-pz  (dx-px)²+(dy-py)²+(dz-pz)²  │
        └                                                    ┘
        
        （假设四面体正向排列）
        """
        a = self.points[tet[0]]
        b = self.points[tet[1]]
        c = self.points[tet[2]]
        d = self.points[tet[3]]
        e = point
        
        # 检查四面体的方向（正向/反向）
        orient_mat = np.array([b - a, c - a, d - a])
        orient = np.linalg.det(orient_mat)
        
        if abs(orient) < 1e-30:
            return False  # 退化四面体（4个点共面）
        
        # 构建4×4行列式
        rows = []
        for p in [a, b, c, d]:
            dx = p[0] - e[0]
            dy = p[1] - e[1]
            dz = p[2] - e[2]
            rows.append([dx, dy, dz, dx * dx + dy * dy + dz * dz])
        
        mat = np.array(rows)
        det = np.linalg.det(mat)
        
        # 根据方向决定符号
        if orient > 0:
            return det > 1e-15
        else:
            return det < -1e-15
    
    def _ensure_orientation(self, v0, v1, v2, v3):
        """
        确保四面体是正向排列
        
        正向 = 从v3看去，v0→v1→v2是逆时针
        
        相当于 det([v1-v0, v2-v0, v3-v0]) > 0
        """
        a = self.points[v0]
        b = self.points[v1]
        c = self.points[v2]
        d = self.points[v3]
        
        det = np.dot(b - a, np.cross(c - a, d - a))
        
        if det < 0:
            return (v0, v2, v1, v3)  # 交换v1和v2来翻转方向
        return (v0, v1, v2, v3)
    
    # ==========================================================
    #  查询方法
    # ==========================================================
    
    def tet_center(self, tet_idx):
        """获取四面体重心"""
        tet = self.tetrahedra[tet_idx]
        return self.points[list(tet)].mean(axis=0)
    
    def tet_volume(self, tet_idx):
        """
        计算四面体体积
        
        体积 = |det([v1-v0, v2-v0, v3-v0])| / 6
        """
        tet = self.tetrahedra[tet_idx]
        a = self.points[tet[0]]
        b = self.points[tet[1]]
        c = self.points[tet[2]]
        d = self.points[tet[3]]
        
        vol = abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0
        return vol
    
    def get_edges(self):
        """获取所有不重复的边"""
        edge_set = set()
        for tet in self.tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = tuple(sorted([tet[i], tet[j]]))
                    edge_set.add(edge)
        return list(edge_set)
    
    def get_surface_faces(self):
        """
        获取网格的外表面三角形
        
        原理：只出现在一个四面体中的面 = 表面面
        """
        face_count = {}
        face_ordered = {}
        
        for tet in self.tetrahedra:
            for face in self._get_faces(tet):
                key = frozenset(face)
                face_count[key] = face_count.get(key, 0) + 1
                if key not in face_ordered:
                    face_ordered[key] = face
        
        surface = [
            face_ordered[key]
            for key, count in face_count.items()
            if count == 1
        ]
        return surface
    
    def get_all_centers(self):
        """获取所有四面体的重心坐标"""
        centers = []
        for i in range(len(self.tetrahedra)):
            centers.append(self.tet_center(i))
        return np.array(centers)
    
    def get_all_volumes(self):
        """获取所有四面体的体积"""
        volumes = []
        for i in range(len(self.tetrahedra)):
            volumes.append(self.tet_volume(i))
        return np.array(volumes)
    
    def get_info(self):
        """打印信息"""
        volumes = self.get_all_volumes()
        
        if len(volumes) > 0:
            vol_min = volumes.min()
            vol_max = volumes.max()
            vol_sum = volumes.sum()
            vol_mean = volumes.mean()
        else:
            vol_min = vol_max = vol_sum = vol_mean = 0.0
        
        info = f"""
        ╔══════════════════════════════════════════╗
        ║       3D Delaunay 四面体剖分信息          ║
        ╠══════════════════════════════════════════╣
        ║ 点数: {len(self.points)}
        ║ 四面体数: {len(self.tetrahedra)}
        ║ 边数: {len(self.get_edges())}
        ║ 表面三角形数: {len(self.get_surface_faces())}
        ║
        ║ 体积统计:
        ║   最小: {vol_min:.2f} m³
        ║   最大: {vol_max:.2f} m³
        ║   总体积: {vol_sum:.0f} m³
        ║   平均: {vol_mean:.2f} m³
        ╚══════════════════════════════════════════╝
        """
        print(info)


class DelaunayTriangulation3D_Fast:
    """
    使用scipy的快速3D Delaunay（生产用）
    
    接口和自研版完全一致，可以互相替换
    
    点数多的时候用这个（>3000个点）
    """
    
    def __init__(self):
        self.points = None
        self.tetrahedra = []
    
    def triangulate(self, points):
        from scipy.spatial import Delaunay
        
        pts = np.array(points, dtype=float)
        print(f"  使用scipy快速3D Delaunay, {len(pts)} 个点...")
        
        t_start = time.time()
        tri = Delaunay(pts)
        elapsed = time.time() - t_start
        
        self.points = pts
        self.tetrahedra = [tuple(s) for s in tri.simplices]
        
        print(f"  完成：{len(self.tetrahedra)} 个四面体 ({elapsed:.2f}秒)")
        return self
    
    # 复用和自研版一样的查询方法
    def tet_center(self, tet_idx):
        tet = self.tetrahedra[tet_idx]
        return self.points[list(tet)].mean(axis=0)
    
    def tet_volume(self, tet_idx):
        tet = self.tetrahedra[tet_idx]
        a, b, c, d = [self.points[i] for i in tet]
        return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0
    
    def get_surface_faces(self):
        face_count = {}
        face_ordered = {}
        for tet in self.tetrahedra:
            v = tet
            faces = [
                (v[0], v[1], v[2]), (v[0], v[1], v[3]),
                (v[0], v[2], v[3]), (v[1], v[2], v[3])
            ]
            for face in faces:
                key = frozenset(face)
                face_count[key] = face_count.get(key, 0) + 1
                if key not in face_ordered:
                    face_ordered[key] = face
        return [face_ordered[k] for k, c in face_count.items() if c == 1]
    
    def get_all_centers(self):
        return np.array([self.tet_center(i) for i in range(len(self.tetrahedra))])
    
    def get_all_volumes(self):
        return np.array([self.tet_volume(i) for i in range(len(self.tetrahedra))])
    
    def get_edges(self):
        edge_set = set()
        for tet in self.tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    edge_set.add(tuple(sorted([tet[i], tet[j]])))
        return list(edge_set)
    
    def get_info(self):
        volumes = self.get_all_volumes()
        
        if len(volumes) > 0:
            vol_min = volumes.min()
            vol_max = volumes.max()
            vol_sum = volumes.sum()
        else:
            vol_min = vol_max = vol_sum = 0.0
        
        print(f"""
        ╔══════════════════════════════════════════╗
        ║     3D Delaunay (scipy快速版) 信息        ║
        ╠══════════════════════════════════════════╣
        ║ 点数: {len(self.points)}
        ║ 四面体数: {len(self.tetrahedra)}
        ║ 体积范围: {vol_min:.2f} ~ {vol_max:.2f} m³
        ║ 总体积: {vol_sum:.0f} m³
        ╚══════════════════════════════════════════╝
        """)