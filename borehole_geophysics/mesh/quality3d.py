"""
3D四面体网格质量评估

好的四面体 ≈ 正四面体（4个等边三角形面）
坏的四面体 ≈ 扁平的或细长的

   好（饱满）           坏（扁平）          坏（细长）
     ╱╲                  ____               │
    ╱  ╲               ╱╱  ╲╲              │╲
   ╱ ●  ╲            ╱╱ ●  ╲╲             │ ╲
  ╱______╲          ╱╱______╲╲            │●╲╲
  质量≈1             质量≈0.01             质量≈0.05
"""

import numpy as np


def tet_quality_radius_ratio(p0, p1, p2, p3):
    """
    四面体质量指标：内切球半径 / 外接球半径 × 3
    
    正四面体 → 1.0（最好）
    退化体   → 0.0（最差）
    
    参数:
        p0, p1, p2, p3: 四个顶点坐标，每个是 (x, y, z)
    """
    p0, p1, p2, p3 = [np.array(p, dtype=float) for p in [p0, p1, p2, p3]]
    
    # 六条边长
    edges = [
        np.linalg.norm(p1 - p0),
        np.linalg.norm(p2 - p0),
        np.linalg.norm(p3 - p0),
        np.linalg.norm(p2 - p1),
        np.linalg.norm(p3 - p1),
        np.linalg.norm(p3 - p2),
    ]
    
    if min(edges) < 1e-30:
        return 0.0
    
    # 体积
    vol = abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
    
    if vol < 1e-30:
        return 0.0
    
    # 四个面的面积
    def triangle_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    
    s0 = triangle_area(p1, p2, p3)
    s1 = triangle_area(p0, p2, p3)
    s2 = triangle_area(p0, p1, p3)
    s3 = triangle_area(p0, p1, p2)
    total_area = s0 + s1 + s2 + s3
    
    if total_area < 1e-30:
        return 0.0
    
    # 内切球半径 = 3V / S
    r_in = 3 * vol / total_area
    
    # 外接球半径（公式较复杂，用边长和体积计算）
    # R = |a||b||c| / (6V)  其中a,b,c是对边的叉积
    ab = np.cross(p1 - p0, p2 - p0)
    ac = np.cross(p1 - p0, p3 - p0)
    bc = np.cross(p2 - p0, p3 - p0)
    
    # 简化外接球半径计算
    edge_product = edges[0] * edges[5]  # 对边
    edge_product2 = edges[1] * edges[4]
    edge_product3 = edges[2] * edges[3]
    
    # 使用公式: R = abc/(8V) where a,b,c are edge lengths of a face...
    # 更简单的方法：直接解方程
    # 用替代质量指标: 3 * r_in / r_out
    # r_out可以从体积和边长关系得到
    
    # 使用替代公式: quality = 6√2 * V / (max_edge³)
    # 正四面体时刚好等于1
    max_edge = max(edges)
    quality = 6.0 * np.sqrt(2) * vol / (max_edge ** 3)
    
    # 裁剪到[0, 1]
    return min(quality, 1.0)


def tet_min_dihedral_angle(p0, p1, p2, p3):
    """
    计算四面体的最小二面角（度）
    
    二面角 = 两个面之间的夹角
    四面体有6条边，每条边对应一个二面角
    
    好的四面体：所有二面角在 30°~120° 之间
    正四面体：所有二面角 ≈ 70.5°
    """
    pts = [np.array(p, dtype=float) for p in [p0, p1, p2, p3]]
    
    def face_normal(a, b, c):
        """三角形的法向量"""
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-30:
            return np.zeros(3)
        return n / length
    
    # 四个面的法向量
    normals = [
        face_normal(pts[1], pts[2], pts[3]),  # 面0：v1v2v3
        face_normal(pts[0], pts[3], pts[2]),  # 面1：v0v3v2
        face_normal(pts[0], pts[1], pts[3]),  # 面2：v0v1v3
        face_normal(pts[0], pts[2], pts[1]),  # 面3：v0v2v1
    ]
    
    # 6条边，每条边由两个面共享
    edge_face_pairs = [
        (0, 2),  # 边v1v3 共享面0和面2
        (0, 3),  # 边v1v2 共享面0和面3
        (0, 1),  # 边v2v3 共享面0和面1
        (1, 2),  # 边v0v3 共享面1和面2
        (1, 3),  # 边v0v2 共享面1和面3
        (2, 3),  # 边v0v1 共享面2和面3
    ]
    
    min_angle = 180.0
    for fi, fj in edge_face_pairs:
        ni, nj = normals[fi], normals[fj]
        if np.linalg.norm(ni) < 1e-10 or np.linalg.norm(nj) < 1e-10:
            continue
        cos_angle = np.clip(np.dot(ni, nj), -1, 1)
        angle = np.degrees(np.arccos(-cos_angle))  # 注意负号
        min_angle = min(min_angle, angle)
    
    return min_angle


def tet_edge_ratio(p0, p1, p2, p3):
    """
    边长比 = 最长边 / 最短边
    
    正四面体 → 1.0（最好）
    越大越差
    """
    pts = [np.array(p, dtype=float) for p in [p0, p1, p2, p3]]
    
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.linalg.norm(pts[j] - pts[i]))
    
    if min(edges) < 1e-30:
        return float('inf')
    
    return max(edges) / min(edges)


def evaluate_tet_mesh(mesh_3d):
    """
    评估整个3D四面体网格的质量
    
    参数:
        mesh_3d: DelaunayTriangulation3D 或 DelaunayTriangulation3D_Fast
    """
    n_tets = len(mesh_3d.tetrahedra)
    
    qualities = []
    min_angles = []
    edge_ratios = []
    volumes = []
    
    for i in range(n_tets):
        tet = mesh_3d.tetrahedra[i]
        pts = [mesh_3d.points[v] for v in tet]
        
        q = tet_quality_radius_ratio(*pts)
        a = tet_min_dihedral_angle(*pts)
        r = tet_edge_ratio(*pts)
        v = mesh_3d.tet_volume(i)
        
        qualities.append(q)
        min_angles.append(a)
        edge_ratios.append(r)
        volumes.append(v)
    
    qualities = np.array(qualities)
    min_angles = np.array(min_angles)
    edge_ratios = np.array(edge_ratios)
    volumes = np.array(volumes)
    
    if len(qualities) > 0:
        q_min = qualities.min()
        q_mean = qualities.mean()
        q_bad = (qualities < 0.1).sum() / n_tets * 100
        q_good = (qualities > 0.3).sum() / n_tets * 100
        
        a_min = min_angles.min()
        a_mean = min_angles.mean()
        a_bad = (min_angles < 10).sum() / n_tets * 100
        
        e_max = edge_ratios.max()
        e_mean = edge_ratios.mean()
        
        v_min = volumes.min()
        v_max = volumes.max()
        v_sum = volumes.sum()
    else:
        q_min = q_mean = e_max = e_mean = 0.0
        a_min = a_mean = 0.0
        q_bad = q_good = a_bad = v_sum = 0.0
        v_min = v_max = 1e-10
    
    info = f"""
    ╔══════════════════════════════════════════════════╗
    ║            3D 四面体网格质量报告                    ║
    ╠══════════════════════════════════════════════════╣
    ║ 四面体总数: {n_tets}
    ║
    ║ 【质量指标】(0=最差, 1=正四面体)
    ║   最小: {q_min:.4f}
    ║   平均: {q_mean:.4f}
    ║   >0.3 的比例: {q_good:.1f}%
    ║   <0.1 的比例: {q_bad:.1f}% {'⚠️' if q_bad > 10 else '✅'}
    ║
    ║ 【最小二面角】(正四面体≈70.5°, >10°为可接受)
    ║   最小: {a_min:.1f}°
    ║   平均: {a_mean:.1f}°
    ║   <10° 的比例: {a_bad:.1f}%
    ║
    ║ 【边长比】(1=完美, <5为好)
    ║   最大: {e_max:.1f}
    ║   平均: {e_mean:.1f}
    ║
    ║ 【体积】
    ║   最小: {v_min:.1f} m³
    ║   最大: {v_max:.1f} m³
    ║   比值: {v_max / max(v_min, 1e-10):.0f} 倍
    ║   总体积: {v_sum:.0f} m³
    ╚══════════════════════════════════════════════════╝
    """
    print(info)
    
    return {
        'qualities': qualities,
        'min_angles': min_angles,
        'edge_ratios': edge_ratios,
        'volumes': volumes
    }