"""
网格质量评估

好的网格 = 三角形尽可能接近等边三角形
坏的网格 = 三角形很瘦长（像一根针）

         好                 坏
        ╱╲               │
       ╱  ╲              │╲
      ╱    ╲             │  ╲
     ╱______╲            │____╲
   接近等边                瘦长，计算不准
   质量≈1.0               质量≈0.01
"""

import numpy as np


def triangle_quality(p0, p1, p2):
    """
    计算一个三角形的质量
    
    使用"半径比"指标：
        quality = 2 * 内切圆半径 / 外接圆半径
        
        等边三角形 → quality = 1.0（最好）
        退化三角形 → quality → 0.0（最差）
    
    参数:
        p0, p1, p2: 三个顶点坐标 (x, z)
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    
    # 三条边长度
    a = np.linalg.norm(p1 - p2)  # 对面p0的边
    b = np.linalg.norm(p0 - p2)  # 对面p1的边
    c = np.linalg.norm(p0 - p1)  # 对面p2的边
    
    # 半周长
    s = (a + b + c) / 2
    
    # 面积（海伦公式）
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return 0.0  # 退化三角形
    area = np.sqrt(area_sq)
    
    # 内切圆半径
    r_in = area / s
    
    # 外接圆半径
    r_out = (a * b * c) / (4 * area)
    
    if r_out == 0:
        return 0.0
    
    quality = 2 * r_in / r_out
    return quality


def min_angle(p0, p1, p2):
    """
    计算三角形的最小角（度）
    
    好的三角形：最小角 > 20°
    可接受：最小角 > 10°
    很差：最小角 < 5°
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    
    # 三条边的向量
    v01 = p1 - p0
    v02 = p2 - p0
    v12 = p2 - p1
    
    # 三个角
    def angle_between(va, vb):
        cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-30)
        cos_a = np.clip(cos_a, -1, 1)
        return np.degrees(np.arccos(cos_a))
    
    a0 = angle_between(v01, v02)
    a1 = angle_between(-v01, v12)
    a2 = angle_between(-v02, -v12)
    
    return min(a0, a1, a2)


def evaluate_mesh_quality(triangulation):
    """
    评估整个三角网格的质量
    
    参数:
        triangulation: DelaunayTriangulation2D 对象
    """
    qualities = []
    min_angles = []
    areas = []
    
    for i, tri in enumerate(triangulation.triangles):
        p0 = triangulation.points[tri[0]]
        p1 = triangulation.points[tri[1]]
        p2 = triangulation.points[tri[2]]
        
        q = triangle_quality(p0, p1, p2)
        a = min_angle(p0, p1, p2)
        area = triangulation.triangle_area(i)
        
        qualities.append(q)
        min_angles.append(a)
        areas.append(area)
    
    qualities = np.array(qualities)
    min_angles = np.array(min_angles)
    areas = np.array(areas)
    
    # 统计
    info = f"""
    ╔══════════════════════════════════════════╗
    ║           网格质量报告                     ║
    ╠══════════════════════════════════════════╣
    ║ 三角形总数: {len(qualities)}
    ║
    ║ 【质量指标】(0=最差, 1=等边三角形)
    ║   最小值: {qualities.min():.4f}
    ║   最大值: {qualities.max():.4f}
    ║   平均值: {qualities.mean():.4f}
    ║   质量>0.5的比例: {(qualities > 0.5).sum() / len(qualities) * 100:.1f}%
    ║   质量<0.1的比例: {(qualities < 0.1).sum() / len(qualities) * 100:.1f}% {'⚠️ 需关注' if (qualities < 0.1).any() else '✅ 良好'}
    ║
    ║ 【最小角度】(等边=60°, >20°为好)
    ║   最小值: {min_angles.min():.1f}°
    ║   平均值: {min_angles.mean():.1f}°
    ║   角度<10°的比例: {(min_angles < 10).sum() / len(min_angles) * 100:.1f}% {'⚠️ 有瘦长三角形' if (min_angles < 10).any() else '✅ 良好'}
    ║
    ║ 【面积】
    ║   最小: {areas.min():.1f} m²
    ║   最大: {areas.max():.1f} m²
    ║   比值: {areas.max() / max(areas.min(), 1e-10):.0f} 倍
    ╚══════════════════════════════════════════╝
    """
    print(info)
    
    return {
        'qualities': qualities,
        'min_angles': min_angles,
        'areas': areas
    }