"""
NumPy向量化几何计算

不需要Numba，纯NumPy就能用
比Python循环快 10~30 倍
"""

import numpy as np


def assign_property_fast(cell_centers, geometry_func_vectorized, prop_array, value):
    """
    快速赋物性（向量化版本）
    
    原来的方式（慢）：
        for i in range(n_cells):
            if geometry_func(x, y, z):
                prop[i] = value
    
    现在：一次性判断所有格子
    """
    mask = geometry_func_vectorized(cell_centers)
    prop_array[mask] = value
    return np.sum(mask)


def sphere_contains_vectorized(centers, sphere_center, radius):
    """
    向量化球体包含判断
    
    参数:
        centers: (N, 3) 所有格子中心
        sphere_center: (3,) 球心
        radius: 半径
    
    返回:
        mask: (N,) 布尔数组
    """
    diff = centers - np.array(sphere_center)
    dist_sq = np.sum(diff ** 2, axis=1)
    return dist_sq <= radius ** 2


def ellipsoid_contains_vectorized(centers, ellipsoid_center, semi_axes):
    """向量化椭球体包含判断"""
    cx, cy, cz = ellipsoid_center
    a, b, c = semi_axes
    
    x = (centers[:, 0] - cx) / a
    y = (centers[:, 1] - cy) / b
    z = (centers[:, 2] - cz) / c
    
    return x**2 + y**2 + z**2 <= 1.0


def block_contains_vectorized(centers, bounds):
    """向量化长方体包含判断"""
    x1, x2, y1, y2, z1, z2 = bounds
    return ((centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
            (centers[:, 1] >= y1) & (centers[:, 1] <= y2) &
            (centers[:, 2] >= z1) & (centers[:, 2] <= z2))


def batch_triangle_quality(points, triangles):
    """
    批量计算三角形质量（向量化）
    
    比逐个计算快 20 倍
    """
    p0 = points[triangles[:, 0]]  # (N, 2)
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)
    
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = np.maximum(area_sq, 0)
    area = np.sqrt(area_sq)
    
    max_edge = np.maximum(np.maximum(a, b), c)
    quality = 4 * np.sqrt(3) * area / (max_edge ** 2 + 1e-30)
    
    return np.clip(quality, 0, 1)


def batch_tet_quality(points, tets):
    """
    批量计算四面体质量（向量化）
    """
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    
    v01 = p1 - p0
    v02 = p2 - p0
    v03 = p3 - p0
    
    # 体积
    cross = np.cross(v02, v03)
    vol = np.abs(np.sum(v01 * cross, axis=1)) / 6.0
    
    # 6条边长
    edges = np.stack([
        np.linalg.norm(p1 - p0, axis=1),
        np.linalg.norm(p2 - p0, axis=1),
        np.linalg.norm(p3 - p0, axis=1),
        np.linalg.norm(p2 - p1, axis=1),
        np.linalg.norm(p3 - p1, axis=1),
        np.linalg.norm(p3 - p2, axis=1),
    ], axis=1)
    
    max_edge = np.max(edges, axis=1)
    quality = 6.0 * np.sqrt(2) * vol / (max_edge ** 3 + 1e-30)
    
    return np.clip(quality, 0, 1)