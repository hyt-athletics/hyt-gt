"""
Numba加速几何计算

加速最常用的两个几何操作：
  1. 点在多边形内判断（射线法）
  2. 批量距离计算
"""

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    
    @njit(cache=True)
    def _point_in_polygon_single(px, pz, poly_x, poly_z, n):
        """
        射线法判断一个点是否在多边形内
        
        Numba编译后极快
        """
        inside = False
        j = n - 1
        for i in range(n):
            xi = poly_x[i]
            zi = poly_z[i]
            xj = poly_x[j]
            zj = poly_z[j]
            
            if ((zi > pz) != (zj > pz)):
                slope = (xj - xi) * (pz - zi) / (zj - zi + 1e-30) + xi
                if px < slope:
                    inside = not inside
            j = i
        
        return inside
    
    
    @njit(parallel=True, cache=True)
    def points_in_polygon_numba(points_x, points_z, poly_x, poly_z):
        """
        批量判断多个点是否在多边形内
        
        参数:
            points_x: (N,) 测试点x坐标
            points_z: (N,) 测试点z坐标
            poly_x: (M,) 多边形顶点x坐标
            poly_z: (M,) 多边形顶点z坐标
        
        返回:
            inside: (N,) 布尔数组
        """
        n_points = len(points_x)
        n_poly = len(poly_x)
        result = np.zeros(n_points, dtype=np.bool_)
        
        for i in prange(n_points):
            result[i] = _point_in_polygon_single(
                points_x[i], points_z[i], poly_x, poly_z, n_poly
            )
        
        return result
    
    
    @njit(parallel=True, cache=True)
    def points_in_3d_body_numba(points_x, points_y, points_z,
                                 poly_x, poly_z, y_min, y_max):
        """
        批量判断3D点是否在柱状体内
        （多边形截面 + Y方向范围）
        """
        n = len(points_x)
        n_poly = len(poly_x)
        result = np.zeros(n, dtype=np.bool_)
        
        for i in prange(n):
            if points_y[i] < y_min or points_y[i] > y_max:
                continue
            result[i] = _point_in_polygon_single(
                points_x[i], points_z[i], poly_x, poly_z, n_poly
            )
        
        return result
    
    
    @njit(parallel=True, cache=True)
    def batch_distances_numba(points, target):
        """
        计算所有点到一个目标点的距离
        
        参数:
            points: (N, 3) 
            target: (3,) 目标点
        
        返回:
            distances: (N,)
        """
        n = len(points)
        dist = np.zeros(n)
        tx, ty, tz = target[0], target[1], target[2]
        
        for i in prange(n):
            dx = points[i, 0] - tx
            dy = points[i, 1] - ty
            dz = points[i, 2] - tz
            dist[i] = (dx * dx + dy * dy + dz * dz) ** 0.5
        
        return dist


# NumPy回退版本（不需要Numba也能用）

def points_in_polygon_numpy(points_x, points_z, poly_x, poly_z):
    """NumPy向量化版本的点在多边形内判断"""
    n_points = len(points_x)
    n_poly = len(poly_x)
    inside = np.zeros(n_points, dtype=bool)
    
    j = n_poly - 1
    for i in range(n_poly):
        xi, zi = poly_x[i], poly_z[i]
        xj, zj = poly_x[j], poly_z[j]
        
        cond1 = (zi > points_z) != (zj > points_z)
        slope = (xj - xi) * (points_z - zi) / (zj - zi + 1e-30) + xi
        cond2 = points_x < slope
        
        mask = cond1 & cond2
        inside[mask] = ~inside[mask]
        
        j = i
    
    return inside


def fast_points_in_polygon(points_x, points_z, poly_x, poly_z):
    """自动选择最快的实现"""
    if HAS_NUMBA:
        return points_in_polygon_numba(
            np.ascontiguousarray(points_x, dtype=np.float64),
            np.ascontiguousarray(points_z, dtype=np.float64),
            np.ascontiguousarray(poly_x, dtype=np.float64),
            np.ascontiguousarray(poly_z, dtype=np.float64),
        )
    else:
        return points_in_polygon_numpy(points_x, points_z, poly_x, poly_z)