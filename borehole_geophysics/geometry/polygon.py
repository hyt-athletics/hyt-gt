"""
不规则多边形几何体

用多边形描述截面上的不规则矿体边界
比 sphere/ellipsoid 更灵活：可以画任意形状

     ●──────●
    ╱        ╲
   ●          ●       ← 用多边形逼近任意形状的矿体
    ╲    ╱───●
     ●──●
"""

import numpy as np


def point_in_polygon_2d(px, pz, polygon):
    """
    判断点(px, pz)是否在2D多边形内
    
    算法：射线法（Ray Casting）
    
    从点向右画一条射线，数它穿过多边形边界几次
    奇数次 → 在里面
    偶数次 → 在外面
    
        ────●────────
            │╲         射线 →→→→→→→→→→
        ────│──●──                穿过1次(奇数) → 在里面
            │   ╲
        ────│────●
            点
    
    参数:
        px, pz: 测试点坐标
        polygon: [(x1,z1), (x2,z2), ...] 多边形顶点（按顺序，自动闭合）
    """
    n = len(polygon)
    inside = False
    
    j = n - 1  # 上一个顶点的索引
    for i in range(n):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        
        # 检查射线是否穿过这条边
        if ((zi > pz) != (zj > pz)) and \
           (px < (xj - xi) * (pz - zi) / (zj - zi + 1e-30) + xi):
            inside = not inside
        
        j = i
    
    return inside


def make_polygon_body(polygon_xz, y_range):
    """
    创建一个不规则多边形柱体
    
    在XZ截面上是多边形形状，沿Y方向延伸
    
    参数:
        polygon_xz: [(x1,z1), (x2,z2), ...] 截面上的多边形顶点
        y_range: (y_min, y_max) Y方向延伸范围
    
    返回:
        geometry_func: 函数(x,y,z) → True/False
    
    用法:
        # 定义一个不规则矿体
        ore = make_polygon_body(
            polygon_xz=[
                (550, 350), (620, 370), (650, 420),
                (630, 480), (560, 470), (530, 400)
            ],
            y_range=(300, 700)
        )
        mesh.assign_property(ore, 'density', 0.5)
    """
    y_min, y_max = y_range
    
    def geometry_func(x, y, z):
        if not (y_min <= y <= y_max):
            return False
        return point_in_polygon_2d(x, z, polygon_xz)
    
    return geometry_func


def make_polygon_body_variable_y(polygon_xz_func, y_range, y_steps=10):
    """
    创建截面形状随Y变化的异常体
    
    参数:
        polygon_xz_func: 函数，输入y值，返回该y位置处的多边形顶点列表
        y_range: (y_min, y_max)
    """
    y_min, y_max = y_range
    
    def geometry_func(x, y, z):
        if not (y_min <= y <= y_max):
            return False
        polygon = polygon_xz_func(y)
        return point_in_polygon_2d(x, z, polygon)
    
    return geometry_func


def make_layered_model(layers):
    """
    创建水平层状模型
    
    参数:
        layers: [(z_top, z_bottom, density), ...]
                从上到下排列
    
    用法:
        layers = [
            (0,   100, 2.0),    # 第一层：0~100m深，密度2.0
            (100, 300, 2.5),    # 第二层
            (300, 600, 2.7),    # 第三层
            (600, 1000, 2.9),   # 第四层
        ]
    """
    def make_layer_func(z_top, z_bottom):
        def func(x, y, z):
            return z_top <= z <= z_bottom
        return func
    
    return [(make_layer_func(zt, zb), density) for zt, zb, density in layers]