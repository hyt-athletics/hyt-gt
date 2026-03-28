"""
几何体定义模块

定义各种形状的几何体（球、椭球、长方体等）
用于给网格的特定区域赋物性值
"""

import numpy as np


def make_sphere(center, radius):
    """
    创建球形几何体判断函数
    
    参数:
        center: (x, y, z) 球心坐标
        radius: 半径
    
    返回:
        一个函数，输入(x, y, z)，返回True(在球内)或False(不在)
    
    举例:
        sphere = make_sphere(center=(500, 500, 400), radius=80)
        if sphere(510, 500, 400):  # 在球内
            print("在球内")
    """
    cx, cy, cz = center
    
    def sphere_func(x, y, z):
        return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2
    
    return sphere_func


def make_ellipsoid(center, semi_axes):
    """
    创建椭球体几何体判断函数
    
    参数:
        center: (x, y, z) 椭球中心坐标
        semi_axes: (a, b, c) 三个半轴长度
    
    返回:
        一个函数，输入(x, y, z)，返回True(在椭球内)或False(不在)
    
    举例:
        ellipsoid = make_ellipsoid(center=(500, 300, 300), semi_axes=(120, 60, 40))
        if ellipsoid(500, 300, 300):  # 在椭球内
            print("在椭球内")
    """
    cx, cy, cz = center
    a, b, c = semi_axes
    
    def ellipsoid_func(x, y, z):
        return ((x - cx)/a)**2 + ((y - cy)/b)**2 + ((z - cz)/c)**2 <= 1.0
    
    return ellipsoid_func


def make_block(bounds):
    """
    创建长方体几何体判断函数
    
    参数:
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
                六个值定义长方体的边界
    
    返回:
        一个函数，输入(x, y, z)，返回True(在长方体内)或False(不在)
    
    举例:
        block = make_block(bounds=(300, 450, 400, 600, 500, 700))
        if block(350, 500, 600):  # 在长方体内
            print("在长方体内")
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    def block_func(x, y, z):
        return (x_min <= x <= x_max and 
                y_min <= y <= y_max and 
                z_min <= z <= z_max)
    
    return block_func
