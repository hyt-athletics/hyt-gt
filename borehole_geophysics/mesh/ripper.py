"""
多边形剖分工具（耳切法）

耳切法是处理多边形三角化的经典算法：
  找到多边形的一个"耳朵"，切掉，重复直到切完
"""

import numpy as np


def polygon_area(polygon):
    """计算多边形面积（鞋带公式）"""
    n = len(polygon)
    area = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def is_convex(p0, p1, p2):
    """
    判断三个点是否构成凸角（逆时针）
    
         p2
        ╱
       ╱  ← 凸角
      p1
     ╱
    p0
    """
    cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
    return cross > 0


def point_in_triangle(p, a, b, c):
    """
    判断点p是否在三角形abc内（重心坐标法）
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


def ear_clipping_triangulation(polygon_points, polygon_indices):
    """
    耳切法三角化多边形
    
    参数:
        polygon_points: 多边形顶点坐标列表
        polygon_indices: 顶点在原点集中的索引
    
    返回:
        triangles: 三角形列表 [(i,j,k), ...] 使用原索引
    """
    if len(polygon_indices) < 3:
        return []
    
    triangles = []
    remaining = list(polygon_indices)
    
    # 安全阀
    max_iter = len(polygon_indices) * 10
    iteration = 0
    
    while len(remaining) > 3 and iteration < max_iter:
        iteration += 1
        ear_found = False
        
        for i in range(len(remaining)):
            prev_idx = remaining[(i - 1) % len(remaining)]
            curr_idx = remaining[i]
            next_idx = remaining[(i + 1) % len(remaining)]
            
            prev_pt = polygon_points[prev_idx]
            curr_pt = polygon_points[curr_idx]
            next_pt = polygon_points[next_idx]
            
            # 检查是否是凸角
            if not is_convex(prev_pt, curr_pt, next_pt):
                continue
            
            # 检查这个三角形是否包含其他点
            is_ear = True
            for j in range(len(remaining)):
                if j in [(i - 1) % len(remaining), i, (i + 1) % len(remaining)]:
                    continue
                
                test_idx = remaining[j]
                test_pt = polygon_points[test_idx]
                
                if point_in_triangle(test_pt, prev_pt, curr_pt, next_pt):
                    is_ear = False
                    break
            
            if is_ear:
                # 找到耳朵！切掉
                triangles.append((prev_idx, curr_idx, next_idx))
                remaining.pop(i)
                ear_found = True
                break
        
        if not ear_found:
            # 找不到耳朵，可能是退化情况
            print("  警告：找不到耳朵，多边形可能自交或退化")
            break
    
    # 最后剩下3个点，形成最后一个三角形
    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    
    return triangles