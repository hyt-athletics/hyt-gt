"""
约束定义工具

方便地从几何描述生成约束边列表
"""

import numpy as np


def polygon_to_constraints(polygon_indices):
    """
    把多边形顶点索引列表转成约束边列表
    
    输入: [0, 1, 2, 3, 4]  表示 5个顶点的多边形
    输出: [(0,1), (1,2), (2,3), (3,4), (4,0)]  5条边
    
    注意：多边形自动闭合
    """
    n = len(polygon_indices)
    constraints = []
    for i in range(n):
        j = (i + 1) % n
        constraints.append((polygon_indices[i], polygon_indices[j]))
    return constraints


def polyline_to_constraints(point_indices):
    """
    把折线点索引转成约束边列表（不闭合）
    
    输入: [0, 1, 2, 3]
    输出: [(0,1), (1,2), (2,3)]
    """
    constraints = []
    for i in range(len(point_indices) - 1):
        constraints.append((point_indices[i], point_indices[i + 1]))
    return constraints


def add_polygon_constraint(all_points, polygon_vertices, existing_point_count=0):
    """
    添加一个多边形约束
    
    参数:
        all_points: 全局点列表（会被修改，添加新点）
        polygon_vertices: 多边形顶点坐标 [(x1,z1), ...]
        existing_point_count: 已有点的数量
    
    返回:
        constraints: 新添加的约束边列表
        new_point_count: 添加的点数
    """
    # 添加多边形顶点到点集
    start_idx = len(all_points)
    all_points.extend(polygon_vertices)
    end_idx = len(all_points)
    
    # 生成顶点索引
    indices = list(range(start_idx, end_idx))
    
    # 生成约束边
    constraints = polygon_to_constraints(indices)
    
    return constraints, len(polygon_vertices)


class ConstraintBuilder:
    """
    约束构建器
    
    方便地构建点集和约束边的集合
    """
    
    def __init__(self):
        self.points = []
        self.constraints = []
        self.point_labels = {}  # 标签 → 索引的映射
    
    def add_point(self, x, z, label=None):
        """添加单个点"""
        idx = len(self.points)
        self.points.append((x, z))
        if label:
            self.point_labels[label] = idx
        return idx
    
    def add_points(self, points_list):
        """批量添加点"""
        start_idx = len(self.points)
        self.points.extend(points_list)
        return list(range(start_idx, len(self.points)))
    
    def add_polygon(self, vertices, label=None):
        """添加多边形（自动添加约束边）"""
        indices = self.add_points(vertices)
        self.constraints.extend(polygon_to_constraints(indices))
        if label:
            self.point_labels[label] = indices
        return indices
    
    def add_polyline(self, vertices, label=None):
        """添加折线（自动添加约束边）"""
        indices = self.add_points(vertices)
        self.constraints.extend(polyline_to_constraints(indices))
        if label:
            self.point_labels[label] = indices
        return indices
    
    def add_constraint_edge_by_labels(self, label1, label2):
        """通过标签添加约束边"""
        if label1 in self.point_labels and label2 in self.point_labels:
            i = self.point_labels[label1]
            j = self.point_labels[label2]
            self.constraints.append((i, j))
    
    def build(self):
        """返回点集和约束列表"""
        return self.points, self.constraints