"""
地层线工具

地层线 = 不同地层之间的分界面

在XZ截面上表现为一条曲线：
    地层1 (密度 2.5)
    ─────●───────●────────●────── ← 地层线（可以弯曲）
    地层2 (密度 2.7)
    ────────●──────────●──────── ← 另一条地层线
    地层3 (密度 2.9)

地层线和矿体的区别：
  矿体 = 封闭的多边形
  地层线 = 横穿整个模型的曲线，上下是不同的密度
"""

import numpy as np
from interactive.body_manager import GeologicalBody


class GeoLayer:
    """
    地层线
    
    由XZ截面上的一系列控制点定义
    两个控制点之间自动插值
    """
    
    def __init__(self, name, control_points, density_above, density_below,
                 y_range=(0, 1000), color='#795548'):
        """
        参数:
            name: 地层线名称
            control_points: 控制点列表 [(x1,z1), (x2,z2), ...]
                           从左到右排列
            density_above: 线上方的密度差
            density_below: 线下方的密度差
            y_range: Y方向范围
            color: 显示颜色
        """
        self.name = name
        self.control_points = sorted(control_points, key=lambda p: p[0])
        self.density_above = density_above
        self.density_below = density_below
        self.y_range = y_range
        self.color = color
    
    def z_at_x(self, x):
        """
        获取给定x位置处的地层线深度（线性插值）
        
        如果x在控制点范围外，用最近端点的值
        """
        pts = self.control_points
        
        if len(pts) == 0:
            return 0
        if len(pts) == 1:
            return pts[0][1]
        
        # x在左端以左
        if x <= pts[0][0]:
            return pts[0][1]
        # x在右端以右
        if x >= pts[-1][0]:
            return pts[-1][1]
        
        # 找到x所在的区间
        for i in range(len(pts) - 1):
            x1, z1 = pts[i]
            x2, z2 = pts[i + 1]
            if x1 <= x <= x2:
                t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                return z1 + t * (z2 - z1)
        
        return pts[-1][1]
    
    def get_dense_points(self, x_range, n_points=200):
        """获取密集点（用于平滑绘制）"""
        x_min, x_max = x_range
        xs = np.linspace(x_min, x_max, n_points)
        zs = [self.z_at_x(x) for x in xs]
        return xs, zs
    
    def point_is_above(self, x, z):
        """判断点是否在地层线上方"""
        layer_z = self.z_at_x(x)
        return z < layer_z  # z向下为正，所以z小 = 在上方
    
    def to_dict(self):
        return {
            'name': self.name,
            'control_points': self.control_points,
            'density_above': self.density_above,
            'density_below': self.density_below,
            'y_range': list(self.y_range),
            'color': self.color,
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d['name'],
            control_points=[tuple(p) for p in d['control_points']],
            density_above=d.get('density_above', 0.0),
            density_below=d.get('density_below', 0.0),
            y_range=tuple(d.get('y_range', (0, 1000))),
            color=d.get('color', '#795548'),
        )


def layers_to_bodies(layers, x_range, z_range):
    """
    把地层线转换成地质体（填充两条线之间的区域）
    
    参数:
        layers: GeoLayer列表（从上到下排列）
        x_range: (x_min, x_max)
        z_range: (z_min, z_max)
    
    返回:
        bodies: GeologicalBody列表
    """
    bodies = []
    x_min, x_max = x_range
    z_min, z_max = z_range
    n_x = 50  # 采样点数
    xs = np.linspace(x_min, x_max, n_x)
    
    # 按平均深度排序
    sorted_layers = sorted(layers, key=lambda l: np.mean([p[1] for p in l.control_points]))
    
    # 地表到第一条线
    if sorted_layers:
        first = sorted_layers[0]
        verts = [(x, first.z_at_x(x)) for x in xs]
        verts = [(x_max, z_min), (x_min, z_min)] + verts[::-1]
        
        body = GeologicalBody(
            name=f"layer_above_{first.name}",
            vertices_xz=verts,
            density=first.density_above,
            y_range=first.y_range,
            color='#d4a76a',
        )
        if abs(body.density) > 1e-10:
            bodies.append(body)
    
    # 相邻两条线之间
    for i in range(len(sorted_layers) - 1):
        top_layer = sorted_layers[i]
        bot_layer = sorted_layers[i + 1]
        
        top_zs = [top_layer.z_at_x(x) for x in xs]
        bot_zs = [bot_layer.z_at_x(x) for x in xs]
        
        verts = []
        for j in range(n_x):
            verts.append((xs[j], top_zs[j]))
        for j in range(n_x - 1, -1, -1):
            verts.append((xs[j], bot_zs[j]))
        
        body = GeologicalBody(
            name=f"layer_{top_layer.name}_to_{bot_layer.name}",
            vertices_xz=verts,
            density=top_layer.density_below,
            y_range=top_layer.y_range,
            color=top_layer.color,
        )
        if abs(body.density) > 1e-10:
            bodies.append(body)
    
    # 最后一条线到底部
    if sorted_layers:
        last = sorted_layers[-1]
        verts = [(x, last.z_at_x(x)) for x in xs]
        verts = verts + [(x_max, z_max), (x_min, z_max)]
        
        body = GeologicalBody(
            name=f"layer_below_{last.name}",
            vertices_xz=verts,
            density=last.density_below,
            y_range=last.y_range,
            color='#8d6e63',
        )
        if abs(body.density) > 1e-10:
            bodies.append(body)
    
    return bodies