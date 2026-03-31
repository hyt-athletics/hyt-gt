"""
地质体管理器

存储、保存、加载所有地质体

每个地质体本质上就是：
  - 一个名字
  - XZ截面上的多边形轮廓
  - Y方向延伸范围
  - 物性值（密度等）
  - 显示颜色
"""

import numpy as np
import json
import os


# 多方法统一物性字段
BODY_PROPERTY_DEFAULTS = {
    'density': 0.0,
    'susceptibility': 0.0,
    'resistivity': 100.0,
    'conductivity': 0.01,
    'chargeability': 0.0,
    'vp': 3000.0,
    'vs': 1500.0,
    'permittivity': 1.0,
    'quality_factor': 100.0,
    'remanent_mag': 0.0,
    'remanent_inclination': 0.0,
    'remanent_declination': 0.0,
}

BODY_PROPERTY_GROUPS = {
    '基本属性': [
        ('name', '名称'),
        ('y_min', 'Y起始 (m)'),
        ('y_max', 'Y结束 (m)'),
    ],
    '重力/磁法': [
        ('density', '密度差 (g/cm³)'),
        ('susceptibility', '磁化率 (SI)'),
    ],
    '电法/电磁': [
        ('resistivity', '电阻率 (Ω·m)'),
        ('conductivity', '电导率 (S/m)'),
        ('chargeability', '极化率 (V/V)'),
        ('permittivity', '相对介电常数'),
    ],
    '声波/地震': [
        ('vp', 'P波速度 (m/s)'),
        ('vs', 'S波速度 (m/s)'),
        ('quality_factor', '品质因子 Q'),
    ],
    '磁法高级': [
        ('remanent_mag', '剩余磁化强度 (A/m)'),
        ('remanent_inclination', '剩余磁化倾角 (°)'),
        ('remanent_declination', '剩余磁化偏角 (°)'),
    ],
}


# 预定义颜色（画新的体时自动分配）
BODY_COLORS = [
    '#e74c3c',  # 红
    '#3498db',  # 蓝
    '#2ecc71',  # 绿
    '#f39c12',  # 橙
    '#9b59b6',  # 紫
    '#1abc9c',  # 青
    '#e67e22',  # 深橙
    '#2c3e50',  # 深灰蓝
    '#e91e63',  # 粉
    '#00bcd4',  # 浅蓝
]


class GeologicalBody:
    """
    一个地质体（矿体/地层/空洞等）
    
    在XZ截面上是一个多边形
    沿Y方向延伸一定范围
    拥有物性值
    """
    
    def __init__(self, name, vertices_xz, density=0.0,
                 y_range=(0, 1000), color='#e74c3c', **property_values):
        """
        参数:
            name: 地质体名称，如 "矿体1"
            vertices_xz: XZ截面上的多边形顶点 [(x1,z1), (x2,z2), ...]
            density: 密度差 (g/cm³)
            y_range: Y方向延伸范围 (y_min, y_max)
            color: 显示颜色
        """
        self.name = name
        self.vertices_xz = list(vertices_xz)  # [(x,z), ...]
        self.y_range = tuple(y_range)
        self.color = color
        
        for key, value in BODY_PROPERTY_DEFAULTS.items():
            setattr(self, key, float(value))
        
        self.density = float(density)
        self.update_properties(property_values)

    @property
    def velocity(self):
        """向后兼容旧字段 velocity。"""
        return self.vp

    @velocity.setter
    def velocity(self, value):
        self.vp = float(value)

    def update_properties(self, properties):
        """批量更新物性，兼容旧字段并维护派生值。"""
        if not properties:
            self._sync_derived_properties()
            return
        
        normalized = dict(properties)
        if 'velocity' in normalized and 'vp' not in normalized:
            normalized['vp'] = normalized.pop('velocity')
        
        for key in BODY_PROPERTY_DEFAULTS:
            if key in normalized and normalized[key] is not None:
                setattr(self, key, float(normalized[key]))
        
        self._sync_derived_properties(explicit_conductivity='conductivity' in normalized)

    def _sync_derived_properties(self, explicit_conductivity=False):
        """同步由其他物性推导出的字段。"""
        if not explicit_conductivity:
            if self.resistivity > 0:
                self.conductivity = 1.0 / self.resistivity
            else:
                self.conductivity = 0.0

    def to_property_dict(self):
        """返回当前物性字段。"""
        return {key: getattr(self, key) for key in BODY_PROPERTY_DEFAULTS}

    def property_form_defaults(self):
        """供交互表单使用的默认值。"""
        values = self.to_property_dict()
        values.update({
            'name': self.name,
            'y_min': self.y_range[0],
            'y_max': self.y_range[1],
        })
        return values
    
    def contains_point_2d(self, px, pz):
        """判断点(px, pz)是否在多边形内（射线法）"""
        n = len(self.vertices_xz)
        inside = False
        j = n - 1
        for i in range(n):
            xi, zi = self.vertices_xz[i]
            xj, zj = self.vertices_xz[j]
            if ((zi > pz) != (zj > pz)) and \
               (px < (xj - xi) * (pz - zi) / (zj - zi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside
    
    def contains_point_3d(self, x, y, z):
        """判断3D点是否在体内"""
        if not (self.y_range[0] <= y <= self.y_range[1]):
            return False
        return self.contains_point_2d(x, z)
    
    def centroid_2d(self):
        """多边形重心"""
        pts = np.array(self.vertices_xz)
        return pts.mean(axis=0)
    
    def area_2d(self):
        """多边形面积（鞋带公式）"""
        pts = self.vertices_xz
        n = len(pts)
        area = 0
        for i in range(n):
            x1, z1 = pts[i]
            x2, z2 = pts[(i + 1) % n]
            area += x1 * z2 - x2 * z1
        return abs(area) / 2

    def bounds_2d(self):
        """XZ 截面的包围盒。"""
        pts = np.asarray(self.vertices_xz, dtype=float)
        return (
            (float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))),
            (float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))),
        )

    def bounds_3d(self, padding=0.0):
        """3D 包围盒，支持统一外扩。"""
        (x_min, x_max), (z_min, z_max) = self.bounds_2d()
        y_min, y_max = self.y_range
        pad = float(max(0.0, padding))
        return {
            'x_range': (x_min - pad, x_max + pad),
            'y_range': (float(y_min) - pad, float(y_max) + pad),
            'z_range': (z_min - pad, z_max + pad),
        }

    def extents_3d(self):
        """返回地质体三向尺度。"""
        bounds = self.bounds_3d(padding=0.0)
        return (
            bounds['x_range'][1] - bounds['x_range'][0],
            bounds['y_range'][1] - bounds['y_range'][0],
            bounds['z_range'][1] - bounds['z_range'][0],
        )
    
    def to_dict(self):
        """转成字典（用于保存）"""
        data = {
            'name': self.name,
            'vertices_xz': self.vertices_xz,
            'y_range': list(self.y_range),
            'color': self.color,
        }
        data.update(self.to_property_dict())
        data['velocity'] = self.velocity
        return data
    
    @classmethod
    def from_dict(cls, d):
        """从字典创建（用于加载）"""
        body = cls(
            name=d['name'],
            vertices_xz=[tuple(v) for v in d['vertices_xz']],
            density=d.get('density', 0.0),
            y_range=tuple(d.get('y_range', (0, 1000))),
            color=d.get('color', '#e74c3c'),
        )
        body.update_properties(d)
        return body


class ModelManager:
    """
    模型管理器
    
    管理所有地质体 + 钻孔 + 模型域信息
    支持保存/加载到JSON文件
    """
    
    def __init__(self):
        self.bodies = []
        self.domain = {
            'x_range': (0, 1000),
            'y_range': (0, 1000),
            'z_range': (0, 1000),
        }
        self.well_info = {
            'x': 500, 'y': 500,
            'z_top': 10, 'z_bottom': 800,
            'n_stations': 40,
        }
    
    def add_body(self, body):
        """添加一个地质体"""
        self.bodies.append(body)
    
    def remove_body(self, index):
        """删除一个地质体"""
        if 0 <= index < len(self.bodies):
            removed = self.bodies.pop(index)
            return removed
        return None
    
    def get_next_color(self):
        """获取下一个可用颜色"""
        idx = len(self.bodies) % len(BODY_COLORS)
        return BODY_COLORS[idx]
    
    def get_next_name(self):
        """获取下一个默认名称"""
        return f"body_{len(self.bodies) + 1}"
    
    def find_body_at(self, x, z):
        """
        找到包含点(x,z)的地质体
        
        从最后添加的开始查找（后画的在上面）
        """
        for i in range(len(self.bodies) - 1, -1, -1):
            if self.bodies[i].contains_point_2d(x, z):
                return i
        return None
    
    def save(self, filename):
        """
        保存模型到JSON文件
        
        参数:
            filename: 文件名，如 'my_model.json'
        """
        data = {
            'domain': self.domain,
            'well': self.well_info,
            'bodies': [b.to_dict() for b in self.bodies],
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ 模型已保存: {filename}")
        print(f"     包含 {len(self.bodies)} 个地质体")
    
    def load(self, filename):
        """
        从JSON文件加载模型
        """
        if not os.path.exists(filename):
            print(f"  ❌ 文件不存在: {filename}")
            return False
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.domain = data.get('domain', self.domain)
        self.well_info = data.get('well', self.well_info)
        self.bodies = [GeologicalBody.from_dict(d) for d in data.get('bodies', [])]
        
        print(f"  ✅ 模型已加载: {filename}")
        print(f"     包含 {len(self.bodies)} 个地质体")
        return True
    
    def print_summary(self):
        """打印模型摘要"""
        print(f"\n  ┌─ 模型摘要 ─────────────────────┐")
        print(f"  │ 地质体数量: {len(self.bodies)}")
        for i, b in enumerate(self.bodies):
            marker = "●" 
            print(f"  │ {marker} [{i}] {b.name}")
            print(f"  │     密度差={b.density:.2f} g/cm³")
            print(f"  │     Y范围={b.y_range[0]:.0f}~{b.y_range[1]:.0f} m")
            print(f"  │     面积={b.area_2d():.0f} m²")
        print(f"  └──────────────────────────────────┘")
