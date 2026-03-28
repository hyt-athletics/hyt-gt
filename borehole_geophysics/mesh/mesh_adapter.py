"""
网格适配器基类

不同地球物理方法的网格需求不同
适配器负责：
  1. 分析特殊点（电极/震源）并确定局部加密需求
  2. 计算合适的网格尺寸
  3. 生成边界扩展
  4. 检查网格是否满足方法的要求
"""

import numpy as np


class MeshAdapter:
    """网格适配器基类"""
    
    def __init__(self, method_name):
        self.method_name = method_name
        self.requirements = {}   # 方法对网格的具体要求
    
    def get_special_points(self):
        """获取需要精确控制的特殊点（子类实现）"""
        raise NotImplementedError
    
    def get_refinement_rules(self):
        """获取加密规则列表（子类实现）"""
        raise NotImplementedError
    
    def get_domain_with_padding(self, core_domain):
        """获取带扩展的域（子类实现）"""
        raise NotImplementedError
    
    def check_mesh_quality(self, mesh_data):
        """检查网格是否满足方法要求（子类实现）"""
        raise NotImplementedError
    
    def print_requirements(self):
        """打印方法对网格的要求"""
        print(f"\n  ┌─ {self.method_name} 网格要求 ─────────────┐")
        for key, value in self.requirements.items():
            print(f"  │  {key}: {value}")
        print(f"  └────────────────────────────────────────┘")


class RefinementRule:
    """
    加密规则
    
    描述在什么位置、以什么方式、加密到什么程度
    """
    
    def __init__(self, point, min_size, max_size, influence_radius,
                 rule_type='radial', priority=1, label=''):
        """
        参数:
            point: (x, y, z) 加密中心
            min_size: 最小格子尺寸 (m)
            max_size: 最大格子尺寸 (m)
            influence_radius: 影响半径 (m)
            rule_type: 'radial'(径向渐变), 'axis'(沿轴), 'surface'(沿面)
            priority: 优先级（数字越大优先级越高）
        """
        self.point = np.array(point)
        self.min_size = min_size
        self.max_size = max_size
        self.influence_radius = influence_radius
        self.rule_type = rule_type
        self.priority = priority
        self.label = label
    
    def desired_size_at(self, position):
        """在给定位置计算期望的格子尺寸"""
        dist = np.linalg.norm(np.array(position) - self.point)
        
        if dist >= self.influence_radius:
            return self.max_size
        
        t = dist / self.influence_radius
        return self.min_size + t * (self.max_size - self.min_size)