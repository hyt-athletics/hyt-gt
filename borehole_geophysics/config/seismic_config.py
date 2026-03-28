"""
VSP地震观测系统配置
"""

import numpy as np


class SeismicSource:
    """震源"""
    def __init__(self, x, y, z, frequency=50.0, label=''):
        self.x = x
        self.y = y
        self.z = z
        self.frequency = frequency       # 主频 (Hz)
        self.label = label or f"Src({x:.0f},{y:.0f},{z:.0f})"
    
    @property
    def position(self):
        return np.array([self.x, self.y, self.z])


class SeismicReceiver:
    """检波器"""
    def __init__(self, x, y, z, label=''):
        self.x = x
        self.y = y
        self.z = z
        self.label = label
    
    @property
    def position(self):
        return np.array([self.x, self.y, self.z])


class VSPSurvey:
    """
    VSP观测系统
    """
    
    def __init__(self, name='vsp'):
        self.name = name
        self.sources = []
        self.receivers = []
    
    def add_source(self, x, y, z, frequency=50.0, label=''):
        src = SeismicSource(x, y, z, frequency, label)
        self.sources.append(src)
        return len(self.sources) - 1
    
    def add_receiver(self, x, y, z, label=''):
        rec = SeismicReceiver(x, y, z, label)
        self.receivers.append(rec)
        return len(self.receivers) - 1
    
    @property
    def max_frequency(self):
        return max(s.frequency for s in self.sources) if self.sources else 50.0
    
    @property
    def all_special_points(self):
        """所有需要精确网格控制的点"""
        pts = []
        for s in self.sources:
            pts.append(s.position)
        for r in self.receivers:
            pts.append(r.position)
        return np.array(pts) if pts else np.empty((0, 3))
    
    def get_info(self):
        print(f"""
    ┌─ VSP观测系统 "{self.name}" ──────────┐
    │ 震源数: {len(self.sources)}
    │ 检波器数: {len(self.receivers)}
    │ 主频: {self.max_frequency:.0f} Hz
    └──────────────────────────────────────┘
        """)


def create_zero_offset_vsp(
        well_x, well_y, z_top, z_bottom,
        n_receivers=40,
        source_offset=50,
        source_frequency=50.0):
    """
    零偏移距VSP
    
    震源在井口附近地表，检波器在钻孔里
    
         ★ 震源（地表，距井口50m）
    ═════╤════════
         │ ← 钻孔
         ●  检波器1
         ●  检波器2
         ●  ...
         ●  检波器n
    """
    survey = VSPSurvey(name='zero_offset_vsp')
    
    survey.add_source(well_x + source_offset, well_y, 0,
                      frequency=source_frequency, label='Source')
    
    zs = np.linspace(z_top, z_bottom, n_receivers)
    for i, z in enumerate(zs):
        survey.add_receiver(well_x, well_y, z, label=f'R_{i}')
    
    return survey


def create_walkaway_vsp(
        well_x, well_y, z_top, z_bottom,
        n_receivers=30,
        source_x_positions=None,
        source_frequency=40.0):
    """
    走离VSP（多个不同偏移距的震源）
    
    ★1  ★2  ★3  ★4  ★5       多个震源，从近到远
    ═════╤════════════════
         │
         ● 检波器（钻孔里）
         ●
         ●
    """
    survey = VSPSurvey(name='walkaway_vsp')
    
    if source_x_positions is None:
        source_x_positions = np.arange(well_x + 50, well_x + 501, 100)
    
    for i, sx in enumerate(source_x_positions):
        survey.add_source(sx, well_y, 0, frequency=source_frequency,
                          label=f'Src_{i}')
    
    zs = np.linspace(z_top, z_bottom, n_receivers)
    for i, z in enumerate(zs):
        survey.add_receiver(well_x, well_y, z, label=f'R_{i}')
    
    return survey