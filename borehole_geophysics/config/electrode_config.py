"""
电极配置

定义井地电法的各种电极排列
"""

import numpy as np


class Electrode:
    """单个电极"""
    
    def __init__(self, x, y, z, electrode_type='M', label=''):
        """
        参数:
            x, y, z: 电极坐标 (m)
            electrode_type: 'A'=正源, 'B'=负源, 'M'=测量正, 'N'=测量负
            label: 标签
        """
        self.x = x
        self.y = y
        self.z = z
        self.type = electrode_type
        self.label = label or f"{electrode_type}({x:.0f},{y:.0f},{z:.0f})"
    
    @property
    def position(self):
        return np.array([self.x, self.y, self.z])
    
    @property
    def is_source(self):
        """是否是供电电极"""
        return self.type in ('A', 'B')


class ElectrodeArray:
    """
    电极排列
    
    包含所有电极 + 测量方案
    """
    
    def __init__(self, name='survey'):
        self.name = name
        self.electrodes = []
        self.measurements = []  # [(A_idx, B_idx, M_idx, N_idx), ...]
    
    def add_electrode(self, x, y, z, electrode_type='M', label=''):
        """添加一个电极"""
        e = Electrode(x, y, z, electrode_type, label)
        self.electrodes.append(e)
        return len(self.electrodes) - 1
    
    def add_measurement(self, a_idx, b_idx, m_idx, n_idx):
        """添加一个四极测量"""
        self.measurements.append((a_idx, b_idx, m_idx, n_idx))
    
    @property
    def source_electrodes(self):
        """所有供电电极"""
        return [e for e in self.electrodes if e.is_source]
    
    @property
    def all_positions(self):
        """所有电极坐标"""
        return np.array([[e.x, e.y, e.z] for e in self.electrodes])
    
    @property
    def unique_source_positions(self):
        """不重复的供电位置（用于确定需要解多少次方程）"""
        src = set()
        for a, b, m, n in self.measurements:
            src.add((self.electrodes[a].x, self.electrodes[a].y, self.electrodes[a].z))
            src.add((self.electrodes[b].x, self.electrodes[b].y, self.electrodes[b].z))
        return [np.array(s) for s in src]
    
    def get_info(self):
        n_src = len([e for e in self.electrodes if e.is_source])
        n_mea = len([e for e in self.electrodes if not e.is_source])
        print(f"""
    ┌─ 电极排列 "{self.name}" ─────────────┐
    │ 总电极数: {len(self.electrodes)}
    │   供电电极: {n_src}
    │   测量电极: {n_mea}
    │ 测量数: {len(self.measurements)}
    └──────────────────────────────────────┘
        """)


# ===================================================================
#  预定义的常用电极排列
# ===================================================================

def create_borehole_surface_survey(
        well_x, well_y, well_z_top, well_z_bottom,
        n_borehole_electrodes=20,
        surface_line_x_range=(0, 1000),
        surface_y=None,
        n_surface_electrodes=21,
        surface_z=0):
    """
    创建井地电法排列
    
    钻孔里均匀分布电极 + 地表一条测线上均匀分布电极
    
    参数:
        well_x, well_y: 钻孔水平位置
        well_z_top, well_z_bottom: 钻孔深度范围
        n_borehole_electrodes: 钻孔内电极数
        surface_line_x_range: 地表测线X范围
        n_surface_electrodes: 地表电极数
        surface_z: 地表高程
    """
    if surface_y is None:
        surface_y = well_y
    
    survey = ElectrodeArray(name='borehole_surface')
    
    # 钻孔电极（既可以当源也可以当接收）
    bh_z = np.linspace(well_z_top, well_z_bottom, n_borehole_electrodes)
    bh_indices = []
    for i, z in enumerate(bh_z):
        idx = survey.add_electrode(well_x, well_y, z, 'A', f'BH_{i}')
        bh_indices.append(idx)
    
    # 地表电极
    surf_x = np.linspace(surface_line_x_range[0], surface_line_x_range[1],
                          n_surface_electrodes)
    surf_indices = []
    for i, x in enumerate(surf_x):
        idx = survey.add_electrode(x, surface_y, surface_z, 'M', f'S_{i}')
        surf_indices.append(idx)
    
    # 测量方案：每个钻孔电极当源(A)，
    # 用一个远处的点当B（模拟单极源），
    # 地表所有电极做接收
    # 简化：用钻孔最底部的电极当B
    b_idx = bh_indices[-1]
    
    for a_idx in bh_indices[:-1]:  # 每个钻孔电极（除了B极）当A
        for m_idx in surf_indices:
            # N极用地表最远处的电极
            n_idx = surf_indices[-1] if m_idx != surf_indices[-1] else surf_indices[0]
            survey.add_measurement(a_idx, b_idx, m_idx, n_idx)
    
    return survey


def create_crosshole_survey(
        well1_x, well2_x, well_y,
        z_top, z_bottom, n_electrodes_per_well=20):
    """
    创建跨孔电法排列
    
    两口钻孔：一口发射，一口接收
    """
    survey = ElectrodeArray(name='crosshole')
    
    zs = np.linspace(z_top, z_bottom, n_electrodes_per_well)
    
    # 井1的电极（源）
    well1_indices = []
    for i, z in enumerate(zs):
        idx = survey.add_electrode(well1_x, well_y, z, 'A', f'W1_{i}')
        well1_indices.append(idx)
    
    # 井2的电极（接收）
    well2_indices = []
    for i, z in enumerate(zs):
        idx = survey.add_electrode(well2_x, well_y, z, 'M', f'W2_{i}')
        well2_indices.append(idx)
    
    # 测量方案
    b_idx = well1_indices[-1]
    for a_idx in well1_indices[:-1]:
        for m_idx in well2_indices:
            n_idx = well2_indices[-1] if m_idx != well2_indices[-1] else well2_indices[0]
            survey.add_measurement(a_idx, b_idx, m_idx, n_idx)
    
    return survey