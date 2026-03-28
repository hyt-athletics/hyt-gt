"""
边界扩展（Padding）

电法和地震都需要把模型域扩大：

电法：模拟"无穷远"边界条件
  ┌─────────────────────────────────┐
  │  padding（网格越来越粗）          │
  │  ┌─────────────────────────┐    │
  │  │  padding                │    │
  │  │  ┌─────────────────┐   │    │
  │  │  │   研究区          │   │    │
  │  │  │  （精细网格）      │   │    │
  │  │  └─────────────────┘   │    │
  │  │                        │    │
  │  └─────────────────────────┘    │
  │                                 │
  └─────────────────────────────────┘

地震：放置吸收边界（PML/Sponge Layer）
  ┌────────────────────────────────┐
  │  吸收层（波进来就被"吃掉"）      │
  │  ┌────────────────────────┐    │
  │  │   研究区                │    │
  │  │  （波在里面传播）        │    │
  │  └────────────────────────┘    │
  │                                │
  └────────────────────────────────┘
"""

import numpy as np


def compute_padding_layers(core_range, n_pad_layers, expansion_factor=1.5,
                            first_pad_size=None):
    """
    计算边界扩展层的位置
    
    参数:
        core_range: (min, max) 核心区范围
        n_pad_layers: 扩展层数
        expansion_factor: 每层增长倍数
        first_pad_size: 第一层扩展厚度（默认=核心区的格子尺寸）
    
    返回:
        edges: 完整的边界位置数组（包括核心区 + 扩展层）
    
    示例:
        core_range = (0, 1000)
        → 核心区: 0 ~ 1000m
        → 扩展后: -500 ~ 1500m（左右各加几层越来越粗的格子）
    """
    core_min, core_max = core_range
    core_size = core_max - core_min
    
    if first_pad_size is None:
        first_pad_size = core_size / 10
    
    # 向左扩展
    left_edges = [core_min]
    size = first_pad_size
    for i in range(n_pad_layers):
        left_edges.insert(0, left_edges[0] - size)
        size *= expansion_factor
    
    # 向右扩展
    right_edges = [core_max]
    size = first_pad_size
    for i in range(n_pad_layers):
        right_edges.append(right_edges[-1] + size)
        size *= expansion_factor
    
    # 合并（去掉重复的核心边界点）
    all_edges = left_edges + right_edges[1:]
    
    return np.array(all_edges)


def compute_padded_domain(core_domain, n_pad_layers=5, expansion_factor=1.5):
    """
    计算三维扩展域
    
    参数:
        core_domain: {'x_range': (0,1000), 'y_range': (0,1000), 'z_range': (0,1000)}
        n_pad_layers: 每个方向的扩展层数
        expansion_factor: 增长倍数
    
    返回:
        padded_domain: 扩展后的域
        padding_info: 扩展信息
    """
    results = {}
    info = {}
    
    for axis in ['x_range', 'y_range', 'z_range']:
        core_range = core_domain[axis]
        edges = compute_padding_layers(core_range, n_pad_layers, expansion_factor)
        results[axis] = (edges[0], edges[-1])
        info[axis] = {
            'core': core_range,
            'padded': (edges[0], edges[-1]),
            'edges': edges,
            'n_pad_left': n_pad_layers,
            'n_pad_right': n_pad_layers,
        }
    
    # Z方向通常只向下扩展（地表以上不需要）
    z_core = core_domain['z_range']
    z_edges = compute_padding_layers(z_core, n_pad_layers, expansion_factor)
    # 地表以上不扩展太多
    z_top = max(z_edges[0], -z_core[0])
    results['z_range'] = (z_top, z_edges[-1])
    info['z_range']['padded'] = (z_top, z_edges[-1])
    
    return results, info


def get_padding_mask(cell_centers, core_domain):
    """
    标记哪些格子在核心区内，哪些在扩展区
    
    返回:
        mask: 布尔数组，True=核心区内
    """
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    z = cell_centers[:, 2]
    
    x0, x1 = core_domain['x_range']
    y0, y1 = core_domain['y_range']
    z0, z1 = core_domain['z_range']
    
    mask = ((x >= x0) & (x <= x1) &
            (y >= y0) & (y <= y1) &
            (z >= z0) & (z <= z1))
    
    return mask


def print_padding_info(core_domain, padded_domain, info):
    """打印扩展信息"""
    print(f"""
    ┌─ 边界扩展信息 ─────────────────────────────────┐
    │ 方向   核心区范围          扩展后范围              │""")
    for axis_name, ax_key in [('X', 'x_range'), ('Y', 'y_range'), ('Z', 'z_range')]:
        c = core_domain[ax_key]
        p = padded_domain[ax_key]
        print(f"    │  {axis_name}    {c[0]:8.0f} ~ {c[1]:8.0f}    "
              f"{p[0]:8.0f} ~ {p[1]:8.0f}   │")
    
    # 计算体积比
    core_vol = 1
    pad_vol = 1
    for ax in ['x_range', 'y_range', 'z_range']:
        core_vol *= core_domain[ax][1] - core_domain[ax][0]
        pad_vol *= padded_domain[ax][1] - padded_domain[ax][0]
    
    print(f"""    │                                                │
    │ 核心区体积: {core_vol/1e9:.2f} km³                 │
    │ 扩展后体积: {pad_vol/1e9:.2f} km³                  │
    │ 体积比: {pad_vol/core_vol:.1f}x                    │
    └────────────────────────────────────────────────┘
    """)