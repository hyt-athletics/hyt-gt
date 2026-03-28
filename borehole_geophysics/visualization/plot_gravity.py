"""
钻孔重力异常可视化

包括：
  - 单井重力剖面
  - 多井对比
  - 重力 + 地质体联合显示
  - 灵敏度分布图
"""

import numpy as np

# 设置中文字体（必须在导入pyplot之前）
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def plot_borehole_gravity(result, title=None, figsize=(6, 10)):
    """
    画单井重力剖面
    
    左边 = 深度轴（向下）
    右边 = 重力异常曲线
    
        depth(m)│ gz (mGal) →
         0 ─────┤
        100 ────┤──·
        200 ────┤────·
        300 ────┤──────·
        400 ────┤────────·    ← 峰值
        500 ────┤──────·
        600 ────┤────·
        700 ────┤──·
        800 ────┤·
                └──────────
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 画重力曲线
    ax.plot(result.gz, result.depths, 'b-', linewidth=2, label='gz')
    
    # 填充正异常（红色）和负异常（蓝色）
    ax.fill_betweenx(result.depths, 0, result.gz,
                      where=(result.gz > 0),
                      color='#e74c3c', alpha=0.2, label='正异常')
    ax.fill_betweenx(result.depths, 0, result.gz,
                      where=(result.gz < 0),
                      color='#3498db', alpha=0.2, label='负异常')
    
    # 零线
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    
    # 标注峰值
    peak_idx = np.argmax(np.abs(result.gz))
    peak_gz = result.gz[peak_idx]
    peak_depth = result.depths[peak_idx]
    ax.plot(peak_gz, peak_depth, 'r*', markersize=15, zorder=10)
    ax.annotate(
        f'{peak_gz:.3f} mGal\n@ {peak_depth:.0f}m',
        xy=(peak_gz, peak_depth),
        xytext=(peak_gz + (result.gz.max() - result.gz.min()) * 0.2, peak_depth),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red', fontweight='bold'
    )
    
    # 轴设置
    ax.set_xlabel('gz (mGal)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.invert_yaxis()  # 深度向下
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title(
            f'Borehole Gravity - ({result.well.x:.0f}, {result.well.y:.0f})',
            fontsize=13
        )
    
    plt.tight_layout()
    plt.show()


def plot_gravity_with_model(result, bodies=None, figsize=(14, 10)):
    """
    重力曲线 + 地质模型 联合显示
    
    左图：XZ截面上的地质体
    右图：重力剖面
    
    ┌──────────────┬────────────┐
    │              │            │
    │   地质模型    │  重力曲线   │
    │   (XZ截面)   │ (gz vs Z)  │
    │              │            │
    │  ┌──┐       │    ·       │
    │  │矿│ 钻孔  │   ·        │
    │  │体│  │    │  ·         │
    │  └──┘       │   ·        │
    │              │    ·       │
    └──────────────┴────────────┘
    """
    fig, (ax_model, ax_gravity) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={'width_ratios': [2, 1]},
        sharey=True
    )
    
    # ===== 左图：地质模型 =====
    ax_model.set_xlabel('X (m)', fontsize=12)
    ax_model.set_ylabel('Depth (m)', fontsize=12)
    ax_model.set_title('XZ Cross Section (Y=well)', fontsize=13)
    ax_model.invert_yaxis()
    ax_model.grid(True, alpha=0.3)
    ax_model.set_aspect('equal')
    
    # 画地质体
    if bodies:
        for body in bodies:
            polygon = MplPolygon(
                body.vertices_xz, closed=True,
                facecolor=body.color, alpha=0.4,
                edgecolor=body.color, linewidth=2
            )
            ax_model.add_patch(polygon)
            
            # 标签
            cx, cz = body.centroid_2d()
            ax_model.text(cx, cz, f'{body.name}\nρ={body.density:.2f}',
                         ha='center', va='center', fontsize=8,
                         fontweight='bold', color='white',
                         bbox=dict(boxstyle='round', facecolor=body.color, alpha=0.7))
    
    # 画钻孔
    well = result.well
    ax_model.axvline(x=well.x, color='red', linewidth=2.5, zorder=10)
    ax_model.plot(well.stations[:, 0], well.stations[:, 2],
                 'r.', markersize=4, zorder=11)
    
    # 设置模型范围
    if result.mesh_data is not None:
        centers = result.mesh_data['centers']
        x_margin = 50
        ax_model.set_xlim(centers[:, 0].min() - x_margin,
                         centers[:, 0].max() + x_margin)
    
    # ===== 右图：重力曲线 =====
    ax_gravity.plot(result.gz, result.depths, 'b-', linewidth=2)
    ax_gravity.fill_betweenx(result.depths, 0, result.gz,
                              where=(result.gz > 0),
                              color='#e74c3c', alpha=0.2)
    ax_gravity.fill_betweenx(result.depths, 0, result.gz,
                              where=(result.gz < 0),
                              color='#3498db', alpha=0.2)
    ax_gravity.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    
    # 峰值标注
    peak_idx = np.argmax(np.abs(result.gz))
    ax_gravity.plot(result.gz[peak_idx], result.depths[peak_idx],
                   'r*', markersize=15, zorder=10)
    
    ax_gravity.set_xlabel('gz (mGal)', fontsize=12)
    ax_gravity.set_title('Borehole Gravity', fontsize=13)
    ax_gravity.grid(True, alpha=0.3)
    
    # 画水平虚线连接两图（峰值处）
    for depth in [result.depths[peak_idx]]:
        ax_model.axhline(y=depth, color='gray', linewidth=0.5,
                        linestyle=':', alpha=0.5)
        ax_gravity.axhline(y=depth, color='gray', linewidth=0.5,
                          linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_multi_well_gravity(results, figsize=(10, 10)):
    """
    多井重力对比
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        label = f'Well ({result.well.x:.0f}, {result.well.y:.0f})'
        ax.plot(result.gz, result.depths, '-', linewidth=2,
               color=color, label=label)
    
    ax.set_xlabel('gz (mGal)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Multi-well Borehole Gravity Comparison', fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_gravity_depth_slice(result, mesh_data, depth, figsize=(10, 8)):
    """
    某个深度处的重力异常平面分布（需要多个虚拟钻孔）
    
    这是高级功能的预留接口
    """
    pass  # 后续实现