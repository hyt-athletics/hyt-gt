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
from matplotlib.lines import Line2D


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
        label = getattr(result, 'label', f'Well ({result.well.x:.0f}, {result.well.y:.0f})')
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


def plot_gravity_comparison(results, figsize=(10, 10)):
    """同方法多次重力结果对比。"""
    return plot_multi_well_gravity(results, figsize=figsize)


def plot_gravity_model_vs_observed(modeled, observed, figsize=(7, 10)):
    """当前正演结果与实测重力曲线对比。"""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(modeled.gz, modeled.depths, linewidth=2.2, color='#1f77b4',
            label=getattr(modeled, 'label', 'modeled'))
    ax.plot(observed.gz, observed.depths, 'o--', linewidth=1.8, markersize=4,
            color='#d62728', label=getattr(observed, 'label', 'observed'))

    aligned_observed = np.interp(modeled.depths, observed.depths, observed.gz)
    rmse = float(np.sqrt(np.mean((modeled.gz - aligned_observed) ** 2)))

    ax.set_xlabel('gz (mGal)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Gravity: Modeled vs Observed  RMSE={rmse:.4f} mGal', fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


def plot_gravity_inversion_result(result, figsize=(14, 10)):
    """显示重力反演结果的观测/预测对比、恢复密度切片和直方图。"""
    fig, (ax_curve, ax_slice, ax_hist) = plt.subplots(
        1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1.1, 1.4, 1]}
    )

    ax_curve.plot(result.observed_gz, result.depths, 'o--', color='#d62728',
                  markersize=4, linewidth=1.6, label='observed')
    ax_curve.plot(result.predicted_gz, result.depths, '-', color='#1f77b4',
                  linewidth=2.2, label='predicted')
    ax_curve.set_xlabel('gz (mGal)', fontsize=12)
    ax_curve.set_ylabel('Depth (m)', fontsize=12)
    ax_curve.set_title(f'Gravity Inversion Fit\nRMSE={result.rmse:.4f} mGal', fontsize=13)
    ax_curve.invert_yaxis()
    ax_curve.grid(True, alpha=0.3)
    ax_curve.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_curve.legend(loc='best')

    _plot_inverse_density_slice(ax_slice, result, bodies=None, slice_y=result.well.y)

    density = np.asarray(result.recovered_density, dtype=float)
    update_mask = result.recovered_mesh_data.get('update_mask')
    fixed_mask = result.recovered_mesh_data.get('fixed_mask')
    if update_mask is not None and fixed_mask is not None:
        ax_hist.hist(density[np.asarray(fixed_mask, dtype=bool)], bins=30,
                     color='#b0b0b0', alpha=0.65, edgecolor='white', label='fixed')
        ax_hist.hist(density[np.asarray(update_mask, dtype=bool)], bins=30,
                     color='#4c72b0', alpha=0.8, edgecolor='white', label='updated')
        ax_hist.legend(loc='best')
    else:
        ax_hist.hist(density, bins=30, color='#4c72b0', alpha=0.85, edgecolor='white')
    ax_hist.set_xlabel('Recovered density (g/cc)', fontsize=12)
    ax_hist.set_ylabel('Cell count', fontsize=12)
    ax_hist.set_title('Recovered Model Histogram', fontsize=13)
    ax_hist.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def _extract_density_xz_slice(mesh_data, slice_y=None):
    """从 recovered mesh_data 提取最接近指定 y 的 XZ 切片。"""
    centers = np.asarray(mesh_data['centers'], dtype=float)
    sizes = np.asarray(mesh_data['sizes'], dtype=float)
    density = np.asarray(mesh_data['density'], dtype=float)
    update_mask = mesh_data.get('update_mask')
    fixed_mask = mesh_data.get('fixed_mask')

    y_values = np.unique(np.round(centers[:, 1], 10))
    if len(y_values) == 0:
        raise ValueError("recovered mesh_data 中没有可用单元")

    target_y = float(slice_y if slice_y is not None else y_values[len(y_values) // 2])
    actual_y = float(y_values[np.argmin(np.abs(y_values - target_y))])
    dy = float(np.median(sizes[:, 1]))
    mask = np.abs(centers[:, 1] - actual_y) <= dy * 0.51
    slice_centers = centers[mask]
    slice_density = density[mask]
    slice_update_mask = None if update_mask is None else np.asarray(update_mask, dtype=bool)[mask]
    slice_fixed_mask = None if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)[mask]
    if len(slice_centers) == 0:
        raise ValueError("未找到对应 y 切片单元")

    x_values = np.unique(np.round(slice_centers[:, 0], 10))
    z_values = np.unique(np.round(slice_centers[:, 2], 10))
    grid = np.full((len(z_values), len(x_values)), np.nan, dtype=float)
    update_grid = None if slice_update_mask is None else np.full((len(z_values), len(x_values)), np.nan, dtype=float)
    fixed_grid = None if slice_fixed_mask is None else np.full((len(z_values), len(x_values)), np.nan, dtype=float)

    x_index = {float(v): i for i, v in enumerate(x_values)}
    z_index = {float(v): i for i, v in enumerate(z_values)}
    for center, value in zip(slice_centers, slice_density):
        ix = x_index[float(round(center[0], 10))]
        iz = z_index[float(round(center[2], 10))]
        grid[iz, ix] = value
    if slice_update_mask is not None:
        for center, value in zip(slice_centers, slice_update_mask):
            ix = x_index[float(round(center[0], 10))]
            iz = z_index[float(round(center[2], 10))]
            update_grid[iz, ix] = 1.0 if value else 0.0
    if slice_fixed_mask is not None:
        for center, value in zip(slice_centers, slice_fixed_mask):
            ix = x_index[float(round(center[0], 10))]
            iz = z_index[float(round(center[2], 10))]
            fixed_grid[iz, ix] = 1.0 if value else 0.0

    dx = float(np.median(sizes[:, 0]))
    dz = float(np.median(sizes[:, 2]))
    extent = [
        float(x_values.min() - dx / 2.0),
        float(x_values.max() + dx / 2.0),
        float(z_values.max() + dz / 2.0),
        float(z_values.min() - dz / 2.0),
    ]
    return grid, extent, actual_y, update_grid, fixed_grid


def _plot_inverse_density_slice(ax, result, bodies=None, slice_y=None):
    """在指定坐标轴上绘制 recovered density 的 XZ 切片。"""
    grid, extent, actual_y, update_grid, fixed_grid = _extract_density_xz_slice(
        result.recovered_mesh_data, slice_y=slice_y
    )

    image = ax.imshow(
        grid,
        extent=extent,
        origin='upper',
        aspect='auto',
        cmap='RdBu_r',
    )
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    update_mode = result.metadata.get('update_mode')
    title = f'Recovered Density Slice\nY={actual_y:.1f} m'
    if update_mode:
        title += f'  [{update_mode}]'
    ax.set_title(title, fontsize=13)
    ax.grid(False)

    if update_grid is not None:
        ax.contour(
            np.where(np.isnan(update_grid), 0.0, update_grid),
            levels=[0.5],
            extent=extent,
            origin='upper',
            colors='black',
            linewidths=1.2,
        )
    if fixed_grid is not None:
        ax.contour(
            np.where(np.isnan(fixed_grid), 0.0, fixed_grid),
            levels=[0.5],
            extent=extent,
            origin='upper',
            colors='#666666',
            linewidths=0.8,
            linestyles=':',
        )
        legend_handles = [
            Line2D([0], [0], color='black', linewidth=1.2, label='updated zone'),
            Line2D([0], [0], color='#666666', linewidth=1.0, linestyle=':', label='fixed zone'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)

    if bodies:
        for body in bodies:
            polygon = MplPolygon(
                body.vertices_xz, closed=True,
                facecolor=body.color, alpha=0.18,
                edgecolor=body.color, linewidth=1.5
            )
            ax.add_patch(polygon)

    ax.axvline(result.well.x, color='black', linewidth=1.8, linestyle='--')
    ax.plot(result.well.stations[:, 0], result.well.stations[:, 2],
            'k.', markersize=3, zorder=10)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Density (g/cc)')


def plot_gravity_inverse_density_slice(result, bodies=None, slice_y=None, figsize=(8, 8)):
    """单独显示 recovered density 的 XZ 切片。"""
    fig, ax = plt.subplots(figsize=figsize)
    _plot_inverse_density_slice(ax, result, bodies=bodies, slice_y=slice_y)
    plt.tight_layout()
    plt.show()


def plot_gravity_depth_slice(result, mesh_data, depth, figsize=(10, 8)):
    """
    某个深度处的重力异常平面分布（需要多个虚拟钻孔）
    
    这是高级功能的预留接口
    """
    pass  # 后续实现
