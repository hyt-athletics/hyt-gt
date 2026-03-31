"""
三分量井中磁测可视化
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D


def _plot_component(ax, values, depths, label, color):
    ax.plot(values, depths, color=color, linewidth=2)
    ax.fill_betweenx(depths, 0, values, where=(values >= 0), color='#e74c3c', alpha=0.18)
    ax.fill_betweenx(depths, 0, values, where=(values < 0), color='#3498db', alpha=0.18)
    ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.set_title(label, fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_borehole_magnetic(result, title=None, figsize=(12, 10)):
    """绘制 Bx/By/Bz/ΔT 四分量随深度变化。"""
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    components = [
        (result.bx, 'Bx (nT)', '#1f77b4'),
        (result.by, 'By (nT)', '#2ca02c'),
        (result.bz, 'Bz (nT)', '#ff7f0e'),
        (result.bt, 'ΔT (nT)', '#d62728'),
    ]

    for ax, (values, label, color) in zip(axes, components):
        _plot_component(ax, values, result.depths, label, color)
        ax.set_xlabel(label, fontsize=10)

    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].invert_yaxis()

    peak_idx = np.argmax(np.abs(result.bt))
    axes[-1].plot(result.bt[peak_idx], result.depths[peak_idx], 'r*', markersize=14, zorder=10)

    fig.suptitle(
        title or f'3C Borehole Magnetic Survey - ({result.well.x:.0f}, {result.well.y:.0f})',
        fontsize=14
    )
    plt.tight_layout()
    plt.show()


def plot_magnetic_with_model(result, bodies=None, figsize=(16, 10)):
    """左侧模型、右侧四条磁法曲线。"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 5, width_ratios=[2.2, 1, 1, 1, 1])
    ax_model = fig.add_subplot(gs[0, 0])
    curve_axes = [fig.add_subplot(gs[0, i], sharey=ax_model) for i in range(1, 5)]

    ax_model.set_xlabel('X (m)', fontsize=12)
    ax_model.set_ylabel('Depth (m)', fontsize=12)
    ax_model.set_title('XZ Cross Section (Y=well)', fontsize=13)
    ax_model.invert_yaxis()
    ax_model.grid(True, alpha=0.3)
    ax_model.set_aspect('equal')

    if bodies:
        for body in bodies:
            polygon = MplPolygon(
                body.vertices_xz,
                closed=True,
                facecolor=body.color,
                alpha=0.35,
                edgecolor=body.color,
                linewidth=2,
            )
            ax_model.add_patch(polygon)
            cx, cz = body.centroid_2d()
            lines = [body.name]
            if abs(body.density) > 1e-12:
                lines.append(f'ρ={body.density:.2f}')
            if abs(body.susceptibility) > 1e-12:
                lines.append(f'χ={body.susceptibility:.3f}')
            ax_model.text(
                cx, cz, '\n'.join(lines),
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor=body.color, alpha=0.7),
            )

    well = result.well
    ax_model.axvline(x=well.x, color='red', linewidth=2.5, zorder=10)
    ax_model.plot(well.stations[:, 0], well.stations[:, 2], 'r.', markersize=4, zorder=11)

    if result.mesh_data is not None:
        centers = result.mesh_data['centers']
        ax_model.set_xlim(centers[:, 0].min() - 50, centers[:, 0].max() + 50)

    components = [
        (result.bx, 'Bx (nT)', '#1f77b4'),
        (result.by, 'By (nT)', '#2ca02c'),
        (result.bz, 'Bz (nT)', '#ff7f0e'),
        (result.bt, 'ΔT (nT)', '#d62728'),
    ]
    for ax, (values, label, color) in zip(curve_axes, components):
        _plot_component(ax, values, result.depths, label, color)
        ax.set_xlabel(label, fontsize=10)
        ax.tick_params(axis='y', labelleft=False)

    peak_idx = np.argmax(np.abs(result.bt))
    peak_depth = result.depths[peak_idx]
    for ax in [ax_model] + curve_axes:
        ax.axhline(peak_depth, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    curve_axes[-1].plot(result.bt[peak_idx], peak_depth, 'r*', markersize=14, zorder=10)

    fig.suptitle('3C Borehole Magnetic Survey', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_magnetic_comparison(results, figsize=(14, 10)):
    """对比多次磁法结果，默认对比 ΔT。"""
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
    components = [
        ('bx', 'Bx (nT)'),
        ('by', 'By (nT)'),
        ('bz', 'Bz (nT)'),
        ('bt', 'ΔT (nT)'),
    ]

    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        label = getattr(result, 'label', f'run_{idx + 1}')
        for ax, (attr, title) in zip(axes, components):
            ax.plot(getattr(result, attr), result.depths, linewidth=2, color=color, label=label)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')

    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].invert_yaxis()
    for ax, (_, title) in zip(axes, components):
        ax.set_xlabel(title, fontsize=10)
    axes[-1].legend(loc='best', fontsize=9)

    fig.suptitle('3C Borehole Magnetic Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_magnetic_model_vs_observed(modeled, observed, figsize=(14, 10)):
    """当前正演结果与实测磁法曲线对比。"""
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    components = [
        ('bx', 'Bx (nT)', '#1f77b4'),
        ('by', 'By (nT)', '#2ca02c'),
        ('bz', 'Bz (nT)', '#ff7f0e'),
        ('bt', 'ΔT (nT)', '#d62728'),
    ]

    observed_interp = {}
    for attr, _, _ in components:
        observed_interp[attr] = np.interp(
            modeled.depths, observed.depths, getattr(observed, attr)
        )

    for ax, (attr, title, color) in zip(axes, components):
        ax.plot(getattr(modeled, attr), modeled.depths, linewidth=2.1, color=color,
                label=getattr(modeled, 'label', 'modeled'))
        ax.plot(getattr(observed, attr), observed.depths, 'o--', linewidth=1.5,
                markersize=3.5, color='black',
                label=getattr(observed, 'label', 'observed'))
        rmse = float(np.sqrt(np.mean((getattr(modeled, attr) - observed_interp[attr]) ** 2)))
        ax.set_title(f'{title}\nRMSE={rmse:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xlabel(title, fontsize=10)

    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].invert_yaxis()
    axes[-1].legend(loc='best', fontsize=9)

    fig.suptitle('3C Magnetic: Modeled vs Observed', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_magnetic_inversion_result(result, figsize=(16, 10)):
    """显示磁法反演的 3C/ΔT 拟合、恢复磁化率切片和直方图。"""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], height_ratios=[1, 1.1])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
    ]
    ax_slice = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[1, 2])

    components = [
        ('observed_bx', 'predicted_bx', 'Bx (nT)', '#1f77b4'),
        ('observed_by', 'predicted_by', 'By (nT)', '#2ca02c'),
        ('observed_bz', 'predicted_bz', 'Bz (nT)', '#ff7f0e'),
        ('observed_bt', 'predicted_bt', 'ΔT (nT)', '#d62728'),
    ]
    for ax, (obs_attr, pred_attr, title, color) in zip(axes, components):
        ax.plot(getattr(result, obs_attr), result.depths, 'o--', color='black',
                markersize=3.2, linewidth=1.3, label='observed')
        ax.plot(getattr(result, pred_attr), result.depths, '-', color=color,
                linewidth=2.0, label='predicted')
        rmse = float(np.sqrt(np.mean((getattr(result, pred_attr) - getattr(result, obs_attr)) ** 2)))
        ax.set_title(f'{title}\nRMSE={rmse:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xlabel(title, fontsize=10)
    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].invert_yaxis()
    axes[-1].set_ylabel('Depth (m)', fontsize=12)
    axes[-1].invert_yaxis()
    axes[-1].legend(loc='best', fontsize=8)

    _plot_inverse_susceptibility_slice(ax_slice, result, bodies=None, slice_y=result.well.y)

    susceptibility = np.asarray(result.recovered_susceptibility, dtype=float)
    update_mask = result.recovered_mesh_data.get('update_mask')
    fixed_mask = result.recovered_mesh_data.get('fixed_mask')
    if update_mask is not None and fixed_mask is not None:
        ax_hist.hist(susceptibility[np.asarray(fixed_mask, dtype=bool)], bins=30,
                     color='#b0b0b0', alpha=0.65, edgecolor='white', label='fixed')
        ax_hist.hist(susceptibility[np.asarray(update_mask, dtype=bool)], bins=30,
                     color='#2ca02c', alpha=0.8, edgecolor='white', label='updated')
        ax_hist.legend(loc='best', fontsize=8)
    else:
        ax_hist.hist(susceptibility, bins=30, color='#2ca02c', alpha=0.85, edgecolor='white')
    ax_hist.set_xlabel('Recovered susceptibility (SI)', fontsize=11)
    ax_hist.set_ylabel('Cell count', fontsize=11)
    ax_hist.set_title('Recovered Model Histogram', fontsize=12)
    ax_hist.grid(True, alpha=0.2)

    fig.suptitle(
        f'3C Magnetic Inversion\nΔT RMSE={result.rmse_bt:.3f} nT',
        fontsize=14
    )
    plt.tight_layout()
    plt.show()


def _extract_susceptibility_xz_slice(mesh_data, slice_y=None):
    centers = np.asarray(mesh_data['centers'], dtype=float)
    sizes = np.asarray(mesh_data['sizes'], dtype=float)
    susceptibility = np.asarray(mesh_data['susceptibility'], dtype=float)
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
    slice_values = susceptibility[mask]
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

    for center, value in zip(slice_centers, slice_values):
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


def _plot_inverse_susceptibility_slice(ax, result, bodies=None, slice_y=None):
    grid, extent, actual_y, update_grid, fixed_grid = _extract_susceptibility_xz_slice(
        result.recovered_mesh_data, slice_y=slice_y
    )
    image = ax.imshow(
        grid,
        extent=extent,
        origin='upper',
        aspect='auto',
        cmap='YlGnBu',
    )
    update_mode = result.metadata.get('update_mode')
    title = f'Recovered Susceptibility Slice\nY={actual_y:.1f} m'
    if update_mode:
        title += f'  [{update_mode}]'
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
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
                edgecolor=body.color, linewidth=1.4
            )
            ax.add_patch(polygon)

    ax.axvline(result.well.x, color='black', linewidth=1.8, linestyle='--')
    ax.plot(result.well.stations[:, 0], result.well.stations[:, 2], 'k.', markersize=3, zorder=10)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Susceptibility (SI)')


def plot_magnetic_inverse_susceptibility_slice(result, bodies=None, slice_y=None, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    _plot_inverse_susceptibility_slice(ax, result, bodies=bodies, slice_y=slice_y)
    plt.tight_layout()
    plt.show()
