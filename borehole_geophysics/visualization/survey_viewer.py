"""
观测系统可视化

画出电极/震源/检波器的位置
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_electrode_survey_2d(electrode_array, ax=None, figsize=(12, 8)):
    """画电极排列的XZ截面"""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False
    
    for e in electrode_array.electrodes:
        if e.is_source:
            ax.plot(e.x, e.z, 'r^', markersize=10, zorder=10, label='Source' if e.type == 'A' else '')
        else:
            ax.plot(e.x, e.z, 'bv', markersize=8, zorder=10, label='Receiver' if not any(m.type == 'M' for m in electrode_array.electrodes[:electrode_array.electrodes.index(e)]) else '')
    
    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m, depth)')
    ax.set_title(f'Electrode Survey: {electrode_array.name}')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_vsp_survey_2d(vsp_survey, ax=None, figsize=(12, 8)):
    """画VSP观测系统的XZ截面"""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False
    
    for src in vsp_survey.sources:
        ax.plot(src.x, src.z, 'r*', markersize=15, zorder=10)
        ax.annotate(src.label, (src.x, src.z), textcoords="offset points",
                   xytext=(5, 5), fontsize=8, color='red')
    
    for rec in vsp_survey.receivers:
        ax.plot(rec.x, rec.z, 'b>', markersize=6, zorder=10)
    
    # 画射线路径示意
    for src in vsp_survey.sources:
        for rec in vsp_survey.receivers[::max(1, len(vsp_survey.receivers)//5)]:
            ax.plot([src.x, rec.x], [src.z, rec.z],
                   'g-', alpha=0.1, linewidth=0.5)
    
    ax.plot([], [], 'r*', markersize=15, label='Source')
    ax.plot([], [], 'b>', markersize=6, label='Receiver')
    ax.plot([], [], 'g-', alpha=0.3, label='Ray path')
    ax.legend()
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m, depth)')
    ax.set_title(f'VSP Survey: {vsp_survey.name}')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_mesh_with_survey(mesh, electrode_array=None, vsp_survey=None,
                           well=None, figsize=(14, 10)):
    """在网格上叠加观测系统"""
    from visualization.viewer import show_octree_wireframe
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 画网格（XZ切面 at y=well_y）
    leaves = mesh.root.get_leaves()
    well_y = well.y if well else 500
    tolerance = 50
    
    for leaf in leaves:
        if abs(leaf.center[1] - well_y) < tolerance:
            rect = plt.Rectangle(
                (leaf.x_min, leaf.z_min),
                leaf.x_max - leaf.x_min,
                leaf.z_max - leaf.z_min,
                fill=False, edgecolor='gray', linewidth=0.3, alpha=0.5
            )
            ax.add_patch(rect)
    
    # 画电极
    if electrode_array is not None:
        plot_electrode_survey_2d(electrode_array, ax=ax)
    
    # 画VSP
    if vsp_survey is not None:
        plot_vsp_survey_2d(vsp_survey, ax=ax)
    
    # 画钻孔
    if well is not None:
        ax.axvline(x=well.x, color='red', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m, depth)')
    ax.set_title('Mesh + Survey Configuration')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()