"""
第十三步：井地电法 / VSP地震 网格适配演示

演示：
  1. 电法网格：电极处极细加密 + 远边界扩展
  2. 地震网格：波长约束 + 吸收层 + CFL条件
  3. 对比两种方法的网格差异

运行: python run_step13.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from borehole.well import VerticalWell
from geometry.bodies import make_sphere
from interactive.body_manager import GeologicalBody

from config.electrode_config import create_borehole_surface_survey
from config.seismic_config import create_zero_offset_vsp, create_walkaway_vsp

from mesh.electrical_mesh import ElectricalMeshAdapter
from mesh.seismic_mesh import SeismicMeshAdapter

from visualization.survey_viewer import (
    plot_electrode_survey_2d, plot_vsp_survey_2d, plot_mesh_with_survey
)


def demo_electrical():
    """演示1：井地电法网格"""
    print("\n" + "=" * 60)
    print("  演示1：井地电法网格适配")
    print("=" * 60)
    
    # 钻孔
    well = VerticalWell(x=500, y=500, z_top=10, z_bottom=800, n_stations=40)
    
    # 电极排列
    survey = create_borehole_surface_survey(
        well_x=500, well_y=500,
        well_z_top=50, well_z_bottom=700,
        n_borehole_electrodes=14,
        surface_line_x_range=(100, 900),
        n_surface_electrodes=17,
    )
    survey.get_info()
    
    # 画电极分布
    print("  显示电极分布...")
    plot_electrode_survey_2d(survey)
    
    # 异常体（低阻矿体）
    ore = make_sphere(center=(600, 500, 400), radius=80)
    ore_body = GeologicalBody("低阻体",
                              vertices_xz=_sphere_poly(600, 400, 80),
                              density=0, color='#e74c3c')
    ore_body.resistivity = 10.0  # 10 Ω·m（低阻）
    
    # 核心域
    core_domain = {
        'x_range': (0, 1000),
        'y_range': (0, 1000),
        'z_range': (0, 1000),
    }
    
    # 电法适配器
    adapter = ElectricalMeshAdapter(survey)
    adapter.print_requirements()
    
    # 生成网格
    mesh, padded_domain = adapter.generate_adapted_octree(
        core_domain, well=well, bodies=[ore_body]
    )
    
    # 显示网格+电极
    print("  显示网格+电极...")
    plot_mesh_with_survey(mesh, electrode_array=survey, well=well)
    
    return mesh


def demo_seismic():
    """演示2：VSP地震网格"""
    print("\n" + "=" * 60)
    print("  演示2：VSP地震网格适配")
    print("=" * 60)
    
    well = VerticalWell(x=500, y=500, z_top=10, z_bottom=800, n_stations=40)
    
    # VSP观测系统
    vsp = create_zero_offset_vsp(
        well_x=500, well_y=500,
        z_top=50, z_bottom=750,
        n_receivers=30,
        source_offset=100,
        source_frequency=50.0,
    )
    vsp.get_info()
    
    # 画观测系统
    print("  显示VSP观测系统...")
    plot_vsp_survey_2d(vsp)
    
    # 速度模型（层状）
    velocity_model = {
        'v_min': 2500,
        'v_max': 5000,
        'layers': [
            (0, 200, 3000),      # 浅部 3000 m/s
            (200, 500, 4000),    # 中部 4000 m/s
            (500, 1000, 5000),   # 深部 5000 m/s
        ]
    }
    
    core_domain = {
        'x_range': (0, 1000),
        'y_range': (0, 1000),
        'z_range': (0, 1000),
    }
    
    # 地震适配器
    adapter = SeismicMeshAdapter(vsp, velocity_model)
    adapter.print_requirements()
    
    # 生成网格
    mesh, padded_domain = adapter.generate_adapted_octree(
        core_domain, well=well
    )
    
    # 显示
    print("  显示网格+观测系统...")
    plot_mesh_with_survey(mesh, vsp_survey=vsp, well=well)
    
    return mesh


def demo_comparison():
    """演示3：对比电法和地震对网格的不同需求"""
    print("\n" + "=" * 60)
    print("  演示3：电法 vs 地震 网格需求对比")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左：电法（电极处极细，远处很粗）
    axes[0].set_title('电法网格特点\n电极处极细 + 对数增长', fontsize=12)
    
    distances = np.logspace(-1, 3, 100)  # 0.1m ~ 1000m
    cell_sizes_dc = np.minimum(distances * 0.5, 100)  # 线性增长到饱和
    axes[0].loglog(distances, cell_sizes_dc, 'r-', linewidth=2)
    axes[0].set_xlabel('距电极距离 (m)')
    axes[0].set_ylabel('网格尺寸 (m)')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='最小1m')
    axes[0].axhline(y=100, color='green', linestyle='--', alpha=0.5, label='最大100m')
    axes[0].legend()
    
    # 右：地震（有硬性上限，低速区更小）
    axes[1].set_title('地震网格特点\n波长约束 + 速度依赖', fontsize=12)
    
    velocities = np.linspace(1500, 6000, 100)
    f = 50  # Hz
    ppw = 8
    max_sizes_seismic = velocities / f / ppw
    axes[1].plot(velocities, max_sizes_seismic, 'b-', linewidth=2)
    axes[1].fill_between(velocities, 0, max_sizes_seismic, alpha=0.1, color='blue')
    axes[1].set_xlabel('速度 (m/s)')
    axes[1].set_ylabel(f'最大网格尺寸 (m)\n(f={f}Hz, {ppw}点/波长)')
    axes[1].grid(True, alpha=0.3)
    
    # 标注典型速度
    for v, name in [(2000, '土层'), (3500, '砂岩'), (5000, '花岗岩')]:
        h = v / f / ppw
        axes[1].axvline(x=v, color='gray', linestyle=':', alpha=0.5)
        axes[1].annotate(f'{name}\nh≤{h:.0f}m', (v, h),
                        textcoords="offset points", xytext=(10, 10), fontsize=9)
    
    plt.suptitle('电法 vs 地震：网格需求差异', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 打印对比表
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║              电法 vs 地震 网格需求对比                      ║
    ╠═══════════════════════════════════════════════════════════╣
    ║ 特征          │ 电法               │ 地震                ║
    ╠═══════════════╪════════════════════╪════════════════════╣
    ║ 驱动因素      │ 电极奇异性          │ 波长约束             ║
    ║ 最小格子位置  │ 电极处              │ 低速区               ║
    ║ 最小格子尺寸  │ ~1m                │ V_min/f/N ≈ 数米    ║
    ║ 格子尺寸变化  │ 随距离线性/对数增长  │ 随速度线性变化        ║
    ║ 远场需求      │ 大范围padding       │ PML吸收层           ║
    ║ 关键约束      │ 电极在节点上         │ CFL条件             ║
    ║ 时间维度      │ 无(稳态)            │ 有(时间步进)         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


def _sphere_poly(cx, cz, r, n=30):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + r * np.cos(a), cz + r * np.sin(a)) for a in angles]


def main():
    demo_electrical()
    demo_seismic()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("  ✅ 电法/地震网格适配演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()