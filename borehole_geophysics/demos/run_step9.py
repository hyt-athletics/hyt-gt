"""
第九步：重力正演 — 完整演示

演示内容：
  1. 简单模型：一个球形矿体
  2. 复杂模型：多个不规则矿体
  3. 多井对比
  4. 联合显示（模型+曲线）

运行：python run_step9.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置中文字体（必须在导入matplotlib相关模块之前）
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from mesh.octree import OctreeMesh
from borehole.well import VerticalWell
from geometry.bodies import make_sphere, make_ellipsoid, make_block
from forward.borehole_gravity import BoreholeGravityCalculator
from visualization.plot_gravity import (
    plot_borehole_gravity,
    plot_gravity_with_model,
    plot_multi_well_gravity
)
from interactive.body_manager import GeologicalBody


def demo1_single_sphere():
    """演示1：一个球形矿体"""
    print("╔══════════════════════════════════════╗")
    print("║  演示1：球形矿体的钻孔重力响应        ║")
    print("╚══════════════════════════════════════╝")
    
    # 钻孔
    well = VerticalWell(x=500, y=500, z_top=10, z_bottom=900, n_stations=50)
    
    # 网格
    mesh = OctreeMesh(
        x_range=(0, 1000), y_range=(0, 1000), z_range=(0, 1000),
        max_level=5
    )
    
    # 加密
    mesh.refine_along_borehole(well, levels_and_radii=[
        (5, 50), (4, 150), (3, 300),
    ])
    
    # 球形矿体：钻孔东边100m处，深度400m
    ore = make_sphere(center=(600, 500, 400), radius=80)
    mesh.refine_at_boundary(ore, target_level=4)
    mesh.balance()
    mesh.assign_property(ore, 'density', 0.5)
    
    mesh.get_info()
    
    # 正演计算
    calc = BoreholeGravityCalculator(engine='auto')
    result = calc.compute(well, mesh.to_arrays())
    result.get_summary()
    
    # 绘图
    body = GeologicalBody(
        name="球形矿体", density=0.5,
        vertices_xz=_sphere_to_polygon(600, 400, 80),
        color='#e74c3c'
    )
    plot_gravity_with_model(result, bodies=[body])
    
    return result


def demo2_multiple_bodies():
    """演示2：多个异常体"""
    print("\n╔══════════════════════════════════════╗")
    print("║  演示2：多个异常体的复合响应           ║")
    print("╚══════════════════════════════════════╝")
    
    well = VerticalWell(x=500, y=500, z_top=10, z_bottom=900, n_stations=50)
    
    mesh = OctreeMesh(
        x_range=(0, 1000), y_range=(0, 1000), z_range=(0, 1000),
        max_level=5
    )
    
    mesh.refine_along_borehole(well, levels_and_radii=[
        (5, 50), (4, 150), (3, 300),
    ])
    
    # 异常体1：高密度球
    ore1 = make_sphere(center=(600, 500, 300), radius=60)
    mesh.refine_at_boundary(ore1, target_level=4)
    mesh.assign_property(ore1, 'density', 0.8)
    
    # 异常体2：低密度椭球（空洞）
    ore2 = make_ellipsoid(center=(400, 500, 500), semi_axes=(100, 80, 50))
    mesh.refine_at_boundary(ore2, target_level=4)
    mesh.assign_property(ore2, 'density', -0.4)
    
    # 异常体3：高密度长方体（脉状矿体）
    ore3 = make_block(bounds=(480, 520, 400, 600, 650, 800))
    mesh.refine_at_boundary(ore3, target_level=4)
    mesh.assign_property(ore3, 'density', 1.0)
    
    mesh.balance()
    mesh.get_info()
    
    # 正演
    calc = BoreholeGravityCalculator()
    result = calc.compute(well, mesh.to_arrays())
    result.get_summary()
    
    # 绘图
    bodies = [
        GeologicalBody("高密度球", _sphere_to_polygon(600, 300, 60),
                       density=0.8, color='#e74c3c'),
        GeologicalBody("低密度椭球", _ellipse_to_polygon(400, 500, 100, 50),
                       density=-0.4, color='#3498db'),
        GeologicalBody("脉状矿体", [(480, 650), (520, 650), (520, 800), (480, 800)],
                       density=1.0, color='#2ecc71'),
    ]
    plot_gravity_with_model(result, bodies=bodies)
    
    return result


def demo3_multi_well():
    """演示3：多井对比"""
    print("\n╔══════════════════════════════════════╗")
    print("║  演示3：多口钻孔对比                  ║")
    print("╚══════════════════════════════════════╝")
    
    # 一个矿体
    mesh = OctreeMesh(
        x_range=(0, 1000), y_range=(0, 1000), z_range=(0, 1000),
        max_level=5
    )
    ore = make_sphere(center=(500, 500, 400), radius=80)
    
    # 3口不同位置的钻孔
    wells = [
        VerticalWell(x=500, y=500, z_top=10, z_bottom=900, n_stations=50),  # 正上方
        VerticalWell(x=600, y=500, z_top=10, z_bottom=900, n_stations=50),  # 偏东100m
        VerticalWell(x=700, y=500, z_top=10, z_bottom=900, n_stations=50),  # 偏东200m
    ]
    
    # 为每口钻孔加密
    for well in wells:
        mesh.refine_along_borehole(well, levels_and_radii=[
            (5, 50), (4, 150),
        ])
    
    mesh.refine_at_boundary(ore, target_level=4)
    mesh.balance()
    mesh.assign_property(ore, 'density', 0.5)
    mesh.get_info()
    
    # 正演
    calc = BoreholeGravityCalculator()
    mesh_data = mesh.to_arrays()
    
    results = []
    for i, well in enumerate(wells):
        print(f"\n--- 钻孔 {i+1}: x={well.x} ---")
        r = calc.compute(well, mesh_data)
        r.get_summary()
        results.append(r)
    
    # 多井对比图
    plot_multi_well_gravity(results)
    
    return results


def demo4_sensitivity():
    """演示4：验证——矿体越远，异常越小"""
    print("\n╔══════════════════════════════════════╗")
    print("║  演示4：距离衰减特性                  ║")
    print("╚══════════════════════════════════════╝")
    
    well = VerticalWell(x=500, y=500, z_top=10, z_bottom=900, n_stations=50)
    
    distances = [50, 100, 150, 200, 300]
    results = []
    
    for dist in distances:
        print(f"\n--- 矿体距钻孔 {dist}m ---")
        
        mesh = OctreeMesh(
            x_range=(0, 1000), y_range=(0, 1000), z_range=(0, 1000),
            max_level=4
        )
        
        mesh.refine_along_borehole(well, levels_and_radii=[(4, 100), (3, 250)])
        
        ore = make_sphere(center=(500 + dist, 500, 400), radius=50)
        mesh.refine_at_boundary(ore, target_level=4)
        mesh.balance()
        mesh.assign_property(ore, 'density', 0.5)
        
        calc = BoreholeGravityCalculator()
        r = calc.compute(well, mesh.to_arrays())
        results.append(r)
        
        print(f"  最大异常: {r.max_anomaly:.4f} mGal")
    
    # 画对比图
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # 左图：所有曲线
    for i, (dist, r) in enumerate(zip(distances, results)):
        ax1.plot(r.gz, r.depths, linewidth=2, label=f'd={dist}m')
    
    ax1.set_xlabel('gz (mGal)', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('矿体在不同距离时的钻孔重力响应', fontsize=13)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.legend(fontsize=10)
    
    # 右图：峰值异常 vs 距离
    peak_values = [np.max(np.abs(r.gz)) for r in results]
    ax2.semilogy(distances, peak_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('矿体到钻孔的距离 (m)', fontsize=12)
    ax2.set_ylabel('最大异常绝对值 (mGal)', fontsize=12)
    ax2.set_title('异常峰值随距离的衰减', fontsize=13)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 标注理论衰减趋势
    d_theory = np.linspace(50, 300, 100)
    scale = peak_values[0] * distances[0]**2
    theory = scale / d_theory**2
    ax2.semilogy(d_theory, theory, 'b--', alpha=0.5, label='~1/r² 理论趋势')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# ===== 辅助函数 =====

def _sphere_to_polygon(cx, cz, r, n=30):
    """把球体截面转成多边形顶点"""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + r * np.cos(a), cz + r * np.sin(a)) for a in angles]


def _ellipse_to_polygon(cx, cz, a, b, n=30):
    """把椭圆截面转成多边形顶点"""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + a * np.cos(t), cz + b * np.sin(t)) for t in angles]


def main():
    demo1_single_sphere()
    demo2_multiple_bodies()
    demo3_multi_well()
    demo4_sensitivity()
    
    print("\n" + "=" * 50)
    print("  ✅ 重力正演演示完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()