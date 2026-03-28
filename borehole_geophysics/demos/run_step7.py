"""
第七步：3D非结构化四面体网格 — 完整演示

你会看到：
  1. 自研3D Delaunay工作过程
  2. 钻孔周围自适应加密的四面体网格
  3. 异常体内部/边界的网格切面
  4. 网格质量报告

运行方法: python run_step7.py

注意：自研版适合<1000个点，超过1000会自动切换到scipy快速版
"""

import numpy as np
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh.delaunay3d import DelaunayTriangulation3D, DelaunayTriangulation3D_Fast
from mesh.adaptive3d import AdaptivePointGenerator3D
from mesh.quality3d import evaluate_tet_mesh
from borehole.well import VerticalWell
from geometry.bodies import make_sphere, make_ellipsoid
from visualization.viewer3d_tet import (
    show_tet_mesh, show_tet_cross_section,
    show_tet_quality, show_tet_anomaly_body
)


def demo1_basic_3d_delaunay():
    """演示1：最基本的3D Delaunay"""
    print("╔══════════════════════════════════════════╗")
    print("║  演示1：基本3D Delaunay四面体剖分         ║")
    print("╚══════════════════════════════════════════╝")
    
    # 生成随机点
    np.random.seed(42)
    n_points = 100
    points = np.random.rand(n_points, 3) * 500  # 0~500m的随机点
    
    print(f"\n{n_points} 个随机3D点的四面体剖分：")
    
    # 自研版（如果失败则使用scipy快速版）
    dt3d = DelaunayTriangulation3D()
    dt3d.triangulate(points)
    dt3d.get_info()
    
    # 如果自研版失败（四面体数为0），使用scipy快速版
    if len(dt3d.tetrahedra) == 0:
        print("\n  ⚠️ 自研3D Delaunay算法未成功生成四面体")
        print("  → 自动切换到 scipy 快速版...")
        dt3d = DelaunayTriangulation3D_Fast()
        dt3d.triangulate(points)
        dt3d.get_info()
    
    # 质量评估
    quality_data = evaluate_tet_mesh(dt3d)
    
    # 可视化
    print("\n显示3D四面体网格（外表面）...")
    show_tet_mesh(dt3d, show_surface_only=True, opacity=0.5)
    
    return dt3d


def demo2_borehole_adaptive():
    """演示2：钻孔自适应四面体网格（主要演示）"""
    print("\n╔══════════════════════════════════════════╗")
    print("║  演示2：钻孔自适应3D四面体网格             ║")
    print("╚══════════════════════════════════════════╝")
    
    # ==========================================
    # 1. 定义钻孔
    # ==========================================
    well = VerticalWell(
        x=500, y=500,
        z_top=10, z_bottom=800,
        n_stations=40
    )
    well.get_info()
    
    # ==========================================
    # 2. 定义异常体
    # ==========================================
    # 球形矿体
    ore_sphere = make_sphere(center=(620, 500, 400), radius=80)
    
    # 椭球形矿体
    ore_ellipsoid = make_ellipsoid(center=(400, 400, 300), semi_axes=(100, 60, 40))
    
    print("  矿体1: 球形 中心(620,500,400) 半径80m")
    print("  矿体2: 椭球 中心(400,400,300) 半轴(100,60,40)m")
    
    # ==========================================
    # 3. 自适应生成3D点
    # ==========================================
    print("\n【生成自适应点】")
    
    gen = AdaptivePointGenerator3D()
    gen.set_domain(
        x_range=(0, 1000),
        y_range=(0, 1000),
        z_range=(0, 1000)
    )
    
    # 域边界点
    gen.add_domain_boundary_points(spacing=200)
    
    # 背景点（稀疏）
    gen.add_background_points(spacing=200)
    
    # 钻孔渐变加密（关键！）
    gen.refine_along_borehole(
        well,
        min_spacing=30,
        max_spacing=150,
        influence_radius=200,
        n_layers=4
    )
    
    # 矿体表面加密
    gen.refine_around_point(
        center=(620, 500, 400),
        radii_and_counts=[
            (40, 20),   # 内层
            (80, 40),   # 矿体表面
            (120, 30),  # 外层
        ]
    )
    gen.refine_around_point(
        center=(400, 400, 300),
        radii_and_counts=[
            (30, 15),
            (60, 30),
            (100, 25),
        ]
    )
    
    gen.get_info()
    points = gen.get_points()
    print(f"\n  最终点数: {len(points)}")
    
    # ==========================================
    # 4. 执行3D Delaunay四面体剖分
    # ==========================================
    print("\n【3D四面体剖分】")
    
    # 根据点数选择引擎
    if len(points) > 1000:
        print("  点数>1000，使用scipy快速引擎")
        dt3d = DelaunayTriangulation3D_Fast()
    else:
        print("  点数≤1000，使用自研引擎")
        dt3d = DelaunayTriangulation3D()
    
    dt3d.triangulate(points)
    dt3d.get_info()
    
    # 如果四面体数为0，自动切换到scipy快速版
    if len(dt3d.tetrahedra) == 0:
        print("\n  ⚠️ 自研3D Delaunay算法未成功生成四面体")
        print("  → 自动切换到 scipy 快速版...")
        dt3d = DelaunayTriangulation3D_Fast()
        dt3d.triangulate(points)
        dt3d.get_info()
    
    # ==========================================
    # 5. 赋物性
    # ==========================================
    print("\n【赋物性值】")
    
    density = np.zeros(len(dt3d.tetrahedra))
    centers = dt3d.get_all_centers()
    
    for i in range(len(dt3d.tetrahedra)):
        cx, cy, cz = centers[i]
        if ore_sphere(cx, cy, cz):
            density[i] = 0.5   # 高密度矿体
        elif ore_ellipsoid(cx, cy, cz):
            density[i] = -0.3  # 低密度体
    
    n_anomaly = np.sum(density != 0)
    print(f"  有异常的四面体: {n_anomaly} / {len(dt3d.tetrahedra)}")
    
    # ==========================================
    # 6. 质量评估
    # ==========================================
    print("\n【网格质量评估】")
    quality_data = evaluate_tet_mesh(dt3d)
    
    # ==========================================
    # 7. 可视化
    # ==========================================
    print("\n【可视化】")
    
    # 图1：外表面
    print("  显示网格外表面...")
    show_tet_mesh(dt3d, well=well, show_surface_only=True, opacity=0.3)
    
    # 图2：Y方向切面（看内部网格）
    print("  显示Y方向切面...")
    show_tet_cross_section(
        dt3d, normal='y',
        origin=(500, 500, 500),
        property_values=density,
        property_name='density',
        well=well
    )
    
    # 图3：只看异常体
    print("  显示异常体...")
    show_tet_anomaly_body(
        dt3d,
        property_values=density,
        property_name='density',
        threshold=0.01,
        well=well
    )
    
    # 图4：质量分布切面
    print("  显示质量分布...")
    show_tet_quality(dt3d, quality_data)
    
    return dt3d, density


def demo3_compare_engines():
    """演示3：对比自研版和scipy版"""
    print("\n╔══════════════════════════════════════════╗")
    print("║  演示3：自研引擎 vs scipy 对比             ║")
    print("╚══════════════════════════════════════════╝")
    
    np.random.seed(123)
    points = np.random.rand(200, 3) * 1000
    
    # 自研版
    print("\n--- 自研引擎 ---")
    t1 = time.time()
    dt_self = DelaunayTriangulation3D()
    dt_self.triangulate(points)
    t_self = time.time() - t1
    
    # scipy版
    print("\n--- scipy引擎 ---")
    t2 = time.time()
    dt_scipy = DelaunayTriangulation3D_Fast()
    dt_scipy.triangulate(points)
    t_scipy = time.time() - t2
    
    print(f"\n  自研: {len(dt_self.tetrahedra)} tets, {t_self:.2f}秒")
    print(f"  scipy: {len(dt_scipy.tetrahedra)} tets, {t_scipy:.2f}秒")
    print(f"  速度比: scipy快 {t_self / max(t_scipy, 0.001):.0f} 倍")
    
    # 四面体数量应该相同（或非常接近）
    diff = abs(len(dt_self.tetrahedra) - len(dt_scipy.tetrahedra))
    print(f"  四面体数差异: {diff} 个")


def main():
    print("=" * 60)
    print("       3D 非结构化四面体网格 — 完整演示")
    print("=" * 60)
    
    # 演示1：基本3D Delaunay
    demo1_basic_3d_delaunay()
    
    # 演示2：钻孔自适应（重头戏）
    demo2_borehole_adaptive()
    
    # 演示3：引擎对比
    demo3_compare_engines()
    
    print("\n" + "=" * 60)
    print("✅ 3D四面体网格演示全部完成！")
    print("=" * 60)
    print("""
    你已经掌握的网格类型：
      ✅ 结构化六面体（方块切割）
      ✅ 八叉树自适应（聪明地切方块）
      ✅ 2D Delaunay三角形（截面三角化）
      ✅ 约束Delaunay（保护���界）
      ✅ 2.5D三棱柱（三角形拉伸）
      ✅ 3D四面体（真正的非结构化3D）← 当前！
    """)


if __name__ == '__main__':
    main()