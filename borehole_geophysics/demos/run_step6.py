import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
第六步：约束Delaunay三角化演示

你会看到：
  1. 普通Delaunay如何"破坏"边界
  2. 约束Delaunay如何保护边界
  3. 多层地质体（地层线不被穿过）
  4. 不规则矿体的精确边界
"""

import numpy as np
from mesh.constrained_delaunay import ConstrainedDelaunayTriangulation
from mesh.delaunay2d import DelaunayTriangulation2D
from geometry.constraints import ConstraintBuilder, polygon_to_constraints
from visualization.viewer2d import (compare_delaunay_vs_constrained,
                                     show_constrained_triangulation)


def demo_basic_constraint():
    """演示1：基本约束——保护一个正方形"""
    print("=" * 50)
    print("演示1：基本约束 - 正方形边界")
    print("=" * 50)
    
    # 生成随机点
    np.random.seed(42)
    background_points = np.random.rand(20, 2) * 1000
    
    # 定义正方形（约束）
    square = np.array([
        [300, 300],
        [700, 300],
        [700, 700],
        [300, 700]
    ])
    
    # 合并点集
    all_points = np.vstack([background_points, square])
    
    # 约束边索引（正方形的4条边）
    constraints = [
        (20, 21), (21, 22), (22, 23), (23, 20)
    ]
    
    print("对比普通Delaunay和约束Delaunay...")
    compare_delaunay_vs_constrained(all_points, constraints)


def demo_irregular_ore_body():
    """演示2：不规则矿体边界保护"""
    print("=" * 50)
    print("演示2：不规则矿体边界")
    print("=" * 50)
    
    builder = ConstraintBuilder()
    
    # 背景点
    for x in np.linspace(0, 1000, 8):
        for z in np.linspace(0, 1000, 8):
            builder.add_point(x, z)
    
    # 不规则矿体（多边形）
    ore_vertices = [
        (550, 400), (620, 420), (680, 480),
        (670, 560), (610, 600), (540, 580),
        (500, 520), (520, 440)
    ]
    builder.add_polygon(ore_vertices, label="ore")
    
    # 矿体内部再加密一些点
    for x in np.linspace(550, 650, 4):
        for z in np.linspace(450, 550, 4):
            builder.add_point(x, z)
    
    points, constraints = builder.build()
    
    print(f"  点数: {len(points)}")
    print(f"  约束边: {len(constraints)}")
    
    # 约束三角化
    cdt = ConstrainedDelaunayTriangulation()
    cdt.triangulate(points, constraints)
    cdt.get_info()
    
    # 可视化
    show_constrained_triangulation(
        cdt,
        title="不规则矿体 - 约束Delaunay（红色为矿体边界）",
        highlight_constraints=True
    )


def demo_layered_strata():
    """演示3：多层地质体（地层线不被穿过）"""
    print("=" * 50)
    print("演示3：多层地质体")
    print("=" * 50)
    
    builder = ConstraintBuilder()
    
    # 背景点
    np.random.seed(123)
    for _ in range(50):
        x = np.random.uniform(0, 1000)
        z = np.random.uniform(0, 1000)
        builder.add_point(x, z)
    
    # 定义地层分界线（波动的曲线）
    def layer_z(x, base_z, amplitude=50):
        return base_z + amplitude * np.sin(x / 150)
    
    # 三层地层 = 两个分界线
    layer1_points = []
    layer2_points = []
    
    for x in np.linspace(0, 1000, 30):
        z1 = layer_z(x, 300)
        z2 = layer_z(x, 600)
        layer1_points.append((x, z1))
        layer2_points.append((x, z2))
    
    builder.add_polyline(layer1_points, label="layer1")
    builder.add_polyline(layer2_points, label="layer2")
    
    # 添加左右边界（让地层线闭合）
    builder.add_constraint_edge_by_labels("layer1"[0], "layer1"[-1])
    
    points, constraints = builder.build()
    
    print(f"  点数: {len(points)}")
    print(f"  约束边: {len(constraints)}")
    
    # 约束三角化
    cdt = ConstrainedDelaunayTriangulation()
    cdt.triangulate(points, constraints)
    cdt.get_info()
    
    # 可视化
    show_constrained_triangulation(
        cdt,
        title="多层地质体 - 地层分界线保护",
        highlight_constraints=True
    )


def demo_borehole_with_boundary():
    """演示4：钻孔 + 矿体（综合演示）"""
    print("=" * 50)
    print("演示4：钻孔 + 矿体边界保护")
    print("=" * 50)
    
    builder = ConstraintBuilder()
    
    # 矩形域边界
    builder.add_polygon([
        (0, 0), (1000, 0), (1000, 1000), (0, 1000)
    ], label="domain")
    
    # 背景点
    for x in np.linspace(100, 900, 6):
        for z in np.linspace(100, 900, 6):
            builder.add_point(x, z)
    
    # 钻孔轨迹（约束线）
    borehole_points = []
    for z in np.linspace(0, 800, 25):
        x = 500 + 20 * np.sin(z / 200)  # 略微弯曲的钻孔
        borehole_points.append((x, z))
    builder.add_polyline(borehole_points, label="borehole")
    
    # 钻孔周围加密
    for i, (x, z) in enumerate(borehole_points):
        for r in [30, 60]:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                px = x + r * np.cos(angle)
                pz = z + r * np.sin(angle)
                if 10 < px < 990 and 10 < pz < 990:
                    builder.add_point(px, pz)
    
    # 不规则矿体
    ore_vertices = [
        (550, 350), (650, 380), (700, 450),
        (680, 540), (600, 580), (520, 550),
        (480, 480), (500, 400)
    ]
    builder.add_polygon(ore_vertices, label="ore")
    
    points, constraints = builder.build()
    
    print(f"  点数: {len(points)}")
    print(f"  约束边: {len(constraints)}")
    
    # 对比演示
    print("\n生成对比图（普通vs约束）...")
    compare_delaunay_vs_constrained(points, constraints)
    
    # 单独显示约束结果
    cdt = ConstrainedDelaunayTriangulation()
    cdt.triangulate(points, constraints)
    
    show_constrained_triangulation(
        cdt,
        title="钻孔轨迹 + 矿体边界（双重约束）",
        highlight_constraints=True
    )


def main():
    demo_basic_constraint()
    demo_irregular_ore_body()
    demo_layered_strata()
    demo_borehole_with_boundary()
    
    print("\n✅ 所有约束Delaunay演示完成！")


if __name__ == '__main__':
    main()