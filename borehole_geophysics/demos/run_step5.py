import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
第五步：完整的2.5D三棱柱网格

这是当前阶段的最终成果！你会看到：
  1. 2D自适应三角化 → 截面上钻孔和矿体周围加密
  2. Y方向不等间距拉伸 → 三棱柱网格
  3. 不规则矿体赋物性
  4. 3D可视化
"""

import numpy as np
from mesh.adaptive2d import AdaptiveTriangulation2D
from mesh.prism25d import PrismMesh25D
from mesh.quality import evaluate_mesh_quality
from borehole.well import VerticalWell
from geometry.polygon import make_polygon_body
from geometry.bodies import make_sphere, make_ellipsoid
from visualization.viewer2d import show_triangulation, show_25d_mesh


def main():
    print("╔══════════════════════════════════════════╗")
    print("║   2.5D 三棱柱网格 — 完整流程演示          ║")
    print("╚══════════════════════════════════════════╝")
    
    # ==========================================
    #  1. 定义钻孔
    # ==========================================
    print("\n【第1步】定义钻孔")
    well = VerticalWell(
        x=500, y=500,
        z_top=10, z_bottom=800,
        n_stations=40
    )
    well.get_info()
    
    # ==========================================
    #  2. 定义异常体
    # ==========================================
    print("\n【第2步】定义异常体")
    
    # 不规则矿体（多边形截面）
    ore_polygon = [
        (550, 300), (620, 320), (660, 380),
        (650, 450), (600, 480), (540, 460),
        (510, 400), (520, 340)
    ]
    ore_body = make_polygon_body(ore_polygon, y_range=(350, 650))
    print("  矿体1: 不规则多边形体，Y方向 350~650m")
    
    # 球形异常体
    sphere_body = make_sphere(center=(350, 500, 600), radius=60)
    print("  矿体2: 球形，中心(350,500,600)，半径60m")
    
    # ==========================================
    #  3. 生成2D自适应三角网格
    # ==========================================
    print("\n【第3步】生成2D自适应三角网格")
    
    adaptive = AdaptiveTriangulation2D()
    
    # 矩形域边界
    adaptive.set_rectangular_domain(
        x_range=(0, 1000),
        z_range=(0, 1000),
        n_boundary=12
    )
    
    # 背景点
    adaptive.generate_background_points(
        x_range=(0, 1000),
        z_range=(0, 1000),
        spacing=120
    )
    
    # 钻孔渐变加密
    adaptive.refine_near_line_graded(
        start=(500, 10),
        end=(500, 800),
        min_spacing=20,
        max_spacing=100,
        influence_radius=200,
        n_layers=5
    )
    
    # 钻孔轴线
    adaptive.refine_along_line(start=(500, 0), end=(500, 800), spacing=20)
    
    # 矿体边界加密
    adaptive.refine_at_polygon_boundary(ore_polygon, spacing=20)
    
    # 球体位置加密（截面上的投影）
    adaptive.refine_around_point(
        center=(350, 600),
        radii_and_counts=[(30, 8), (60, 12), (100, 16)]
    )
    
    # 执行三角化
    triangulation = adaptive.build()
    
    # 质量评估
    evaluate_mesh_quality(triangulation)
    
    # 显示2D截面
    print("\n显示2D截面三角网格...")
    show_triangulation(
        triangulation,
        title="2.5D网格的XZ截面（自适应三角化）",
        highlight_boundary=ore_polygon
    )
    
    # ==========================================
    #  4. 设置Y方向分层（不等间距）
    # ==========================================
    print("\n【第4步】设置Y方向分层")
    
    # 关键：Y方向不等间距
    # 钻孔在Y=500处，所以以500为中心，向两边逐渐变疏
    y_layers = np.sort(np.concatenate([
        # 钻孔附近密集
        np.arange(450, 551, 10),         # 450~550, 间距10m
        # 中等距离
        np.arange(350, 450, 25),         # 350~450, 间距25m
        np.arange(550, 651, 25),         # 550~650, 间距25m
        # 远处稀疏
        np.array([0, 50, 100, 200, 300]),   # 0~300, 稀疏
        np.array([700, 800, 900, 1000]),    # 700~1000, 稀疏
    ]))
    y_layers = np.unique(y_layers)  # 去重并排序
    
    # ==========================================
    #  5. 生成2.5D三棱柱网格
    # ==========================================
    print("\n【第5步】生成2.5D三棱柱网格")
    
    prism_mesh = PrismMesh25D()
    prism_mesh.set_triangulation(triangulation)
    prism_mesh.set_y_layers(y_layers)
    prism_mesh.build()
    prism_mesh.get_info()
    
    # ==========================================
    #  6. 赋物性
    # ==========================================
    print("\n【第6步】赋物性值")
    
    prism_mesh.assign_property(ore_body, 'density', 0.5)      # 矿体: 高密度
    prism_mesh.assign_property(sphere_body, 'density', -0.3)   # 球体: 低密度
    
    # 统计
    n_anomaly = np.sum(prism_mesh.density != 0)
    print(f"  有异常的三棱柱: {n_anomaly} / {prism_mesh.n_prisms}")
    
    # ==========================================
    #  7. 3D可视化
    # ==========================================
    print("\n【第7步】3D可视化")
    print("  （弹出3D窗口，可旋转/缩放）")
    
    show_25d_mesh(
        prism_mesh,
        property_name='density',
        well=well,
        threshold=0.01,
        opacity=0.8
    )
    
    # ==========================================
    #  8. 导出数据（供后续正演使用）
    # ==========================================
    print("\n【第8步】导出数据")
    data = prism_mesh.to_arrays()
    print(f"  导出数据:")
    print(f"    节点: {data['nodes'].shape}")
    print(f"    三棱柱: {data['prisms'].shape}")
    print(f"    中心坐标: {data['centers'].shape}")
    print(f"    密度数组: {data['density'].shape}")
    print(f"    总体积: {data['volumes'].sum():.0f} m³")
    
    print("\n✅ 2.5D三棱柱网格完成！")


if __name__ == '__main__':
    main()