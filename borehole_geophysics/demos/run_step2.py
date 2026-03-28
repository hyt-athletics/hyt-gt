import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
第二步：八叉树自适应网格 + 可视化

这是最关键的一步！你将看到：
  - 钻孔旁边的网格变密
  - 异常体边界处的网格变密
  - 远处的网格保持粗大

运行方法：python run_step2.py
"""

from mesh.octree import OctreeMesh
from borehole.well import VerticalWell
from geometry.bodies import make_sphere, make_ellipsoid, make_block
from visualization.viewer import show_octree_wireframe, show_octree_mesh


def main():
    # ==========================================
    #  1. 创建八叉树网格
    # ==========================================
    print("="*50)
    print("第一步：创建八叉树网格")
    print("="*50)
    
    mesh = OctreeMesh(
        x_range=(0, 1000),
        y_range=(0, 1000),
        z_range=(0, 1000),
        max_level=6    # 最小格子 = 1000/2^6 ≈ 15.6m
    )
    
    print(f"初始状态：{mesh.n_cells} 个格子（就一个大方块）")
    
    # ==========================================
    #  2. 定义钻孔
    # ==========================================
    well = VerticalWell(
        x=500, y=500,
        z_top=10, z_bottom=800,
        n_stations=40
    )
    well.get_info()
    
    # ==========================================
    #  3. 沿钻孔加密（由内到外，逐层加密）
    # ==========================================
    print("="*50)
    print("第二步：沿钻孔加密")
    print("="*50)
    
    mesh.refine_along_borehole(well, levels_and_radii=[
        (3, 500),   # 500m内：加密到层级3 (~125m格子)
        (4, 200),   # 200m内：加密到层级4 (~62m格子)
        (5, 80),    # 80m内：加密到层级5 (~31m格子)
        (6, 30),    # 30m内：加密到层级6 (~15m格子)
    ])
    
    print(f"钻孔加密后：{mesh.n_cells} 个格子")
    
    # ==========================================
    #  4. 定义异常体并在边界加密
    # ==========================================
    print("="*50)
    print("第三步：定义异常体并在边界加密")
    print("="*50)
    
    # 球形矿体
    ore_sphere = make_sphere(center=(600, 500, 400), radius=80)
    mesh.refine_at_boundary(ore_sphere, target_level=5)
    print(f"球体边界加密后：{mesh.n_cells} 个格子")
    
    # 椭球形矿体
    ore_ellipsoid = make_ellipsoid(center=(400, 400, 300), semi_axes=(100, 60, 40))
    mesh.refine_at_boundary(ore_ellipsoid, target_level=5)
    print(f"椭球边界加密后：{mesh.n_cells} 个格子")
    
    # ==========================================
    #  5. 平衡网格
    # ==========================================
    print("="*50)
    print("第四步：平衡网格")
    print("="*50)
    
    mesh.balance()
    print(f"平衡后：{mesh.n_cells} 个格子")
    
    # ==========================================
    #  6. 赋物性
    # ==========================================
    print("="*50)
    print("第五步：赋物性值")
    print("="*50)
    
    mesh.assign_property(ore_sphere, 'density', 0.5)
    mesh.assign_property(ore_ellipsoid, 'density', -0.3)
    
    # ==========================================
    #  7. 打印信息
    # ==========================================
    mesh.get_info()
    
    # ==========================================
    #  8. 可视化
    # ==========================================
    print("="*50)
    print("第六步：可视化")
    print("="*50)
    
    # 图1：只看网格线框（查看疏密分布）
    print("显示网格线框...")
    show_octree_wireframe(mesh, well=well)
    
    # 图2：看物性分布
    print("显示物性分布...")
    show_octree_mesh(mesh, property_name='density', 
                     well=well, threshold=0.01)


if __name__ == '__main__':
    main()