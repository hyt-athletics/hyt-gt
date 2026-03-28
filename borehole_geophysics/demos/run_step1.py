import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
第一步：结构化网格 + 可视化

运行方法：在终端输入 python run_step1.py
"""

# 导入我们自己写的模块
from mesh.structured import StructuredMesh
from borehole.well import VerticalWell
from geometry.bodies import make_sphere, make_ellipsoid, make_block
from visualization.viewer import show_structured_mesh


def main():
    # ==========================================
    #  1. 创建网格：1000×1000×1000m，切成20×20×20
    # ==========================================
    print("正在创建结构化网格...")
    
    mesh = StructuredMesh(
        x_range=(0, 1000),    # 东西方向 0~1000m
        y_range=(0, 1000),    # 南北方向 0~1000m
        z_range=(0, 1000),    # 深度方向 0~1000m
        nx=20, ny=20, nz=20   # 每个方向切20块 = 8000个格子
    )
    
    mesh.get_info()
    
    # ==========================================
    #  2. 定义钻孔：在模型中心，垂直打到800m深
    # ==========================================
    print("正在定义钻孔...")
    
    well = VerticalWell(
        x=500, y=500,           # 井口在模型中心
        z_top=10,               # 从10m深开始（避免地表）
        z_bottom=800,           # 打到800m深
        n_stations=40           # 40个观测点（每约20m一个）
    )
    
    well.get_info()
    
    # ==========================================
    #  3. 定义异常体（矿体/地质体）
    # ==========================================
    print("正在定义异常体...")
    
    # 异常体1：一个球形高密度体（模拟球状矿体）
    ore_body_1 = make_sphere(
        center=(600, 500, 400),   # 在钻孔东边100m，深度400m
        radius=80                  # 半径80m
    )
    
    # 异常体2：一个长方体低密度体（模拟空洞或低密度地层）
    ore_body_2 = make_block(
        bounds=(300, 450, 400, 600, 500, 700)  # x,y,z的范围
    )
    
    # 异常体3：一个椭球体（模拟透镜状矿体）
    ore_body_3 = make_ellipsoid(
        center=(500, 300, 300),
        semi_axes=(120, 60, 40)   # 扁椭球
    )
    
    # ==========================================
    #  4. 给异常体赋密度值
    # ==========================================
    print("正在赋物性值...")
    
    mesh.assign_property_to_region(ore_body_1, 'density', 0.5)    # 高密度异常
    mesh.assign_property_to_region(ore_body_2, 'density', -0.3)   # 低密度异常
    mesh.assign_property_to_region(ore_body_3, 'density', 0.8)    # 高密度异常
    
    # 统计
    n_anomaly = np.sum(mesh.density != 0)
    print(f"有异常的格子数: {n_anomaly} / {mesh.n_cells}")
    
    # ==========================================
    #  5. 可视化！
    # ==========================================
    print("正在启动3D可视化...")
    print("（会弹出一个3D窗口，可以用鼠标旋转/缩放）")
    
    # 只显示有异常的区域（threshold=0.01 表示只显示密度差绝对值 > 0.01 的格子）
    show_structured_mesh(
        mesh, 
        property_name='density',
        threshold=0.01,        # 过滤掉背景
        well=well,             # 画上钻孔
        show_edges=True,       
        opacity=0.8
    )


# 需要导入numpy（在assign_property_to_region里用到）
import numpy as np

if __name__ == '__main__':
    main()