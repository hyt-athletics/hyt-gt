"""
第十一步：VTK导出 — 完整演示

创建一个完整模型，然后全部导出为VTK文件
用ParaView打开查看

运行: python run_step11.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh.octree import OctreeMesh
from borehole.well import VerticalWell
from geometry.bodies import make_sphere, make_ellipsoid, make_block
from interactive.body_manager import GeologicalBody
from forward.borehole_gravity import BoreholeGravityCalculator
from export.vtk_export import VTKExporter


def main():
    print("╔══════════════════════════════════════════╗")
    print("║    VTK导出演示 — 全流程                   ║")
    print("╚══════════════════════════════════════════╝")
    
    # ==========================================
    #  1. 定义钻孔
    # ==========================================
    print("\n【1】定义钻孔")
    well = VerticalWell(x=500, y=500, z_top=10, z_bottom=800, n_stations=40)
    well.get_info()
    
    # ==========================================
    #  2. 定义地质体
    # ==========================================
    print("\n【2】定义地质体")
    
    bodies = []
    
    # 矿体1：高密度球
    body1 = GeologicalBody(
        name="高密度球",
        vertices_xz=_sphere_to_poly(600, 400, 80),
        density=0.5,
        y_range=(350, 650),
        color='#e74c3c'
    )
    bodies.append(body1)
    
    # 矿体2：低密度椭球
    body2 = GeologicalBody(
        name="低密度椭球",
        vertices_xz=_ellipse_to_poly(400, 500, 100, 50),
        density=-0.3,
        y_range=(300, 700),
        color='#3498db'
    )
    bodies.append(body2)
    
    # 矿体3：脉状矿体
    body3 = GeologicalBody(
        name="脉状矿体",
        vertices_xz=[(480, 650), (520, 650), (520, 800), (480, 800)],
        density=1.0,
        y_range=(400, 600),
        color='#2ecc71'
    )
    bodies.append(body3)
    
    for b in bodies:
        print(f"  • {b.name}: ρ={b.density} g/cm³, Y={b.y_range}")
    
    # ==========================================
    #  3. 生成网格
    # ==========================================
    print("\n【3】生成八叉树网格")
    
    mesh = OctreeMesh(
        x_range=(0, 1000), y_range=(0, 1000), z_range=(0, 1000),
        max_level=5
    )
    
    mesh.refine_along_borehole(well, levels_and_radii=[
        (5, 50), (4, 150), (3, 300),
    ])
    
    ore1_func = make_sphere(center=(600, 500, 400), radius=80)
    ore2_func = make_ellipsoid(center=(400, 500, 500), semi_axes=(100, 80, 50))
    ore3_func = make_block(bounds=(480, 520, 400, 600, 650, 800))
    
    mesh.refine_at_boundary(ore1_func, target_level=4)
    mesh.refine_at_boundary(ore2_func, target_level=4)
    mesh.refine_at_boundary(ore3_func, target_level=4)
    mesh.balance()
    
    mesh.assign_property(ore1_func, 'density', 0.5)
    mesh.assign_property(ore2_func, 'density', -0.3)
    mesh.assign_property(ore3_func, 'density', 1.0)
    
    mesh.get_info()
    
    # ==========================================
    #  4. 正演计算
    # ==========================================
    print("\n【4】重力正演")
    
    calc = BoreholeGravityCalculator(engine='auto')
    result = calc.compute(well, mesh.to_arrays())
    result.get_summary()
    
    # ==========================================
    #  5. VTK导出
    # ==========================================
    print("\n【5】VTK导出")
    
    exporter = VTKExporter(output_dir='vtk_output')
    
    exporter.export_all(
        octree_mesh=mesh,
        well=well,
        bodies=bodies,
        gravity_result=result,
    )
    
    # ==========================================
    #  6. ParaView操作指南
    # ==========================================
    print_paraview_guide()


def _sphere_to_poly(cx, cz, r, n=30):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + r * np.cos(a), cz + r * np.sin(a)) for a in angles]


def _ellipse_to_poly(cx, cz, a, b, n=30):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + a * np.cos(t), cz + b * np.sin(t)) for t in angles]


def print_paraview_guide():
    """打印ParaView操作指南"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                ParaView 操作指南                                ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  第0步：安装ParaView                                            ║
║    https://www.paraview.org/download/                          ║
║    下载对应系统的版本，安装即可（免费）                            ║
║                                                                ║
║  第1步：打开文件                                                ║
║    File → Open → 选择 vtk_output/model.vtm → OK → Apply        ║
║                                                                ║
║  第2步：查看网格                                                ║
║    左侧 Pipeline Browser 中展开 model.vtm                       ║
║    点击 mesh 旁边的小眼睛 👁 → 显示/隐藏网格                     ║
║                                                                ║
║  第3步：用颜色显示密度                                           ║
║    选中 mesh → 上方工具栏找到 "Solid Color"                      ║
║    → 改成 "density"                                             ║
║    → 颜色映射自动出现                                            ║
║                                                                ║
║  第4步：切片查看内部                                              ║
║    选中 mesh → Filters → Common → Slice                        ║
║    → 在 Properties 中设置切面位置                                ║
║    → Apply                                                     ║
║                                                                ║
║  第5步：只看异常体                                                ║
║    选中 mesh → Filters → Common → Threshold                    ║
║    → Scalars: density                                          ║
║    → 设置 Minimum: 0.01                                        ║
║    → Apply                                                     ║
║    → 只显示有异常的格子                                          ║
║                                                                ║
║  第6步：显示钻孔                                                  ║
║    点击 borehole_path 的眼睛 → 红色线条出现                      ║
║    点击 borehole_stations → 观测点出现                           ║
║                                                                ║
║  第7步：显示地质体                                                ║
║    点击各个 body_xxx 的眼睛 → 半透明的地质体出现                  ║
║                                                                ║
║  第8步：查看重力曲线                                              ║
║    点击 gravity_curve 的眼睛                                     ║
║    → 选择 "gz_mGal" 作为颜色变量                                 ║
║    → 看到一条沿钻孔的3D曲线                                      ║
║                                                                ║
║  常用快捷键:                                                     ║
║    左键拖动 = 旋转                                               ║
║    右键拖动 = 缩放                                               ║
║    中键拖动 = 平移                                               ║
║    R = 重置视角                                                  ║
║    Ctrl+S = 保存截图                                             ║
║                                                                ║
║  高级操作:                                                       ║
║    • Filters → Data Analysis → Plot Over Line                   ║
║      → 沿任意线画剖面                                            ║
║    • Filters → Common → Contour                                 ║
║      → 画等值面                                                  ║
║    • Filters → Common → Clip                                    ║
║      → 切掉一半看内部                                            ║
║    • View → Animation View                                      ║
║      → 制作旋转动画                                              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()