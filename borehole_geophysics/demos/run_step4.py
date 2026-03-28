"""
第四步：约束Delaunay三角化 + 可视化

这是关键的一步！你将看到：
  - 普通Delaunay三角化
  - 强制某些边（约束边）出现在网格中
  - 约束边恢复算法

运行方法：python run_step4.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mesh.constrained_delaunay import ConstrainedDelaunayTriangulation
from visualization.viewer import show_delaunay_mesh


def main():
    # ==========================================
    # 1. 定义点集
    # ==========================================
    print("=" * 50)
    print("第一步：定义点集")
    print("=" * 50)
    
    points = [
        (0, 0),
        (2, 0),
        (4, 0),
        (6, 0),
        (8, 0),
        (1, 2),
        (3, 2),
        (5, 2),
        (7, 2),
        (2, 4),
        (4, 4),
        (6, 4),
        (3, 6),
        (5, 6),
        (4, 8),
    ]
    
    print(f"  点数: {len(points)}")
    print(f"  X范围: {min(p[0] for p in points):.1f} ~ {max(p[0] for p in points):.1f}")
    print(f"  Z范围: {min(p[1] for p in points):.1f} ~ {max(p[1] for p in points):.1f}")
    
    # ==========================================
    # 2. 定义约束边
    # ==========================================
    print("\n第二步：定义约束边")
    print("=" * 50)
    
    constraints = [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 0),
        (2, 6),
        (6, 10),
        (10, 14),
        (14, 2),
    ]
    
    print(f"  约束边数: {len(constraints)}")
    print("  约束边：")
    for i, (a, b) in enumerate(constraints):
        print(f"    {i+1}. ({a}, {b})")
    
    # ==========================================
    # 3. 执行约束Delaunay三角化
    # ==========================================
    print("\n第三步：执行约束Delaunay三角化")
    print("=" * 50)
    
    cdt = ConstrainedDelaunayTriangulation()
    cdt.triangulate(points, constraints)
    
    cdt.get_info()
    
    # ==========================================
    # 4. 验证约束边
    # ==========================================
    print("\n第四步：验证约束边")
    print("=" * 50)
    
    all_present = True
    for i, (a, b) in enumerate(constraints):
        found = False
        for tri in cdt.triangles:
            edges = [
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[2], tri[0])
            ]
            for (edge_a, edge_b) in edges:
                if (edge_a == a and edge_b == b) or (edge_a == b and edge_b == a):
                    found = True
                    break
            if found:
                break
        
        status = "✅" if found else "❌"
        print(f"  约束边 ({a}, {b}): {status}")
        if not found:
            all_present = False
    
    if all_present:
        print("\n  ✅ 所有约束边都已成功插入！")
    else:
        print("\n  ⚠️ 部分约束边插入失败")
    
    # ==========================================
    # 5. 可视化
    # ==========================================
    print("\n第五步：可视化")
    print("=" * 50)
    
    show_delaunay_mesh(
        points=cdt.points,
        triangles=cdt.triangles,
        constraints=constraints,
        title="约束Delaunay三角化"
    )
    
    print("\n" + "=" * 50)
    print("  完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()
