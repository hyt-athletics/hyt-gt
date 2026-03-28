"""
第十四步：性能优化演示

内容:
  1. 基准测试：对比 Python循环 / NumPy / Numba
  2. 实际案例：优化前后的计算时间对比
  3. 大规模模型测试

运行: python run_step14.py
提示: pip install numba  可以获得最大加速
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_benchmark():
    """基准测试"""
    from acceleration.benchmark import run_gravity_benchmark, run_geometry_benchmark
    
    print("╔══════════════════════════════════════════╗")
    print("║       性能基准测试                        ║")
    print("╚══════════════════════════════════════════╝")
    
    # 重力正演基准
    run_gravity_benchmark(
        n_obs_list=[20, 50],
        n_prism_list=[100, 500, 2000]
    )
    
    # 几何计算基准
    run_geometry_benchmark()


def demo_real_case():
    """实际案例对比"""
    from mesh.octree import OctreeMesh
    from borehole.well import VerticalWell
    from geometry.bodies import make_sphere
    from forward.gravity_engine import forward_gravity
    
    print("\n╔══════════════════════════════════════════╗")
    print("║       实际案例 — 优化前后对比              ║")
    print("╚══════════════════════════════════════════╝")
    
    well = VerticalWell(x=500, y=500, z_top=10, z_bottom=800, n_stations=50)
    
    mesh = OctreeMesh(
        x_range=(0, 1000), y_range=(0, 1000), z_range=(0, 1000),
        max_level=5
    )
    mesh.refine_along_borehole(well, levels_and_radii=[
        (5, 50), (4, 150), (3, 300),
    ])
    ore = make_sphere(center=(600, 500, 400), radius=80)
    mesh.refine_at_boundary(ore, target_level=4)
    mesh.balance()
    mesh.assign_property(ore, 'density', 0.5)
    
    data = mesh.to_arrays()
    obs = well.stations
    
    print(f"\n  模型规模: {data['n_cells']} 个格子, {len(obs)} 个观测点")
    n_active = np.sum(np.abs(data['density']) > 1e-10)
    print(f"  有异常的: {n_active} 个格子")
    
    # 测试不同引擎
    engines_to_test = ['builtin', 'numpy']
    
    try:
        import numba
        engines_to_test.append('numba')
        # 预热
        forward_gravity(obs[:2], data['centers'][:10],
                       data['sizes'][:10], data['density'][:10], engine='numba')
    except ImportError:
        pass
    
    try:
        import harmonica
        engines_to_test.append('harmonica')
    except ImportError:
        pass
    
    print(f"\n  测试引擎: {engines_to_test}")
    print("-" * 60)
    
    results = {}
    for eng in engines_to_test:
        t0 = time.perf_counter()
        gz = forward_gravity(obs, data['centers'], data['sizes'],
                            data['density'], engine=eng)
        elapsed = time.perf_counter() - t0
        results[eng] = {'time': elapsed, 'gz': gz}
        print(f"  {eng:>12}: {elapsed:.3f}s | "
              f"gz: {gz.min():.4f} ~ {gz.max():.4f} mGal")
    
    # 验证一致性
    if len(results) > 1:
        ref_engine = 'builtin'
        ref_gz = results[ref_engine]['gz']
        print(f"\n  结果一致性（相对于 {ref_engine}）:")
        for eng, r in results.items():
            if eng == ref_engine:
                continue
            diff = np.max(np.abs(r['gz'] - ref_gz))
            rel = diff / (np.max(np.abs(ref_gz)) + 1e-30)
            print(f"    {eng}: 最大差异 {diff:.2e} mGal "
                  f"(相对 {rel:.2e}) {'✅' if rel < 1e-6 else '⚠️'}")
    
    # 加速比总结
    base_time = results.get('builtin', {}).get('time', 1)
    print(f"\n  加速比总结:")
    for eng, r in results.items():
        speedup = base_time / r['time']
        bar = '█' * int(min(speedup, 50))
        print(f"    {eng:>12}: {speedup:>6.1f}x {bar}")


def demo_scaling():
    """规模扩展测试"""
    from acceleration.numpy_gravity import gz_all_prisms_numpy
    
    print("\n╔══════════════════════════════════════════╗")
    print("║       规模扩展测试                        ║")
    print("╚══════════════════════════════════════════╝")
    
    try:
        from acceleration.numba_gravity import forward_gravity_numba, HAS_NUMBA
    except ImportError:
        HAS_NUMBA = False
    
    n_obs = 50
    obs = np.random.rand(n_obs, 3) * 1000
    
    sizes_to_test = [500, 1000, 2000, 5000, 10000, 20000]
    
    print(f"\n  固定 {n_obs} 个观测点，增加棱柱数量:")
    print(f"  {'棱柱数':>10} | {'NumPy':>12} | ", end="")
    if HAS_NUMBA:
        print(f"{'Numba':>12} | {'加速比':>8}")
    else:
        print()
    print("  " + "-" * 55)
    
    # Numba预热
    if HAS_NUMBA:
        c = np.random.rand(10, 3) * 1000
        s = np.random.rand(10, 3) * 30 + 10
        d = np.random.rand(10) * 0.5
        forward_gravity_numba(obs[:2], c, s, d)
    
    for n_prisms in sizes_to_test:
        centers = np.random.rand(n_prisms, 3) * 1000
        sizes = np.random.rand(n_prisms, 3) * 30 + 10
        densities = np.random.rand(n_prisms) * 0.5
        densities[np.random.rand(n_prisms) > 0.3] = 0
        
        # NumPy
        t0 = time.perf_counter()
        gz_np = gz_all_prisms_numpy(obs, centers, sizes, densities)
        t_numpy = time.perf_counter() - t0
        
        row = f"  {n_prisms:>10} | {t_numpy:>10.3f}s |"
        
        if HAS_NUMBA:
            t0 = time.perf_counter()
            gz_nb = forward_gravity_numba(obs, centers, sizes, densities)
            t_numba = time.perf_counter() - t0
            speedup = t_numpy / max(t_numba, 1e-10)
            row += f" {t_numba:>10.3f}s | {speedup:>6.1f}x"
        
        print(row)
    
    print(f"""
  ┌─ 性能建议 ──────────────────────────────────────────┐
  │                                                      │
  │  棱柱数 < 1000:   任何引擎都够快，不用优化              │
  │  棱柱数 1000~5000: 推荐 NumPy 向量化                  │
  │  棱柱数 > 5000:    强烈推荐 Numba（差距 10~100 倍）    │
  │  棱柱数 > 50000:   考虑 GPU 加速（CuPy/CUDA）        │
  │                                                      │
  │  安装Numba: pip install numba                        │
  │                                                      │
  └──────────────────────────────────────────────────────┘
    """)


def main():
    demo_benchmark()
    demo_real_case()
    demo_scaling()
    
    print("\n" + "=" * 60)
    print("  ✅ 性能优化演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()