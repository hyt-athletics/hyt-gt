"""
性能测试工具

对比不同实现的速度
生成性能报告
"""

import numpy as np
import time
import sys


class BenchmarkTimer:
    """简单的计时器"""
    
    def __init__(self, label=""):
        self.label = label
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        return False


def run_gravity_benchmark(n_obs_list=None, n_prism_list=None):
    """
    重力正演基准测试
    
    测试不同规模下各引擎的速度
    """
    from forward.gravity_prism import gz_multi_prisms
    from acceleration.numpy_gravity import gz_all_prisms_numpy
    
    try:
        from acceleration.numba_gravity import forward_gravity_numba, HAS_NUMBA
    except ImportError:
        HAS_NUMBA = False
    
    if n_obs_list is None:
        n_obs_list = [10, 40, 100]
    if n_prism_list is None:
        n_prism_list = [100, 500, 2000, 5000]
    
    print("\n" + "=" * 70)
    print("  重力正演 性能基准测试")
    print("=" * 70)
    
    # 列标题
    engines = ['Python循环']
    engines.append('NumPy向量化')
    if HAS_NUMBA:
        engines.append('Numba JIT')
    
    header = f"{'观测点':>8} {'棱柱':>8}"
    for eng in engines:
        header += f" | {eng:>14}"
    header += " | 加速比"
    
    print(header)
    print("-" * len(header))
    
    results = []
    
    for n_obs in n_obs_list:
        for n_prisms in n_prism_list:
            np.random.seed(42)
            
            obs = np.random.rand(n_obs, 3) * 1000
            centers = np.random.rand(n_prisms, 3) * 1000
            sizes = np.random.rand(n_prisms, 3) * 50 + 10
            densities = (np.random.rand(n_prisms) - 0.5) * 0.5
            # 80%的棱柱密度为0（模拟真实情况）
            densities[np.random.rand(n_prisms) > 0.2] = 0
            
            times = {}
            
            # 1. Python循环（原始版本）
            if n_prisms <= 2000:  # 太大就跳过（太慢了）
                with BenchmarkTimer() as t:
                    gz_python = gz_multi_prisms(obs, centers, sizes, densities)
                times['Python循环'] = t.elapsed
            else:
                times['Python循环'] = float('inf')
            
            # 2. NumPy向量化
            with BenchmarkTimer() as t:
                gz_numpy = gz_all_prisms_numpy(obs, centers, sizes, densities)
            times['NumPy向量化'] = t.elapsed
            
            # 3. Numba JIT
            if HAS_NUMBA:
                # 预热编译
                if n_obs == n_obs_list[0] and n_prisms == n_prism_list[0]:
                    _ = forward_gravity_numba(obs[:2], centers[:10], sizes[:10], densities[:10])
                
                with BenchmarkTimer() as t:
                    gz_numba = forward_gravity_numba(obs, centers, sizes, densities)
                times['Numba JIT'] = t.elapsed
            
            # 计算加速比
            base_time = times['Python循环']
            fastest_time = min(times.values())
            speedup = base_time / fastest_time if fastest_time > 0 else float('inf')
            
            # 打印结果
            row = f"{n_obs:>8} {n_prisms:>8}"
            for eng in engines:
                t = times.get(eng, float('inf'))
                if t == float('inf'):
                    row += f" | {'skip':>14}"
                elif t < 0.001:
                    row += f" | {t*1000:>11.2f} ms"
                else:
                    row += f" | {t:>12.3f} s"
            
            if speedup != float('inf'):
                row += f" | {speedup:>5.0f}x"
            else:
                row += f" | {'N/A':>5}"
            
            print(row)
            results.append((n_obs, n_prisms, times))
    
    # 验证结果一致性
    print("\n  结果验证:")
    np.random.seed(42)
    obs_v = np.random.rand(20, 3) * 1000
    c_v = np.random.rand(100, 3) * 1000
    s_v = np.random.rand(100, 3) * 50 + 10
    d_v = (np.random.rand(100) - 0.5) * 0.5
    
    gz_ref = gz_multi_prisms(obs_v, c_v, s_v, d_v)
    gz_np = gz_all_prisms_numpy(obs_v, c_v, s_v, d_v)
    
    diff_np = np.max(np.abs(gz_ref - gz_np))
    print(f"    Python vs NumPy 最大差异: {diff_np:.2e} mGal {'✅' if diff_np < 1e-8 else '❌'}")
    
    if HAS_NUMBA:
        gz_nb = forward_gravity_numba(obs_v, c_v, s_v, d_v)
        diff_nb = np.max(np.abs(gz_ref - gz_nb))
        print(f"    Python vs Numba 最大差异: {diff_nb:.2e} mGal {'✅' if diff_nb < 1e-8 else '❌'}")
    
    return results


def run_geometry_benchmark():
    """几何计算基准测试"""
    from geometry.polygon import point_in_polygon_2d
    from acceleration.numba_geometry import fast_points_in_polygon
    from acceleration.numpy_geometry import (
        sphere_contains_vectorized, block_contains_vectorized
    )
    
    print("\n" + "=" * 70)
    print("  几何计算 性能基准测试")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 多边形（20个顶点）
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    poly_x = 500 + 100 * np.cos(angles)
    poly_z = 500 + 80 * np.sin(angles)
    polygon = list(zip(poly_x, poly_z))
    
    for n_points in [1000, 10000, 100000]:
        test_x = np.random.rand(n_points) * 1000
        test_z = np.random.rand(n_points) * 1000
        
        # Python循环
        with BenchmarkTimer() as t1:
            result_python = np.array([
                point_in_polygon_2d(test_x[i], test_z[i], polygon)
                for i in range(n_points)
            ])
        
        # 快速版（Numba或NumPy）
        with BenchmarkTimer() as t2:
            result_fast = fast_points_in_polygon(test_x, test_z, poly_x, poly_z)
        
        speedup = t1.elapsed / max(t2.elapsed, 1e-10)
        match = np.all(result_python == result_fast)
        
        print(f"  {n_points:>8} 个点: "
              f"Python {t1.elapsed:.3f}s → 加速版 {t2.elapsed:.4f}s "
              f"({speedup:.0f}x 加速) {'✅' if match else '❌'}")
    
    # 球体判断
    print("\n  球体包含判断:")
    centers = np.random.rand(100000, 3) * 1000
    
    with BenchmarkTimer() as t1:
        mask1 = np.array([
            (c[0]-500)**2 + (c[1]-500)**2 + (c[2]-400)**2 <= 80**2
            for c in centers
        ])
    
    with BenchmarkTimer() as t2:
        mask2 = sphere_contains_vectorized(centers, (500, 500, 400), 80)
    
    speedup = t1.elapsed / max(t2.elapsed, 1e-10)
    print(f"    100000个格子: Python {t1.elapsed:.3f}s → NumPy {t2.elapsed:.4f}s "
          f"({speedup:.0f}x)")