"""
重力正演引擎

平台层负责统一入口和引擎调度：
  - builtin: 自研解析公式
  - numpy:   NumPy 向量化
  - numba:   Numba JIT
  - harmonica: 外部成熟库
"""

from __future__ import annotations

import time

import numpy as np


def _check_harmonica():
    """检查 harmonica 是否可用。"""
    try:
        import harmonica  # noqa: F401
        return True
    except ImportError:
        return False


def _check_numba():
    """检查 numba 是否可用。"""
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def forward_gravity_builtin(obs_points, centers, sizes, densities):
    """自研解析公式引擎。"""
    from forward.gravity_prism import gz_multi_prisms

    print("  引擎: 自研解析公式 (Nagy, 1966)")
    t0 = time.time()
    gz = gz_multi_prisms(obs_points, centers, sizes, densities)
    print(f"  计算完成: {time.time() - t0:.2f} 秒")
    return gz


def forward_gravity_numpy_vectorized(obs_points, centers, sizes, densities):
    """NumPy 全向量化引擎。"""
    from acceleration.numpy_gravity import gz_all_prisms_numpy

    print("  引擎: NumPy全向量化")
    t0 = time.time()
    gz = gz_all_prisms_numpy(obs_points, centers, sizes, densities)
    print(f"  计算完成: {time.time() - t0:.2f} 秒")
    return gz


def forward_gravity_numba_engine(obs_points, centers, sizes, densities):
    """Numba JIT 加速引擎。"""
    from acceleration.numba_gravity import forward_gravity_numba

    print("  引擎: Numba JIT (并行)")
    t0 = time.time()
    gz = forward_gravity_numba(obs_points, centers, sizes, densities)
    print(f"  计算完成: {time.time() - t0:.2f} 秒")
    return gz


def forward_gravity_harmonica(obs_points, centers, sizes, densities):
    """harmonica 重力引擎。"""
    import harmonica as hm

    print("  引擎: harmonica (Fatiando a Terra)")
    t0 = time.time()

    active = np.abs(densities) > 1e-10
    if not np.any(active):
        return np.zeros(len(obs_points))

    ac = centers[active]
    az = sizes[active]
    ad = densities[active]

    prisms = np.column_stack([
        ac[:, 0] - az[:, 0] / 2.0,
        ac[:, 0] + az[:, 0] / 2.0,
        ac[:, 1] - az[:, 1] / 2.0,
        ac[:, 1] + az[:, 1] / 2.0,
        -(ac[:, 2] + az[:, 2] / 2.0),
        -(ac[:, 2] - az[:, 2] / 2.0),
    ])

    density_kgm3 = ad * 1000.0
    coordinates = (
        obs_points[:, 0],
        obs_points[:, 1],
        -obs_points[:, 2],
    )

    gz = hm.prism_gravity(
        coordinates,
        prisms,
        density_kgm3,
        field='g_z',
    )

    print(f"  计算完成: {time.time() - t0:.2f} 秒")
    return gz


def forward_gravity(obs_points, centers, sizes, densities, engine='auto'):
    """
    统一重力正演入口。

    引擎优先级:
      1. numba
      2. harmonica
      3. numpy
      4. builtin
    """
    print(f"\n  ┌─ 重力正演计算 ────────────────┐")
    print(f"  │ 观测点: {len(obs_points)} 个")
    print(f"  │ 棱柱总数: {len(centers)} 个")
    n_active = int(np.sum(np.abs(densities) > 1e-10))
    print(f"  │ 有异常的: {n_active} 个")
    print(f"  └───────────────────────────────┘")

    if engine == 'auto':
        if _check_numba():
            print("  ✅ Numba已安装，将使用JIT加速")
            engine = 'numba'
        elif _check_harmonica():
            engine = 'harmonica'
        else:
            engine = 'numpy'

    engine_map = {
        'numba': (forward_gravity_numba_engine, _check_numba),
        'harmonica': (forward_gravity_harmonica, _check_harmonica),
        'numpy': (forward_gravity_numpy_vectorized, lambda: True),
        'builtin': (forward_gravity_builtin, lambda: True),
    }

    func, check = engine_map.get(engine, (forward_gravity_builtin, lambda: True))
    if not check():
        print(f"  ⚠️ {engine} 不可用，回退到 numpy")
        func = forward_gravity_numpy_vectorized

    gz = func(obs_points, centers, sizes, densities)

    print(f"\n  ┌─ 计算结果 ──────────────────────┐")
    print(f"  │ gz 范围: {gz.min():.4f} ~ {gz.max():.4f} mGal")
    print(f"  │ gz 均值: {gz.mean():.4f} mGal")
    print(f"  └────────────────────────────────────┘")
    return gz
