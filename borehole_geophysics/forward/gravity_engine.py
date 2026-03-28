"""
重力正演引擎

提供两个后端：
  1. 自研解析公式（永远可用，不需要额外安装）
  2. harmonica（Fatiando a Terra项目，更快更完善）

自动检测：如果安装了harmonica就用它，否则用自研的
"""

import numpy as np
import time


def _check_harmonica():
    """检查harmonica是否可用"""
    try:
        import harmonica as hm
        return True
    except ImportError:
        return False


def _check_simpeg():
    """检查SimPEG是否可用"""
    try:
        import SimPEG
        return True
    except ImportError:
        return False


# =============================================================
#  引擎1：自研解析公式
# =============================================================

def forward_gravity_builtin(obs_points, centers, sizes, densities):
    """
    自研正演引擎
    
    直接调用 gz_multi_prisms
    """
    from forward.gravity_prism import gz_multi_prisms
    
    print("  引擎: 自研解析公式 (Nagy, 1966)")
    t0 = time.time()
    
    gz = gz_multi_prisms(obs_points, centers, sizes, densities)
    
    elapsed = time.time() - t0
    print(f"  计算完成: {elapsed:.2f} 秒")
    
    return gz


# =============================================================
#  引擎2：harmonica（Fatiando a Terra）
# =============================================================

def forward_gravity_harmonica(obs_points, centers, sizes, densities):
    """
    使用 harmonica 库计算重力正演
    
    安装方法: pip install harmonica
    """
    import harmonica as hm
    
    print("  引擎: harmonica (Fatiando a Terra)")
    t0 = time.time()
    
    # 只计算有异常的棱柱
    active = np.abs(densities) > 1e-10
    
    if not np.any(active):
        return np.zeros(len(obs_points))
    
    ac = centers[active]
    az = sizes[active]
    ad = densities[active]
    
    # harmonica的棱柱格式: [west, east, south, north, bottom, top]
    prisms = np.column_stack([
        ac[:, 0] - az[:, 0] / 2,  # west (x_min)
        ac[:, 0] + az[:, 0] / 2,  # east (x_max)
        ac[:, 1] - az[:, 1] / 2,  # south (y_min)
        ac[:, 1] + az[:, 1] / 2,  # north (y_max)
        ac[:, 2] - az[:, 2] / 2,  # bottom (z_min)  
        ac[:, 2] + az[:, 2] / 2,  # top (z_max)
    ])
    
    # harmonica的密度单位是 kg/m³
    density_kgm3 = ad * 1000.0
    
    # harmonica坐标系：z向上为正
    # 我们的坐标系：z向下为正
    # 需要翻转z坐标
    coordinates = (
        obs_points[:, 0],     # easting
        obs_points[:, 1],     # northing
        -obs_points[:, 2],    # upward (翻转z)
    )
    
    # 翻转棱柱的z坐标
    prisms_flipped = prisms.copy()
    prisms_flipped[:, 4] = -prisms[:, 5]  # bottom = -top
    prisms_flipped[:, 5] = -prisms[:, 4]  # top = -bottom
    
    # 计算gz（向下为正）
    gz = hm.prism_gravity(
        coordinates,
        prisms_flipped,
        density_kgm3,
        field="g_z"
    )
    
    # harmonica的gz是向上为正，我们翻转
    gz = -gz
    
    # 转换单位：harmonica输出 m/s² → mGal
    gz = gz * 1e5
    
    elapsed = time.time() - t0
    print(f"  计算完成: {elapsed:.2f} 秒")
    
    return gz


# =============================================================
#  自动选择引擎
# =============================================================

def forward_gravity(obs_points, centers, sizes, densities, engine='auto'):
    """
    计算重力正演（自动选择最佳引擎）
    
    参数:
        obs_points: 观测点坐标 (N_obs, 3) [x, y, z] 单位m
        centers: 棱柱中心坐标 (N_prism, 3)
        sizes: 棱柱尺寸 (N_prism, 3)
        densities: 密度差 (N_prism,) 单位 g/cm³
        engine: 'auto', 'builtin', 'harmonica'
    
    返回:
        gz: 垂直重力异常 (N_obs,) 单位 mGal
    """
    print(f"\n  ┌─ 重力正演计算 ────────────────┐")
    print(f"  │ 观测点: {len(obs_points)} 个")
    print(f"  │ 棱柱总数: {len(centers)} 个")
    print(f"  │ 有异常的: {np.sum(np.abs(densities) > 1e-10)} 个")
    print(f"  └───────────────────────────────┘")
    
    if engine == 'auto':
        if _check_harmonica():
            engine = 'harmonica'
        else:
            engine = 'builtin'
    
    if engine == 'harmonica':
        if not _check_harmonica():
            print("  ⚠️ harmonica未安装，切换到自研引擎")
            print("     安装方法: pip install harmonica")
            engine = 'builtin'
    
    if engine == 'harmonica':
        gz = forward_gravity_harmonica(obs_points, centers, sizes, densities)
    else:
        gz = forward_gravity_builtin(obs_points, centers, sizes, densities)
    
    # 结果统计
    print(f"\n  ┌─ 计算结果 ──────────────────────┐")
    print(f"  │ gz 范围: {gz.min():.4f} ~ {gz.max():.4f} mGal")
    print(f"  │ gz 均值: {gz.mean():.4f} mGal")
    print(f"  │ gz 标准差: {gz.std():.4f} mGal")
    print(f"  └────────────────────────────────────┘")
    
    return gz
# ======= 在文件中添加以下内容 =======

# =============================================================
#  引擎3：NumPy向量化（不需要额外安装）
# =============================================================

def forward_gravity_numpy_vectorized(obs_points, centers, sizes, densities):
    """NumPy全向量化引擎"""
    from acceleration.numpy_gravity import gz_all_prisms_numpy
    
    print("  引擎: NumPy全向量化")
    t0 = time.time()
    gz = gz_all_prisms_numpy(obs_points, centers, sizes, densities)
    elapsed = time.time() - t0
    print(f"  计算完成: {elapsed:.2f} 秒")
    return gz


# =============================================================
#  引擎4：Numba JIT加速
# =============================================================

def _check_numba():
    try:
        import numba
        return True
    except ImportError:
        return False


def forward_gravity_numba_engine(obs_points, centers, sizes, densities):
    """Numba JIT加速引擎"""
    from acceleration.numba_gravity import forward_gravity_numba
    
    print("  引擎: Numba JIT (并行)")
    t0 = time.time()
    gz = forward_gravity_numba(obs_points, centers, sizes, densities)
    elapsed = time.time() - t0
    print(f"  计算完成: {elapsed:.2f} 秒")
    return gz


# =============================================================
#  更新 forward_gravity 函数：加入新引擎
# =============================================================

def forward_gravity(obs_points, centers, sizes, densities, engine='auto'):
    """
    计算重力正演（自动选择最佳引擎）
    
    引擎优先级:
      1. numba（最快，需要 pip install numba）
      2. harmonica（成熟库，需要 pip install harmonica）
      3. numpy（纯NumPy，不需要额外安装，比循环快10-50倍）
      4. builtin（原始Python循环，最慢但最可靠）
    """
    print(f"\n  ┌─ 重力正演计算 ────────────────┐")
    print(f"  │ 观测点: {len(obs_points)} 个")
    print(f"  │ 棱柱总数: {len(centers)} 个")
    n_active = np.sum(np.abs(densities) > 1e-10)
    print(f"  │ 有异常的: {n_active} 个")
    print(f"  └───────────────────────────────┘")
    
    if engine == 'auto':
        if _check_numba():
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