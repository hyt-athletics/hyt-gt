"""
Numba JIT加速重力正演

比NumPy向量化还快 3~10 倍
支持并行计算（多核CPU同时算）

安装: pip install numba
"""

import numpy as np

# 检测Numba是否可用
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
    print("  ✅ Numba已安装，将使用JIT加速")
except ImportError:
    HAS_NUMBA = False
    print("  ⚠️ Numba未安装，将使用NumPy向量化（pip install numba 可以更快）")


# 物理常数
GAMMA = 6.674e-11
SI_TO_MGAL = 1e5
DENSITY_FACTOR = 1e3


if HAS_NUMBA:
    
    @njit(cache=True)
    def _gz_single_prism_single_obs(ox, oy, oz,
                                     x1, x2, y1, y2, z1, z2):
        """
        计算一个棱柱对一个观测点的gz
        
        纯数值计算，没有任何Python对象操作
        Numba会把这个函数编译成机器码
        
        返回: 无量纲结果（还没乘G和密度）
        """
        result = 0.0
        
        dx = (x1 - ox, x2 - ox)
        dy = (y1 - oy, y2 - oy)
        dz = (z1 - oz, z2 - oz)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = dx[i]
                    y = dy[j]
                    z = dz[k]
                    
                    r = (x * x + y * y + z * z) ** 0.5
                    
                    if r < 1e-10:
                        continue
                    
                    sign = 1.0 if (i + j + k) % 2 == 0 else -1.0
                    
                    # 项1
                    val = y + r
                    term1 = 0.0
                    if abs(val) > 1e-10:
                        if val > 0:
                            term1 = x * np.log(val)
                        else:
                            term1 = x * np.log(-val)
                    
                    # 项2
                    val2 = x + r
                    term2 = 0.0
                    if abs(val2) > 1e-10:
                        if val2 > 0:
                            term2 = y * np.log(val2)
                        else:
                            term2 = y * np.log(-val2)
                    
                    # 项3
                    term3 = 0.0
                    if abs(z) > 1e-10:
                        term3 = -z * np.arctan2(x * y, z * r)
                    
                    result += sign * (term1 + term2 + term3)
        
        return result
    
    
    @njit(parallel=True, cache=True)
    def gz_all_prisms_numba(obs_x, obs_y, obs_z,
                             x1, x2, y1, y2, z1, z2,
                             densities):
        """
        Numba并行正演：所有棱柱 × 所有观测点
        
        parallel=True → 自动多核并行
        prange → 并行循环（代替range）
        
        参数全部是一维numpy数组（Numba不喜欢复杂对象）
        """
        n_obs = len(obs_x)
        n_prisms = len(x1)
        
        gz = np.zeros(n_obs)
        
        # 外层循环并行（每个观测点独立计算）
        for iobs in prange(n_obs):
            ox = obs_x[iobs]
            oy = obs_y[iobs]
            oz = obs_z[iobs]
            
            total = 0.0
            
            # 内层循环不并行（对每个棱柱累加）
            for ip in range(n_prisms):
                if abs(densities[ip]) < 1e-10:
                    continue
                
                val = _gz_single_prism_single_obs(
                    ox, oy, oz,
                    x1[ip], x2[ip], y1[ip], y2[ip], z1[ip], z2[ip]
                )
                
                total += val * densities[ip]
            
            gz[iobs] = total * GAMMA * DENSITY_FACTOR * SI_TO_MGAL
        
        return gz
    
    
    @njit(cache=True)
    def gz_single_source_numba(obs_x, obs_y, obs_z,
                                px1, px2, py1, py2, pz1, pz2,
                                density):
        """
        一个棱柱对所有观测点（用于逐个棱柱累加的场景）
        """
        n_obs = len(obs_x)
        gz = np.zeros(n_obs)
        
        for iobs in range(n_obs):
            val = _gz_single_prism_single_obs(
                obs_x[iobs], obs_y[iobs], obs_z[iobs],
                px1, px2, py1, py2, pz1, pz2
            )
            gz[iobs] = val * density * GAMMA * DENSITY_FACTOR * SI_TO_MGAL
        
        return gz


def forward_gravity_numba(obs_points, centers, sizes, densities):
    """
    Numba加速的正演入口
    
    自动处理数据格式转换
    """
    if not HAS_NUMBA:
        from acceleration.numpy_gravity import gz_all_prisms_numpy
        return gz_all_prisms_numpy(obs_points, centers, sizes, densities)
    
    # 过滤零密度
    active = np.abs(densities) > 1e-10
    if not np.any(active):
        return np.zeros(len(obs_points))
    
    ac = centers[active].astype(np.float64)
    az = sizes[active].astype(np.float64)
    ad = densities[active].astype(np.float64)
    
    # 转换为棱柱边界坐标
    x1 = np.ascontiguousarray(ac[:, 0] - az[:, 0] / 2)
    x2 = np.ascontiguousarray(ac[:, 0] + az[:, 0] / 2)
    y1 = np.ascontiguousarray(ac[:, 1] - az[:, 1] / 2)
    y2 = np.ascontiguousarray(ac[:, 1] + az[:, 1] / 2)
    z1 = np.ascontiguousarray(ac[:, 2] - az[:, 2] / 2)
    z2 = np.ascontiguousarray(ac[:, 2] + az[:, 2] / 2)
    
    obs = obs_points.astype(np.float64)
    ox = np.ascontiguousarray(obs[:, 0])
    oy = np.ascontiguousarray(obs[:, 1])
    oz = np.ascontiguousarray(obs[:, 2])
    
    # 第一次调用会触发编译（稍慢），后续调用极快
    gz = gz_all_prisms_numba(ox, oy, oz, x1, x2, y1, y2, z1, z2, ad)
    
    return gz