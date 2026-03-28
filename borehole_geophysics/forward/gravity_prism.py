"""
长方体（矩形棱柱）重力异常解析公式

这是地球物理中最经典的公式之一（Nagy, 1966; Plouff, 1976）

一个均匀密度的长方体对外部一个点产生的垂直重力分量 gz
有精确的解析解（不需要数值积分！）

原理：
    牛顿万有引力定律 + 体积分
    对长方体可以推导出封闭解

公式说明：
    这里的 gz 指的是"垂直方向的引力加速度异常"
    单位是 mGal（毫伽）
    1 mGal = 10⁻⁵ m/s²

    对比：地球表面正常重力 ≈ 980000 mGal
    矿体异常一般在 0.01 ~ 几十 mGal 量级
"""

import numpy as np


# 万有引力常数
GAMMA = 6.674e-11  # m³/(kg·s²)

# 单位换算：结果要从 m/s² 转成 mGal
# 1 mGal = 1e-5 m/s²
# 密度单位：输入 g/cm³ = 1000 kg/m³
# 所以 G * ρ(g/cm³) * 1000(kg/m³) / 1e-5(mGal) = G * ρ * 1e8
SI_TO_MGAL = 1e5  # m/s² → mGal
DENSITY_FACTOR = 1e3  # g/cm³ → kg/m³


def gz_prism(x_obs, y_obs, z_obs,
             x1, x2, y1, y2, z1, z2, density):
    """
    计算一个长方体对一个观测点的垂直重力异常 gz
    
    参数:
        x_obs, y_obs, z_obs: 观测点坐标 (m)
        x1, x2: 长方体X方向范围 (m)，x1 < x2
        y1, y2: 长方体Y方向范围 (m)，y1 < y2
        z1, z2: 长方体Z方向范围 (m)，z1 < z2
        density: 密度差 (g/cm³)
    
    返回:
        gz: 垂直重力异常 (mGal)
    
    坐标系:
        X: 东方向
        Y: 北方向
        Z: 向下为正（深度）
        gz: Z方向的引力分量（向下为正）
    
    参考文献:
        Nagy, D., 1966, The gravitational attraction of a right rectangular prism.
        Plouff, D., 1976, Gravity and magnetic fields of polygonal prisms.
    """
    # 把观测点移到棱柱的局部坐标系（棱柱角点为原点）
    # 需要对8个角点求和（正负交替）
    dx = [x1 - x_obs, x2 - x_obs]
    dy = [y1 - y_obs, y2 - y_obs]
    dz = [z1 - z_obs, z2 - z_obs]
    
    result = 0.0
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = dx[i]
                y = dy[j]
                z = dz[k]
                
                # 距离
                r = np.sqrt(x * x + y * y + z * z)
                
                if r < 1e-10:
                    continue  # 观测点在角点上，跳过（奇点）
                
                # 符号：(-1)^(i+j+k)
                sign = (-1) ** (i + j + k)
                
                # Nagy公式的三项
                # 项1: x * ln(y + r)
                term1 = 0.0
                if abs(y + r) > 1e-10:
                    term1 = x * np.log(y + r)
                
                # 项2: y * ln(x + r)
                term2 = 0.0
                if abs(x + r) > 1e-10:
                    term2 = y * np.log(x + r)
                
                # 项3: -z * arctan(x*y / (z*r))
                term3 = 0.0
                if abs(z) > 1e-10 and abs(r) > 1e-10:
                    term3 = -z * np.arctan2(x * y, z * r)
                
                result += sign * (term1 + term2 + term3)
    
    # 乘以常数
    # gz = G * rho * result
    gz = GAMMA * density * DENSITY_FACTOR * result * SI_TO_MGAL
    
    return gz


def gz_prism_vectorized(obs_points, prism_bounds, density):
    """
    向量化版本：计算一个长方体对多个观测点的重力异常
    
    参数:
        obs_points: 观测点坐标数组 (N, 3) → [x, y, z]
        prism_bounds: (x1, x2, y1, y2, z1, z2)
        density: 密度差 (g/cm³)
    
    返回:
        gz: 重力异常数组 (N,) (mGal)
    """
    x1, x2, y1, y2, z1, z2 = prism_bounds
    
    obs_x = obs_points[:, 0]
    obs_y = obs_points[:, 1]
    obs_z = obs_points[:, 2]
    
    dx = [x1 - obs_x, x2 - obs_x]
    dy = [y1 - obs_y, y2 - obs_y]
    dz = [z1 - obs_z, z2 - obs_z]
    
    result = np.zeros(len(obs_points))
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = dx[i]
                y = dy[j]
                z = dz[k]
                
                r = np.sqrt(x**2 + y**2 + z**2)
                
                sign = (-1) ** (i + j + k)
                
                # 安全处理（避免log(0)和除以0）
                safe_r = np.where(r > 1e-10, r, 1e-10)
                
                term1 = np.where(
                    np.abs(y + safe_r) > 1e-10,
                    x * np.log(np.abs(y + safe_r) + 1e-30),
                    0.0
                )
                
                term2 = np.where(
                    np.abs(x + safe_r) > 1e-10,
                    y * np.log(np.abs(x + safe_r) + 1e-30),
                    0.0
                )
                
                term3 = np.where(
                    (np.abs(z) > 1e-10) & (safe_r > 1e-10),
                    -z * np.arctan2(x * y, z * safe_r),
                    0.0
                )
                
                result += sign * (term1 + term2 + term3)
    
    gz = GAMMA * density * DENSITY_FACTOR * result * SI_TO_MGAL
    return gz


def gz_multi_prisms(obs_points, centers, sizes, densities):
    """
    计算多个长方体对多个观测点的总重力异常
    
    这是实际使用最多的函数！
    
    参数:
        obs_points: 观测点 (N_obs, 3)
        centers: 长方体中心坐标 (N_prism, 3)
        sizes: 长方体三方向尺寸 (N_prism, 3)
        densities: 密度差数组 (N_prism,) (g/cm³)
    
    返回:
        gz: 每个观测点的总重力异常 (N_obs,) (mGal)
    """
    n_obs = len(obs_points)
    n_prisms = len(centers)
    
    gz_total = np.zeros(n_obs)
    
    # 只计算密度不为0的棱柱（跳过背景，大幅加速）
    active = np.abs(densities) > 1e-10
    n_active = np.sum(active)
    
    if n_active == 0:
        return gz_total
    
    active_centers = centers[active]
    active_sizes = sizes[active]
    active_densities = densities[active]
    
    print(f"    正在计算: {n_obs} 个观测点 × {n_active} 个有效棱柱...")
    
    report_interval = max(n_active // 10, 1)
    
    for p in range(n_active):
        if p % report_interval == 0 and p > 0:
            progress = p / n_active * 100
            print(f"      进度: {progress:.0f}%")
        
        cx, cy, cz = active_centers[p]
        dx, dy, dz = active_sizes[p]
        
        prism_bounds = (
            cx - dx / 2, cx + dx / 2,
            cy - dy / 2, cy + dy / 2,
            cz - dz / 2, cz + dz / 2
        )
        
        gz_total += gz_prism_vectorized(
            obs_points, prism_bounds, active_densities[p]
        )
    
    return gz_total