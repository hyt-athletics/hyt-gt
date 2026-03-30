"""
三分量井中磁测正演引擎

遵循平台原则：这里只封装外部库，不实现复杂自研磁法解析核。
当前后端使用 harmonica.prism_magnetic。
"""

from __future__ import annotations

import math
import time

import numpy as np


MU0 = 4e-7 * math.pi


def _check_harmonica():
    try:
        import harmonica  # noqa: F401
        return True
    except ImportError:
        return False


def background_unit_vector(inclination_deg, declination_deg):
    """背景场单位方向，采用 x-east, y-north, z-down 坐标系。"""
    inc = math.radians(inclination_deg)
    dec = math.radians(declination_deg)
    return np.array([
        math.cos(inc) * math.cos(dec),
        math.cos(inc) * math.sin(dec),
        math.sin(inc),
    ])


def induced_magnetization_from_field(
    susceptibilities,
    b0_strength,
    b0_inclination,
    b0_declination,
):
    """
    由背景磁场计算感应磁化强度。

    输入背景场强度单位为 nT，输出磁化强度单位为 A/m。
    """
    b0_tesla = float(b0_strength) * 1e-9
    h0_am = b0_tesla / MU0
    direction = background_unit_vector(b0_inclination, b0_declination)
    return np.asarray(susceptibilities, dtype=float)[:, np.newaxis] * h0_am * direction


def remanent_magnetization_vector(
    remanent_mag,
    remanent_inclination,
    remanent_declination,
):
    """剩磁参数转磁化矢量，采用 x-east, y-north, z-down。"""
    rem_mag = np.asarray(remanent_mag, dtype=float)
    inc = np.radians(np.asarray(remanent_inclination, dtype=float))
    dec = np.radians(np.asarray(remanent_declination, dtype=float))
    return np.column_stack([
        rem_mag * np.cos(inc) * np.cos(dec),
        rem_mag * np.cos(inc) * np.sin(dec),
        rem_mag * np.sin(inc),
    ])


def _prisms_from_mesh(centers, sizes):
    """统一网格中心+尺寸转换为 harmonica prisms。"""
    prisms = np.column_stack([
        centers[:, 0] - sizes[:, 0] / 2.0,
        centers[:, 0] + sizes[:, 0] / 2.0,
        centers[:, 1] - sizes[:, 1] / 2.0,
        centers[:, 1] + sizes[:, 1] / 2.0,
        -(centers[:, 2] + sizes[:, 2] / 2.0),
        -(centers[:, 2] - sizes[:, 2] / 2.0),
    ])
    return prisms


def _coordinates_from_obs(obs_points, eps=1e-3):
    """
    观测点转换为 harmonica 使用的 upward 坐标系。

    对外部库调用引入极小坐标扰动，避免观测点恰落在棱柱边界/角点时出现奇异值。
    该扰动仅用于外部正演坐标，不修改项目内部井轨迹和深度。
    """
    return (
        obs_points[:, 0] + eps,
        obs_points[:, 1] + eps,
        -obs_points[:, 2] + eps,
    )


def _containment_mask(obs_points, centers, sizes, tol=1e-9):
    """判断哪些观测点落在棱柱内部或边界上。"""
    dx = np.abs(obs_points[:, np.newaxis, 0] - centers[np.newaxis, :, 0])
    dy = np.abs(obs_points[:, np.newaxis, 1] - centers[np.newaxis, :, 1])
    dz = np.abs(obs_points[:, np.newaxis, 2] - centers[np.newaxis, :, 2])
    half_sizes = sizes[np.newaxis, :, :] / 2.0
    return (
        (dx <= half_sizes[:, :, 0] + tol) &
        (dy <= half_sizes[:, :, 1] + tol) &
        (dz <= half_sizes[:, :, 2] + tol)
    )


def _compute_b_components(coords, prisms, magnetization_e, magnetization_n, magnetization_u):
    import harmonica as hm
    return hm.prism_magnetic(
        coords,
        prisms,
        (magnetization_e, magnetization_n, magnetization_u),
        field='b',
    )


def forward_magnetic_harmonica(
    obs_points,
    centers,
    sizes,
    susceptibilities,
    b0_strength,
    b0_inclination,
    b0_declination,
    remanent_mag=None,
    remanent_inclination=None,
    remanent_declination=None,
):
    """使用 harmonica 计算三分量磁异常。"""
    import harmonica as hm

    print("  引擎: harmonica.prism_magnetic")
    t0 = time.time()

    susceptibilities = np.asarray(susceptibilities, dtype=float)
    active = np.abs(susceptibilities) > 1e-12

    if remanent_mag is not None:
        remanent_mag = np.asarray(remanent_mag, dtype=float)
        active = active | (np.abs(remanent_mag) > 1e-12)

    if not np.any(active):
        zeros = np.zeros(len(obs_points), dtype=float)
        return {'bx': zeros, 'by': zeros, 'bz': zeros, 'bt': zeros}

    active_centers = np.asarray(centers, dtype=float)[active]
    active_sizes = np.asarray(sizes, dtype=float)[active]
    prisms = _prisms_from_mesh(active_centers, active_sizes)

    magnetization = induced_magnetization_from_field(
        susceptibilities[active],
        b0_strength,
        b0_inclination,
        b0_declination,
    )

    if remanent_mag is not None:
        remanence = remanent_magnetization_vector(
            remanent_mag[active],
            np.zeros(np.sum(active)) if remanent_inclination is None else np.asarray(remanent_inclination, dtype=float)[active],
            np.zeros(np.sum(active)) if remanent_declination is None else np.asarray(remanent_declination, dtype=float)[active],
        )
        magnetization = magnetization + remanence

    # harmonica 使用 upward 为正，因此要翻转 z 分量。
    magnetization_e = magnetization[:, 0]
    magnetization_n = magnetization[:, 1]
    magnetization_u = -magnetization[:, 2]

    obs_points = np.asarray(obs_points, dtype=float)
    contain_mask = _containment_mask(obs_points, active_centers, active_sizes)

    if np.any(contain_mask):
        n_problem = int(np.sum(np.any(contain_mask, axis=1)))
        print(f"  ⚠️ {n_problem} 个观测点落在磁性单元内部，将跳过对应单元的局部贡献以避免外部库奇异值")
        be = np.zeros(len(obs_points), dtype=float)
        bn = np.zeros(len(obs_points), dtype=float)
        bu = np.zeros(len(obs_points), dtype=float)
        for i in range(len(obs_points)):
            valid = ~contain_mask[i]
            if not np.any(valid):
                continue
            coords_i = _coordinates_from_obs(obs_points[i:i+1])
            be_i, bn_i, bu_i = _compute_b_components(
                coords_i,
                prisms[valid],
                magnetization_e[valid],
                magnetization_n[valid],
                magnetization_u[valid],
            )
            be[i] = be_i[0]
            bn[i] = bn_i[0]
            bu[i] = bu_i[0]
    else:
        be, bn, bu = _compute_b_components(
            _coordinates_from_obs(obs_points),
            prisms,
            magnetization_e,
            magnetization_n,
            magnetization_u,
        )

    bx = np.asarray(be, dtype=float)
    by = np.asarray(bn, dtype=float)
    bz = -np.asarray(bu, dtype=float)

    if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by)) and np.all(np.isfinite(bz))):
        raise RuntimeError("harmonica 返回了非有限磁场值，请检查井轨迹是否与棱柱边界重合")

    direction = background_unit_vector(b0_inclination, b0_declination)
    bt = bx * direction[0] + by * direction[1] + bz * direction[2]

    elapsed = time.time() - t0
    print(f"  计算完成: {elapsed:.2f} 秒")
    return {'bx': bx, 'by': by, 'bz': bz, 'bt': bt}


def forward_magnetic(
    obs_points,
    centers,
    sizes,
    susceptibilities,
    b0_strength,
    b0_inclination,
    b0_declination,
    remanent_mag=None,
    remanent_inclination=None,
    remanent_declination=None,
    engine='auto',
):
    """三分量井中磁测正演统一入口。"""
    print(f"\n  ┌─ 三分量磁法正演 ───────────────┐")
    print(f"  │ 观测点: {len(obs_points)} 个")
    print(f"  │ 棱柱总数: {len(centers)} 个")
    print(f"  │ 背景场: {b0_strength:.1f} nT")
    print(f"  │ 倾角/偏角: {b0_inclination:.1f}° / {b0_declination:.1f}°")
    print(f"  └───────────────────────────────┘")

    if engine == 'auto':
        engine = 'harmonica'

    if engine != 'harmonica':
        raise NotImplementedError(f"磁法当前只接入 harmonica，暂不支持引擎: {engine}")
    if not _check_harmonica():
        raise RuntimeError("harmonica 未安装，无法进行磁法正演")

    result = forward_magnetic_harmonica(
        obs_points=obs_points,
        centers=centers,
        sizes=sizes,
        susceptibilities=susceptibilities,
        b0_strength=b0_strength,
        b0_inclination=b0_inclination,
        b0_declination=b0_declination,
        remanent_mag=remanent_mag,
        remanent_inclination=remanent_inclination,
        remanent_declination=remanent_declination,
    )

    print(f"\n  ┌─ 计算结果 ──────────────────────┐")
    print(f"  │ Bx 范围: {result['bx'].min():.3f} ~ {result['bx'].max():.3f} nT")
    print(f"  │ By 范围: {result['by'].min():.3f} ~ {result['by'].max():.3f} nT")
    print(f"  │ Bz 范围: {result['bz'].min():.3f} ~ {result['bz'].max():.3f} nT")
    print(f"  │ ΔT 范围: {result['bt'].min():.3f} ~ {result['bt'].max():.3f} nT")
    print(f"  └────────────────────────────────────┘")
    return result
