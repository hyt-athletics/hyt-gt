"""
重磁潜力场网格策略规划器

目标不是复制商业软件的内部算法，而是把公开工作流里稳定的共性落实到当前项目：
    - 观测系统感知 refinement
    - 异常体尺度感知 refinement
    - 关注区域（ROI / active volume）规划
    - 网格质量摘要（QC）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass
class PotentialFieldMeshPlan:
    method_code: str
    purpose: str
    strategy_mode: str
    template_name: str
    domain_min_edge: float
    station_spacing: float
    body_min_extent: float
    target_cell_size: float
    max_level: int
    boundary_refine_level: int
    body_core_refine_level: int
    body_padding_refine_level: int
    body_padding_distance: float
    borehole_refine_levels: list
    roi_horizontal_padding: float
    roi_vertical_padding: float
    radius_of_influence: float
    active_model_mode: str
    roi_bounds: dict

    def to_dict(self):
        return asdict(self)


def _safe_positive(values, fallback):
    positives = [float(v) for v in values if float(v) > 0]
    return min(positives) if positives else float(fallback)


def _estimate_station_spacing(well):
    if well.n_stations <= 1:
        return max(float(well.z_bottom) - float(well.z_top), 1.0)
    return max((float(well.z_bottom) - float(well.z_top)) / (well.n_stations - 1), 1.0)


def _estimate_body_min_extent(bodies, domain_min_edge):
    if not bodies:
        return float(domain_min_edge / 6.0)
    extents = []
    for body in bodies:
        if hasattr(body, 'extents_3d'):
            extents.extend(body.extents_3d())
    return _safe_positive(extents, fallback=domain_min_edge / 6.0)


def _level_for_cell_size(domain_min_edge, target_cell_size, default_level):
    if target_cell_size <= 0:
        return int(default_level)
    ratio = max(float(domain_min_edge) / float(target_cell_size), 1.0)
    return max(int(default_level), int(math.ceil(math.log2(ratio))))


def _coerce_optional_int(value, default_value):
    if value is None:
        return int(default_value)
    return int(value)


def _coerce_bounds(bounds):
    return {
        'x_range': (float(bounds['x_range'][0]), float(bounds['x_range'][1])),
        'y_range': (float(bounds['y_range'][0]), float(bounds['y_range'][1])),
        'z_range': (float(bounds['z_range'][0]), float(bounds['z_range'][1])),
    }


def _union_bounds(bounds_list, fallback_bounds):
    if not bounds_list:
        return _coerce_bounds(fallback_bounds)
    x0 = min(bounds['x_range'][0] for bounds in bounds_list)
    x1 = max(bounds['x_range'][1] for bounds in bounds_list)
    y0 = min(bounds['y_range'][0] for bounds in bounds_list)
    y1 = max(bounds['y_range'][1] for bounds in bounds_list)
    z0 = min(bounds['z_range'][0] for bounds in bounds_list)
    z1 = max(bounds['z_range'][1] for bounds in bounds_list)
    return {
        'x_range': (float(x0), float(x1)),
        'y_range': (float(y0), float(y1)),
        'z_range': (float(z0), float(z1)),
    }


def _expand_bounds(bounds, horizontal_padding, vertical_padding):
    return {
        'x_range': (
            float(bounds['x_range'][0] - horizontal_padding),
            float(bounds['x_range'][1] + horizontal_padding),
        ),
        'y_range': (
            float(bounds['y_range'][0] - horizontal_padding),
            float(bounds['y_range'][1] + horizontal_padding),
        ),
        'z_range': (
            float(bounds['z_range'][0] - vertical_padding),
            float(bounds['z_range'][1] + vertical_padding),
        ),
    }


def _clip_bounds(bounds, domain):
    return {
        'x_range': (
            max(float(domain['x_range'][0]), float(bounds['x_range'][0])),
            min(float(domain['x_range'][1]), float(bounds['x_range'][1])),
        ),
        'y_range': (
            max(float(domain['y_range'][0]), float(bounds['y_range'][0])),
            min(float(domain['y_range'][1]), float(bounds['y_range'][1])),
        ),
        'z_range': (
            max(float(domain['z_range'][0]), float(bounds['z_range'][0])),
            min(float(domain['z_range'][1]), float(bounds['z_range'][1])),
        ),
    }


def _well_bounds(well):
    return {
        'x_range': (float(well.x), float(well.x)),
        'y_range': (float(well.y), float(well.y)),
        'z_range': (float(well.z_top), float(well.z_bottom)),
    }


def _intersects_bounds(centers, sizes, bounds):
    mins = centers - sizes / 2.0
    maxs = centers + sizes / 2.0
    return (
        (maxs[:, 0] >= float(bounds['x_range'][0])) &
        (mins[:, 0] <= float(bounds['x_range'][1])) &
        (maxs[:, 1] >= float(bounds['y_range'][0])) &
        (mins[:, 1] <= float(bounds['y_range'][1])) &
        (maxs[:, 2] >= float(bounds['z_range'][0])) &
        (mins[:, 2] <= float(bounds['z_range'][1]))
    )


def _distance_to_vertical_well(centers, well):
    centers = np.asarray(centers, dtype=float)
    radial = np.sqrt((centers[:, 0] - float(well.x)) ** 2 + (centers[:, 1] - float(well.y)) ** 2)
    above = np.maximum(float(well.z_top) - centers[:, 2], 0.0)
    below = np.maximum(centers[:, 2] - float(well.z_bottom), 0.0)
    vertical_offset = above + below
    return np.sqrt(radial ** 2 + vertical_offset ** 2)


def _validate_manual_bounds(bounds):
    if not isinstance(bounds, dict):
        return None
    try:
        x_range = bounds['x_range']
        y_range = bounds['y_range']
        z_range = bounds['z_range']
        manual = {
            'x_range': (float(x_range[0]), float(x_range[1])),
            'y_range': (float(y_range[0]), float(y_range[1])),
            'z_range': (float(z_range[0]), float(z_range[1])),
        }
    except Exception:
        return None
    if manual['x_range'][0] >= manual['x_range'][1]:
        return None
    if manual['y_range'][0] >= manual['y_range'][1]:
        return None
    if manual['z_range'][0] >= manual['z_range'][1]:
        return None
    return manual


def build_potential_field_mesh_plan(domain, well, bodies, mesh_settings, method_code='gravity', purpose='forward'):
    """为重力/磁法构造统一的自动网格策略。"""
    x_span = float(domain['x_range'][1] - domain['x_range'][0])
    y_span = float(domain['y_range'][1] - domain['y_range'][0])
    z_span = float(domain['z_range'][1] - domain['z_range'][0])
    domain_min_edge = _safe_positive([x_span, y_span, z_span], fallback=1.0)

    station_spacing = _estimate_station_spacing(well)
    body_min_extent = _estimate_body_min_extent(bodies, domain_min_edge)
    configured_level = int(mesh_settings.get('max_level', 5))
    strategy_mode = mesh_settings.get('strategy', 'auto')

    purpose = str(purpose or 'forward')
    if method_code == 'magnetic_3c':
        auto_target = min(
            station_spacing * 0.5,
            body_min_extent / 5.0,
            domain_min_edge / 24.0,
        )
        auto_target = max(auto_target, domain_min_edge / 160.0)
        default_boundary = configured_level
        default_core = configured_level
        default_padding = max(configured_level - 1, 1)
        default_padding_distance = max(body_min_extent * 0.75, station_spacing * 2.0, auto_target * 4.0)
        roi_horizontal_padding = max(body_min_extent * 1.0, station_spacing * 2.5, auto_target * 6.0)
        roi_vertical_padding = max(body_min_extent * 0.75, station_spacing * 2.0, auto_target * 5.0)
    else:
        auto_target = min(
            station_spacing * 0.7,
            body_min_extent / 4.0,
            domain_min_edge / 18.0,
        )
        auto_target = max(auto_target, domain_min_edge / 128.0)
        default_boundary = max(configured_level - 1, 1)
        default_core = configured_level
        default_padding = max(configured_level - 1, 1)
        default_padding_distance = max(body_min_extent, station_spacing * 3.0, auto_target * 6.0)
        roi_horizontal_padding = max(body_min_extent * 1.5, station_spacing * 4.0, auto_target * 8.0)
        roi_vertical_padding = max(body_min_extent * 1.0, station_spacing * 3.0, auto_target * 6.0)

    if purpose == 'inverse':
        auto_target *= 1.75 if method_code == 'gravity' else 1.5
        default_boundary = max(default_boundary - 1, 1)
        default_core = max(default_core - 1, default_boundary)
        default_padding = max(default_padding - 1, 1)
        default_padding_distance *= 1.35
        roi_horizontal_padding *= 1.35
        roi_vertical_padding *= 1.25
        template_name = f'{method_code}_inverse_active_volume'
    else:
        template_name = f'{method_code}_forward_nearwell'

    roi_horizontal_padding = min(roi_horizontal_padding, 0.30 * min(x_span, y_span))
    roi_vertical_padding = min(
        roi_vertical_padding,
        (0.25 if method_code == 'magnetic_3c' else 0.35) * z_span,
    )

    manual_target = float(mesh_settings.get('target_cell_size') or 0.0)
    target_cell_size = manual_target if manual_target > 0 else auto_target

    max_level = _level_for_cell_size(domain_min_edge, target_cell_size, configured_level)
    max_level = min(max(max_level, configured_level), 8)

    boundary_refine_level = _coerce_optional_int(
        mesh_settings.get('boundary_refine_level', default_boundary if default_boundary <= max_level else max_level),
        default_boundary if default_boundary <= max_level else max_level,
    )
    boundary_refine_level = max(1, min(boundary_refine_level, max_level))

    body_core_refine_level = _coerce_optional_int(
        mesh_settings.get('body_core_refine_level', default_core if default_core <= max_level else max_level),
        default_core if default_core <= max_level else max_level,
    )
    body_core_refine_level = max(boundary_refine_level, min(body_core_refine_level, max_level))

    body_padding_refine_level = _coerce_optional_int(
        mesh_settings.get('body_padding_refine_level', default_padding if default_padding <= max_level else max_level),
        default_padding if default_padding <= max_level else max_level,
    )
    body_padding_refine_level = max(1, min(body_padding_refine_level, body_core_refine_level))

    padding_distance = float(mesh_settings.get('body_padding_distance') or 0.0)
    if padding_distance <= 0:
        padding_distance = default_padding_distance

    configured_borehole = mesh_settings.get('borehole_refine_levels') or []
    if configured_borehole and strategy_mode == 'manual':
        borehole_refine_levels = [(int(level), float(radius)) for level, radius in configured_borehole]
    else:
        cell = domain_min_edge / (2 ** max_level)
        if method_code == 'magnetic_3c':
            borehole_refine_levels = [
                (max_level, max(cell * 2.0, station_spacing * 0.9)),
                (max(max_level - 1, 1), max(cell * 4.0, station_spacing * 1.8)),
                (max(max_level - 2, 1), max(cell * 8.0, station_spacing * 3.0)),
            ]
        else:
            borehole_refine_levels = [
                (max_level, max(cell * 2.5, station_spacing * 1.0)),
                (max(max_level - 1, 1), max(cell * 6.0, station_spacing * 2.5)),
                (max(max_level - 2, 1), max(cell * 12.0, station_spacing * 4.5)),
            ]
        dedup = {}
        for level, radius in borehole_refine_levels:
            dedup[int(level)] = max(float(radius), dedup.get(int(level), 0.0))
        borehole_refine_levels = sorted(
            [(level, radius) for level, radius in dedup.items()],
            key=lambda item: (-item[0], item[1]),
        )

    body_bounds = [body.bounds_3d(padding=0.0) for body in bodies if hasattr(body, 'bounds_3d')]
    focus_bounds = _union_bounds(body_bounds + [_well_bounds(well)], fallback_bounds=domain)
    roi_bounds = _clip_bounds(
        _expand_bounds(
            focus_bounds,
            horizontal_padding=roi_horizontal_padding,
            vertical_padding=roi_vertical_padding,
        ),
        domain,
    )
    active_model = mesh_settings.get('active_model') or {}
    active_model_mode = str(active_model.get('mode', 'auto'))
    manual_roi_bounds = _validate_manual_bounds(active_model.get('bounds'))
    if purpose == 'inverse' and active_model_mode == 'manual' and manual_roi_bounds is not None:
        roi_bounds = _clip_bounds(manual_roi_bounds, domain)
    else:
        active_model_mode = 'auto'
    radius_of_influence = max(
        roi_horizontal_padding,
        padding_distance,
        station_spacing * (6.0 if method_code == 'gravity' else 4.0),
    )

    return PotentialFieldMeshPlan(
        method_code=method_code,
        purpose=purpose,
        strategy_mode=strategy_mode,
        template_name=template_name,
        domain_min_edge=domain_min_edge,
        station_spacing=station_spacing,
        body_min_extent=body_min_extent,
        target_cell_size=target_cell_size,
        max_level=max_level,
        boundary_refine_level=boundary_refine_level,
        body_core_refine_level=body_core_refine_level,
        body_padding_refine_level=body_padding_refine_level,
        body_padding_distance=padding_distance,
        borehole_refine_levels=borehole_refine_levels,
        roi_horizontal_padding=roi_horizontal_padding,
        roi_vertical_padding=roi_vertical_padding,
        radius_of_influence=radius_of_influence,
        active_model_mode=active_model_mode,
        roi_bounds=roi_bounds,
    )


def annotate_potential_field_mesh(mesh_data, plan, well):
    """
    为网格数据补充 ROI / distance / QC 摘要。

    这些字段不改变正演数值本身，但为反演裁剪、VTK 导出和结果解释提供稳定输入。
    """
    centers = np.asarray(mesh_data['centers'], dtype=float)
    sizes = np.asarray(mesh_data['sizes'], dtype=float)
    cell_size = np.asarray(mesh_data.get('cell_size', np.min(sizes, axis=1)), dtype=float)
    octree_level = np.asarray(mesh_data.get('octree_level', np.zeros(len(centers), dtype=int)))

    roi_mask = _intersects_bounds(centers, sizes, plan.roi_bounds)
    distance_to_well = _distance_to_vertical_well(centers, well)
    near_well_mask = distance_to_well <= float(plan.radius_of_influence)
    density_mask = np.abs(np.asarray(mesh_data.get('density', np.zeros(len(centers))))) > 1e-12
    susceptibility_mask = np.abs(np.asarray(mesh_data.get('susceptibility', np.zeros(len(centers))))) > 1e-12
    property_mask = density_mask | susceptibility_mask

    qc_summary = {
        'method_code': plan.method_code,
        'purpose': plan.purpose,
        'total_cells': int(len(centers)),
        'roi_cells': int(np.count_nonzero(roi_mask)),
        'roi_fraction': float(np.count_nonzero(roi_mask) / max(len(centers), 1)),
        'near_well_cells': int(np.count_nonzero(near_well_mask)),
        'near_well_fraction': float(np.count_nonzero(near_well_mask) / max(len(centers), 1)),
        'property_cells': int(np.count_nonzero(property_mask)),
        'property_fraction': float(np.count_nonzero(property_mask) / max(len(centers), 1)),
        'min_cell_size': float(np.min(cell_size)),
        'median_cell_size': float(np.median(cell_size)),
        'max_cell_size': float(np.max(cell_size)),
        'target_cell_size': float(plan.target_cell_size),
        'cell_size_to_target_ratio': float(np.min(cell_size) / max(float(plan.target_cell_size), 1e-9)),
        'min_level': int(np.min(octree_level)) if len(octree_level) else 0,
        'max_level': int(np.max(octree_level)) if len(octree_level) else 0,
        'roi_bounds': plan.roi_bounds,
        'radius_of_influence': float(plan.radius_of_influence),
    }

    return {
        'roi_mask': roi_mask,
        'near_well_mask': near_well_mask,
        'distance_to_well': distance_to_well,
        'property_mask': property_mask,
        'mesh_qc': qc_summary,
    }
