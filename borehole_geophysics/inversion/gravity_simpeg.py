"""
基于 SimPEG 的重力反演接入

这里只负责把平台统一网格/数据转换成 SimPEG 可接受的形式，
不在这里实现自研反演算法。
"""

from __future__ import annotations

import importlib

import numpy as np


def _check_simpeg():
    """检查 SimPEG 依赖。"""
    try:
        importlib.import_module('simpeg')
        importlib.import_module('discretize')
        return True
    except Exception:
        return False


def _mesh_bounds(mesh_data):
    centers = np.asarray(mesh_data['centers'], dtype=float)
    sizes = np.asarray(mesh_data['sizes'], dtype=float)
    mins = np.min(centers - sizes / 2.0, axis=0)
    maxs = np.max(centers + sizes / 2.0, axis=0)
    return mins, maxs


def _subset_mesh_data(mesh_data, mask):
    subset = {}
    mask = np.asarray(mask, dtype=bool)
    for key, value in mesh_data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape:
            subset[key] = value[mask]
        else:
            subset[key] = value
    subset['n_cells'] = int(np.count_nonzero(mask))
    subset['source_n_cells'] = int(mesh_data.get('n_cells', len(mask)))
    return subset


def _coerce_bounds(bounds):
    if not isinstance(bounds, dict):
        return None
    try:
        coerced = {
            'x_range': (float(bounds['x_range'][0]), float(bounds['x_range'][1])),
            'y_range': (float(bounds['y_range'][0]), float(bounds['y_range'][1])),
            'z_range': (float(bounds['z_range'][0]), float(bounds['z_range'][1])),
        }
    except Exception:
        return None
    if coerced['x_range'][0] >= coerced['x_range'][1]:
        return None
    if coerced['y_range'][0] >= coerced['y_range'][1]:
        return None
    if coerced['z_range'][0] >= coerced['z_range'][1]:
        return None
    return coerced


def _base_cell_sizes(mesh_data):
    sizes = np.asarray(mesh_data['sizes'], dtype=float)
    min_sizes = np.min(sizes, axis=0)
    mesh_strategy = mesh_data.get('mesh_strategy') or {}
    mesh_qc = mesh_data.get('mesh_qc') or {}
    target_cell = float(
        mesh_strategy.get('target_cell_size')
        or mesh_qc.get('target_cell_size')
        or 0.0
    )
    if target_cell > 0:
        target_sizes = np.full(3, target_cell, dtype=float)
        return np.maximum(min_sizes, target_sizes)
    return min_sizes


def _build_tensor_mesh(mesh_data):
    from discretize import TensorMesh

    mins, maxs = _mesh_bounds(mesh_data)
    dx, dy, dz = _base_cell_sizes(mesh_data)
    nx = int(round((maxs[0] - mins[0]) / dx))
    ny = int(round((maxs[1] - mins[1]) / dy))
    nz = int(round((maxs[2] - mins[2]) / dz))

    hx = np.full(nx, dx, dtype=float)
    hy = np.full(ny, dy, dtype=float)
    hz = np.full(nz, dz, dtype=float)

    origin = [mins[0], mins[1], -maxs[2]]
    mesh = TensorMesh([hx, hy, hz], origin=origin)
    return mesh, mins, maxs, np.array([dx, dy, dz], dtype=float)


def _fill_tensor_property(mesh_data, prop_name, mins, base_sizes):
    centers = np.asarray(mesh_data['centers'], dtype=float)
    sizes = np.asarray(mesh_data['sizes'], dtype=float)
    values = np.asarray(mesh_data[prop_name], dtype=float)

    dx, dy, dz = base_sizes
    nx = int(round((np.max(centers[:, 0] + sizes[:, 0] / 2.0) - mins[0]) / dx))
    ny = int(round((np.max(centers[:, 1] + sizes[:, 1] / 2.0) - mins[1]) / dy))
    nz = int(round((np.max(centers[:, 2] + sizes[:, 2] / 2.0) - mins[2]) / dz))

    tensor = np.zeros((nx, ny, nz), dtype=float)

    for center, size, value in zip(centers, sizes, values):
        x0, y0, z0 = center - size / 2.0
        x1, y1, z1 = center + size / 2.0
        ix0 = int(round((x0 - mins[0]) / dx))
        iy0 = int(round((y0 - mins[1]) / dy))
        iz0 = int(round((z0 - mins[2]) / dz))
        ix1 = int(round((x1 - mins[0]) / dx))
        iy1 = int(round((y1 - mins[1]) / dy))
        iz1 = int(round((z1 - mins[2]) / dz))
        tensor[ix0:ix1, iy0:iy1, iz0:iz1] = value

    return tensor.reshape(-1, order='F')


def _fill_tensor_boolean(mesh_data, source_mask, mins, base_sizes):
    centers = np.asarray(mesh_data['centers'], dtype=float)
    sizes = np.asarray(mesh_data['sizes'], dtype=float)
    mask = np.asarray(source_mask, dtype=bool)

    dx, dy, dz = base_sizes
    nx = int(round((np.max(centers[:, 0] + sizes[:, 0] / 2.0) - mins[0]) / dx))
    ny = int(round((np.max(centers[:, 1] + sizes[:, 1] / 2.0) - mins[1]) / dy))
    nz = int(round((np.max(centers[:, 2] + sizes[:, 2] / 2.0) - mins[2]) / dz))
    tensor = np.zeros((nx, ny, nz), dtype=bool)

    for center, size, is_active in zip(centers, sizes, mask):
        if not is_active:
            continue
        x0, y0, z0 = center - size / 2.0
        x1, y1, z1 = center + size / 2.0
        ix0 = int(round((x0 - mins[0]) / dx))
        iy0 = int(round((y0 - mins[1]) / dy))
        iz0 = int(round((z0 - mins[2]) / dz))
        ix1 = int(round((x1 - mins[0]) / dx))
        iy1 = int(round((y1 - mins[1]) / dy))
        iz1 = int(round((z1 - mins[2]) / dz))
        tensor[ix0:ix1, iy0:iy1, iz0:iz1] = True

    return tensor.reshape(-1, order='F')


def _mask_tensor_cells_in_bounds(cell_centers_down, bounds):
    bounds = _coerce_bounds(bounds)
    if bounds is None:
        return None
    centers = np.asarray(cell_centers_down, dtype=float)
    return (
        (centers[:, 0] >= bounds['x_range'][0]) &
        (centers[:, 0] <= bounds['x_range'][1]) &
        (centers[:, 1] >= bounds['y_range'][0]) &
        (centers[:, 1] <= bounds['y_range'][1]) &
        (centers[:, 2] >= bounds['z_range'][0]) &
        (centers[:, 2] <= bounds['z_range'][1])
    )


def _coerce_regularization_mode(mode):
    mode = str(mode or 'smooth').lower()
    return mode if mode in ('smooth', 'compact') else 'smooth'


def _coerce_reference_model_mode(mode):
    mode = str(mode or 'property').lower()
    return mode if mode in ('property', 'zero') else 'property'


def _coerce_norms(norms, fallback=(0.0, 1.0, 1.0, 1.0)):
    if not isinstance(norms, (list, tuple)) or len(norms) != 4:
        return list(fallback)
    try:
        return [float(value) for value in norms]
    except Exception:
        return list(fallback)


def _build_reference_model(property_reference, regularization_options):
    reference_mode = _coerce_reference_model_mode(
        (regularization_options or {}).get('reference_model', 'property')
    )
    if reference_mode == 'zero':
        return np.zeros_like(np.asarray(property_reference, dtype=float)), reference_mode
    return np.asarray(property_reference, dtype=float).copy(), reference_mode


def _build_regularization(tensor_mesh, property_reference, inversion_options):
    from simpeg import directives, regularization

    regularization_options = (inversion_options or {}).get('regularization') or {}
    regularization_mode = _coerce_regularization_mode(
        regularization_options.get('mode', 'smooth')
    )
    reference_model, reference_mode = _build_reference_model(
        property_reference,
        regularization_options,
    )
    kwargs = {
        'alpha_s': float(regularization_options.get('alpha_s', 1.0)),
        'alpha_x': float(regularization_options.get('alpha_x', 1.0)),
        'alpha_y': float(regularization_options.get('alpha_y', 1.0)),
        'alpha_z': float(regularization_options.get('alpha_z', 1.0)),
        'reference_model': reference_model,
        'reference_model_in_smooth': bool(
            regularization_options.get('reference_model_in_smooth', False)
        ),
    }

    directives_list = []
    norms = None
    if regularization_mode == 'compact':
        norms = _coerce_norms(
            regularization_options.get('norms'),
            fallback=(0.0, 1.0, 1.0, 1.0),
        )
        reg = regularization.Sparse(
            tensor_mesh,
            norms=norms,
            **kwargs,
        )
        directives_list.append(
            directives.UpdateIRLS(
                max_irls_iterations=int(regularization_options.get('max_irls_iterations', 10)),
                verbose=False,
            )
        )
    else:
        reg = regularization.WeightedLeastSquares(
            tensor_mesh,
            **kwargs,
        )

    return reg, directives_list, {
        'regularization_mode': regularization_mode,
        'reference_model_mode': reference_mode,
        'reference_model_in_smooth': bool(kwargs['reference_model_in_smooth']),
        'alpha_s': float(kwargs['alpha_s']),
        'alpha_x': float(kwargs['alpha_x']),
        'alpha_y': float(kwargs['alpha_y']),
        'alpha_z': float(kwargs['alpha_z']),
        'norms': norms,
    }


def _build_constraint_state(mesh_data, tensor_mesh, mins, base_sizes, starting_density, constraint_options):
    options = constraint_options or {}
    lower_bound = float(options.get('lower_bound', -1.5))
    upper_bound = float(options.get('upper_bound', 1.5))
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    update_region = options.get('update_region') or {}
    update_mode = str(update_region.get('mode', 'all_active'))
    if update_mode not in ('all_active', 'property_only', 'manual'):
        update_mode = 'all_active'

    centers_up = tensor_mesh.cell_centers.copy()
    centers_down = centers_up.copy()
    centers_down[:, 2] *= -1.0

    if update_mode == 'property_only':
        source_mask = mesh_data.get('property_mask')
        if source_mask is None:
            density_mask = np.abs(np.asarray(mesh_data.get('density', np.zeros(mesh_data['n_cells'])))) > 1e-12
            source_mask = density_mask
        update_mask = _fill_tensor_boolean(mesh_data, source_mask, mins, base_sizes)
        if not np.any(update_mask):
            update_mode = 'all_active'

    elif update_mode == 'manual':
        update_mask = _mask_tensor_cells_in_bounds(centers_down, update_region.get('bounds'))
        if update_mask is None or not np.any(update_mask):
            update_mode = 'all_active'

    if update_mode == 'all_active':
        update_mask = np.ones(tensor_mesh.n_cells, dtype=bool)

    lower_bounds = np.full(tensor_mesh.n_cells, lower_bound, dtype=float)
    upper_bounds = np.full(tensor_mesh.n_cells, upper_bound, dtype=float)
    fixed_mask = ~update_mask
    lower_bounds[fixed_mask] = starting_density[fixed_mask]
    upper_bounds[fixed_mask] = starting_density[fixed_mask]

    return {
        'update_mode': update_mode,
        'update_mask': update_mask,
        'fixed_mask': fixed_mask,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
    }


def build_simpeg_gravity_problem(
    mesh_data,
    obs_points,
    observed_gz,
    relative_error=0.03,
    noise_floor=None,
    use_roi_mask=True,
    constraint_options=None,
):
    """构造 SimPEG 所需的网格、survey、data 和初始模型。"""
    from simpeg import data, maps
    from simpeg.potential_fields import gravity

    working_mesh = mesh_data
    used_roi_mask = False
    if use_roi_mask and 'roi_mask' in mesh_data:
        roi_mask = np.asarray(mesh_data['roi_mask'], dtype=bool)
        if np.any(roi_mask):
            working_mesh = _subset_mesh_data(mesh_data, roi_mask)
            used_roi_mask = True

    tensor_mesh, mins, _maxs, base_sizes = _build_tensor_mesh(working_mesh)
    density_model = _fill_tensor_property(working_mesh, 'density', mins, base_sizes)
    constraint_state = _build_constraint_state(
        working_mesh,
        tensor_mesh,
        mins,
        base_sizes,
        density_model,
        constraint_options=constraint_options,
    )

    obs_points = np.asarray(obs_points, dtype=float)
    rx_locations = obs_points.copy()
    rx_locations[:, 2] *= -1.0  # SimPEG 使用 z-up

    receiver = gravity.receivers.Point(rx_locations, components='gz')
    source = gravity.sources.SourceField(receiver_list=[receiver])
    survey = gravity.survey.Survey(source)

    simulation = gravity.simulation.Simulation3DIntegral(
        mesh=tensor_mesh,
        survey=survey,
        rhoMap=maps.IdentityMap(nP=tensor_mesh.n_cells),
        store_sensitivities='ram',
    )

    observed_up = -np.asarray(observed_gz, dtype=float)
    if noise_floor is None:
        noise_floor = max(1e-3, relative_error * max(np.max(np.abs(observed_up)), 1e-6))
    uncertainties = np.full_like(observed_up, noise_floor, dtype=float)
    data_obj = data.Data(survey, dobs=observed_up, standard_deviation=uncertainties)

    return {
        'mesh': tensor_mesh,
        'survey': survey,
        'simulation': simulation,
        'data': data_obj,
        'm0': density_model.copy(),
        'reference_density': density_model,
        'base_sizes': base_sizes,
        'used_roi_mask': used_roi_mask,
        'working_mesh_n_cells': int(working_mesh.get('n_cells', tensor_mesh.n_cells)),
        'constraint_state': constraint_state,
    }


def invert_gravity_simpeg(
    mesh_data,
    obs_points,
    observed_gz,
    max_iter=5,
    relative_error=0.03,
    noise_floor=None,
    use_roi_mask=True,
    constraint_options=None,
):
    """用 SimPEG 执行最小重力反演。"""
    from simpeg import (
        data_misfit,
        directives,
        inverse_problem,
        inversion,
        optimization,
    )

    problem = build_simpeg_gravity_problem(
        mesh_data=mesh_data,
        obs_points=obs_points,
        observed_gz=observed_gz,
        relative_error=relative_error,
        noise_floor=noise_floor,
        use_roi_mask=use_roi_mask,
        constraint_options=constraint_options,
    )

    simulation = problem['simulation']
    data_obj = problem['data']
    tensor_mesh = problem['mesh']

    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)
    reg, extra_directives, reg_metadata = _build_regularization(
        tensor_mesh,
        property_reference=problem['reference_density'],
        inversion_options=constraint_options,
    )
    constraint_state = problem['constraint_state']
    opt = optimization.ProjectedGNCG(
        maxIter=max_iter,
        lower=constraint_state['lower_bounds'],
        upper=constraint_state['upper_bounds'],
        cg_atol=1e-3,
        cg_rtol=0.0,
    )
    inv_problem = inverse_problem.BaseInvProblem(dmis, reg, opt)
    directive_list = [
        directives.UpdateSensitivityWeights(),
        directives.BetaEstimate_ByEig(beta0_ratio=1.0),
    ]
    directive_list.extend(extra_directives)
    directive_list.extend([
        directives.UpdatePreconditioner(),
        directives.TargetMisfit(chifact=1.0),
    ])
    inv = inversion.BaseInversion(
        inv_problem,
        directiveList=directive_list,
    )

    recovered = inv.run(problem['m0'])
    predicted_up = simulation.dpred(recovered)
    predicted_down = -np.asarray(predicted_up, dtype=float)

    centers_up = tensor_mesh.cell_centers.copy()
    centers_down = centers_up.copy()
    centers_down[:, 2] *= -1.0
    sizes = np.tile(problem['base_sizes'], (tensor_mesh.n_cells, 1))

    return {
        'predicted_gz': predicted_down,
        'observed_gz': np.asarray(observed_gz, dtype=float),
        'recovered_density': np.asarray(recovered, dtype=float),
        'recovered_mesh_data': {
            'centers': centers_down,
            'sizes': sizes,
            'density': np.asarray(recovered, dtype=float),
            'update_mask': np.asarray(constraint_state['update_mask'], dtype=bool),
            'fixed_mask': np.asarray(constraint_state['fixed_mask'], dtype=bool),
            'n_cells': tensor_mesh.n_cells,
        },
        'metadata': {
            'engine': 'simpeg',
            'max_iter': max_iter,
            'relative_error': relative_error,
            'noise_floor': float(np.asarray(problem['data'].standard_deviation)[0]),
            'used_roi_mask': bool(problem['used_roi_mask']),
            'working_mesh_n_cells': int(problem['working_mesh_n_cells']),
            'update_mode': constraint_state['update_mode'],
            'n_update_cells': int(np.count_nonzero(constraint_state['update_mask'])),
            'n_fixed_cells': int(np.count_nonzero(constraint_state['fixed_mask'])),
            'lower_bound': float(constraint_state['lower_bound']),
            'upper_bound': float(constraint_state['upper_bound']),
            **reg_metadata,
        },
    }
