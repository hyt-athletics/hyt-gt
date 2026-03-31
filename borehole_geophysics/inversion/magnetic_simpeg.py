"""
基于 SimPEG 的 3C 井中磁测反演接入

当前版本只支持：
    - 标量磁化率
    - 感应磁化
    - 3C 分量拟合（bx/by/bz）

不支持：
    - 剩磁反演
    - 矢量磁化率模型
"""

from __future__ import annotations

import numpy as np

from forward.magnetic_engine import background_unit_vector
from inversion.gravity_simpeg import (
    _base_cell_sizes,
    _build_regularization,
    _build_tensor_mesh,
    _check_simpeg,
    _coerce_bounds,
    _fill_tensor_boolean,
    _fill_tensor_property,
    _mask_tensor_cells_in_bounds,
    _subset_mesh_data,
)


def _build_constraint_state(mesh_data, tensor_mesh, mins, base_sizes, starting_susceptibility, constraint_options):
    options = constraint_options or {}
    lower_bound = float(options.get('lower_bound', 0.0))
    upper_bound = float(options.get('upper_bound', 0.08))
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
        source_mask = np.abs(np.asarray(mesh_data.get('susceptibility', np.zeros(mesh_data['n_cells'])))) > 1e-12
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
    lower_bounds[fixed_mask] = starting_susceptibility[fixed_mask]
    upper_bounds[fixed_mask] = starting_susceptibility[fixed_mask]

    return {
        'update_mode': update_mode,
        'update_mask': update_mask,
        'fixed_mask': fixed_mask,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
    }


def _split_component_vector(values, n_points, components):
    values = np.asarray(values, dtype=float).reshape(n_points, len(components), order='C')
    return {name: values[:, idx] for idx, name in enumerate(components)}


def build_simpeg_magnetic_problem(
    mesh_data,
    obs_points,
    observed_components,
    b0_strength,
    b0_inclination,
    b0_declination,
    use_roi_mask=True,
    constraint_options=None,
    components=None,
):
    """构造 SimPEG 磁法反演问题。"""
    from simpeg import data, maps
    from simpeg.potential_fields import magnetics

    if components is None:
        components = ['bx', 'by', 'bz']

    working_mesh = mesh_data
    used_roi_mask = False
    if use_roi_mask and 'roi_mask' in mesh_data:
        roi_mask = np.asarray(mesh_data['roi_mask'], dtype=bool)
        if np.any(roi_mask):
            working_mesh = _subset_mesh_data(mesh_data, roi_mask)
            used_roi_mask = True

    tensor_mesh, mins, _maxs, base_sizes = _build_tensor_mesh(working_mesh)
    susceptibility_model = _fill_tensor_property(working_mesh, 'susceptibility', mins, base_sizes)
    constraint_state = _build_constraint_state(
        working_mesh,
        tensor_mesh,
        mins,
        base_sizes,
        susceptibility_model,
        constraint_options=constraint_options,
    )

    obs_points = np.asarray(obs_points, dtype=float)
    rx_locations = obs_points.copy()
    rx_locations[:, 2] *= -1.0

    receiver = magnetics.receivers.Point(rx_locations, components=components)
    source = magnetics.sources.UniformBackgroundField(
        receiver_list=[receiver],
        amplitude=float(b0_strength),
        inclination=float(b0_inclination),
        declination=float(b0_declination),
    )
    survey = magnetics.survey.Survey(source)

    simulation = magnetics.simulation.Simulation3DIntegral(
        mesh=tensor_mesh,
        survey=survey,
        chiMap=maps.IdentityMap(nP=tensor_mesh.n_cells),
        model_type='scalar',
        store_sensitivities='ram',
        engine='geoana',
    )

    obs_bx = np.asarray(observed_components['bx'], dtype=float)
    obs_by = np.asarray(observed_components['by'], dtype=float)
    obs_bz_up = -np.asarray(observed_components['bz'], dtype=float)
    observed_vector = np.column_stack([obs_bx, obs_by, obs_bz_up]).reshape(-1, order='C')

    max_abs = max(np.max(np.abs(observed_vector)), 1e-6)
    uncertainties = np.full_like(observed_vector, 0.03 * max_abs, dtype=float)
    data_obj = data.Data(survey, dobs=observed_vector, standard_deviation=uncertainties)

    return {
        'mesh': tensor_mesh,
        'survey': survey,
        'simulation': simulation,
        'data': data_obj,
        'm0': susceptibility_model.copy(),
        'reference_susceptibility': susceptibility_model,
        'base_sizes': base_sizes,
        'used_roi_mask': used_roi_mask,
        'working_mesh_n_cells': int(working_mesh.get('n_cells', tensor_mesh.n_cells)),
        'constraint_state': constraint_state,
        'components': components,
    }


def invert_magnetic_simpeg(
    mesh_data,
    obs_points,
    observed_bx,
    observed_by,
    observed_bz,
    b0_strength,
    b0_inclination,
    b0_declination,
    max_iter=8,
    use_roi_mask=True,
    constraint_options=None,
):
    """用 SimPEG 执行 3C 井中磁测磁化率反演。"""
    from simpeg import (
        data_misfit,
        directives,
        inverse_problem,
        inversion,
        optimization,
    )

    problem = build_simpeg_magnetic_problem(
        mesh_data=mesh_data,
        obs_points=obs_points,
        observed_components={
            'bx': observed_bx,
            'by': observed_by,
            'bz': observed_bz,
        },
        b0_strength=b0_strength,
        b0_inclination=b0_inclination,
        b0_declination=b0_declination,
        use_roi_mask=use_roi_mask,
        constraint_options=constraint_options,
        components=['bx', 'by', 'bz'],
    )

    simulation = problem['simulation']
    data_obj = problem['data']
    tensor_mesh = problem['mesh']
    constraint_state = problem['constraint_state']

    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)
    reg, extra_directives, reg_metadata = _build_regularization(
        tensor_mesh,
        property_reference=problem['reference_susceptibility'],
        inversion_options=constraint_options,
    )
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
    predicted_vector = simulation.dpred(recovered)
    predicted_split = _split_component_vector(
        predicted_vector,
        n_points=len(obs_points),
        components=problem['components'],
    )

    predicted_bx = np.asarray(predicted_split['bx'], dtype=float)
    predicted_by = np.asarray(predicted_split['by'], dtype=float)
    predicted_bz = -np.asarray(predicted_split['bz'], dtype=float)
    direction = background_unit_vector(b0_inclination, b0_declination)
    predicted_bt = (
        predicted_bx * direction[0] +
        predicted_by * direction[1] +
        predicted_bz * direction[2]
    )
    observed_bt = (
        np.asarray(observed_bx, dtype=float) * direction[0] +
        np.asarray(observed_by, dtype=float) * direction[1] +
        np.asarray(observed_bz, dtype=float) * direction[2]
    )

    centers_up = tensor_mesh.cell_centers.copy()
    centers_down = centers_up.copy()
    centers_down[:, 2] *= -1.0
    sizes = np.tile(problem['base_sizes'], (tensor_mesh.n_cells, 1))

    return {
        'observed_bx': np.asarray(observed_bx, dtype=float),
        'observed_by': np.asarray(observed_by, dtype=float),
        'observed_bz': np.asarray(observed_bz, dtype=float),
        'observed_bt': np.asarray(observed_bt, dtype=float),
        'predicted_bx': predicted_bx,
        'predicted_by': predicted_by,
        'predicted_bz': predicted_bz,
        'predicted_bt': predicted_bt,
        'recovered_mesh_data': {
            'centers': centers_down,
            'sizes': sizes,
            'susceptibility': np.asarray(recovered, dtype=float),
            'update_mask': np.asarray(constraint_state['update_mask'], dtype=bool),
            'fixed_mask': np.asarray(constraint_state['fixed_mask'], dtype=bool),
            'n_cells': tensor_mesh.n_cells,
        },
        'metadata': {
            'engine': 'simpeg',
            'max_iter': max_iter,
            'used_roi_mask': bool(problem['used_roi_mask']),
            'working_mesh_n_cells': int(problem['working_mesh_n_cells']),
            'update_mode': constraint_state['update_mode'],
            'n_update_cells': int(np.count_nonzero(constraint_state['update_mask'])),
            'n_fixed_cells': int(np.count_nonzero(constraint_state['fixed_mask'])),
            'lower_bound': float(constraint_state['lower_bound']),
            'upper_bound': float(constraint_state['upper_bound']),
            'components': list(problem['components']),
            'assumptions': 'scalar susceptibility + induced magnetization only',
            **reg_metadata,
        },
    }
