"""
重力方法适配器
"""

from __future__ import annotations

from adapters.base import MethodAdapter
from forward.gravity_engine import forward_gravity
from forward.result import ForwardResult
from inversion.gravity_result import GravityInversionResult
from inversion.gravity_simpeg import _check_simpeg, invert_gravity_simpeg


class GravityAdapter(MethodAdapter):
    method_name = "井中重力"
    method_code = "gravity"
    required_properties = ['density']
    optional_properties = []

    def create_survey(self, well, **kwargs):
        return {
            'well': well,
            'obs_points': well.stations.copy(),
        }

    def get_mesh_requirements(self):
        return {
            'strategy_family': 'potential_field',
            'borehole_refinement': True,
            'boundary_refinement': True,
            'body_core_refinement': True,
            'solver': 'integral',
        }

    def prepare_mesh(self, mesh_data, survey):
        return mesh_data

    def prepare_model(self, mesh_data):
        return mesh_data['density']

    def forward(self, mesh, model, survey, **kwargs):
        engine = kwargs.get('engine', 'auto')
        gz = forward_gravity(
            survey['obs_points'],
            mesh['centers'],
            mesh['sizes'],
            model,
            engine=engine,
        )
        return {'gz': gz}

    def inverse(self, mesh, data, survey, **kwargs):
        if not _check_simpeg():
            raise RuntimeError("SimPEG 未安装，无法执行重力反演")

        settings = kwargs.get('settings')
        inversion_settings = settings.get('inversion.gravity', {}) if settings else {}

        max_iter = kwargs.get('max_iter', inversion_settings.get('max_iter', 5))
        relative_error = kwargs.get('relative_error', inversion_settings.get('relative_error', 0.03))
        noise_floor = kwargs.get('noise_floor', inversion_settings.get('noise_floor'))
        use_roi_mask = kwargs.get('use_roi_mask', True)
        constraint_options = kwargs.get('constraint_options', inversion_settings)

        raw = invert_gravity_simpeg(
            mesh_data=mesh,
            obs_points=survey['obs_points'],
            observed_gz=data.gz,
            max_iter=max_iter,
            relative_error=relative_error,
            noise_floor=noise_floor,
            use_roi_mask=use_roi_mask,
            constraint_options=constraint_options,
        )
        return GravityInversionResult(
            well=survey['well'],
            depths=survey['obs_points'][:, 2],
            observed_gz=raw['observed_gz'],
            predicted_gz=raw['predicted_gz'],
            recovered_mesh_data=raw['recovered_mesh_data'],
            metadata=raw.get('metadata'),
        )

    def convert_result(self, raw_result, **kwargs):
        well = kwargs['well']
        mesh_data = kwargs.get('mesh_data')
        depths = kwargs['survey']['obs_points'][:, 2]
        return ForwardResult(
            well=well,
            depths=depths,
            gz=raw_result['gz'],
            mesh_data=mesh_data,
        )

    def check_dependencies(self, operation='forward'):
        if operation == 'inverse' and not _check_simpeg():
            return False, "SimPEG 未安装，无法执行重力反演"
        return True, "ok"
