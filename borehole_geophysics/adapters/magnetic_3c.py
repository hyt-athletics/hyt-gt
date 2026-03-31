"""
三分量井中磁测适配器

平台只负责建模、网格和交互；
正演/反演求解交给 harmonica 与 SimPEG。
"""

from __future__ import annotations

from adapters.base import MethodAdapter
from forward.magnetic_engine import _check_harmonica, forward_magnetic
from forward.magnetic_result import MagneticForwardResult
from inversion.gravity_simpeg import _check_simpeg
from inversion.magnetic_result import MagneticInversionResult
from inversion.magnetic_simpeg import invert_magnetic_simpeg


class Magnetic3CAdapter(MethodAdapter):
    method_name = "3C井中磁测"
    method_code = "magnetic_3c"
    required_properties = ['susceptibility']
    optional_properties = [
        'remanent_mag',
        'remanent_inclination',
        'remanent_declination',
    ]

    def create_survey(self, well, **kwargs):
        settings = kwargs.get('settings')
        return {
            'well': well,
            'obs_points': well.stations.copy(),
            'b0_strength': settings.get('magnetic.b0_strength', 52000) if settings else 52000,
            'b0_inclination': settings.get('magnetic.b0_inclination', 55.0) if settings else 55.0,
            'b0_declination': settings.get('magnetic.b0_declination', -6.0) if settings else -6.0,
        }

    def get_mesh_requirements(self):
        return {
            'strategy_family': 'potential_field',
            'borehole_refinement': True,
            'boundary_refinement': True,
            'body_core_refinement': True,
            'solver': 'external_library_only',
        }

    def prepare_mesh(self, mesh_data, survey):
        return mesh_data

    def prepare_model(self, mesh_data):
        return {
            'susceptibility': mesh_data['susceptibility'],
            'remanent_mag': mesh_data.get('remanent_mag'),
            'remanent_inclination': mesh_data.get('remanent_inclination'),
            'remanent_declination': mesh_data.get('remanent_declination'),
        }

    def forward(self, mesh, model, survey, **kwargs):
        engine = kwargs.get('engine', 'auto')
        return forward_magnetic(
            obs_points=survey['obs_points'],
            centers=mesh['centers'],
            sizes=mesh['sizes'],
            susceptibilities=model['susceptibility'],
            b0_strength=survey['b0_strength'],
            b0_inclination=survey['b0_inclination'],
            b0_declination=survey['b0_declination'],
            remanent_mag=model.get('remanent_mag'),
            remanent_inclination=model.get('remanent_inclination'),
            remanent_declination=model.get('remanent_declination'),
            engine=engine,
        )

    def inverse(self, mesh, data, survey, **kwargs):
        if not _check_simpeg():
            raise RuntimeError("SimPEG 未安装，无法执行 3C 井中磁测反演")

        settings = kwargs.get('settings')
        inversion_settings = settings.get('inversion.magnetic_3c', {}) if settings else {}
        max_iter = kwargs.get('max_iter', inversion_settings.get('max_iter', 8))
        use_roi_mask = kwargs.get('use_roi_mask', True)
        constraint_options = kwargs.get('constraint_options', inversion_settings)

        raw = invert_magnetic_simpeg(
            mesh_data=mesh,
            obs_points=survey['obs_points'],
            observed_bx=data.bx,
            observed_by=data.by,
            observed_bz=data.bz,
            b0_strength=survey['b0_strength'],
            b0_inclination=survey['b0_inclination'],
            b0_declination=survey['b0_declination'],
            max_iter=max_iter,
            use_roi_mask=use_roi_mask,
            constraint_options=constraint_options,
        )
        return MagneticInversionResult(
            well=survey['well'],
            depths=survey['obs_points'][:, 2],
            observed_bx=raw['observed_bx'],
            observed_by=raw['observed_by'],
            observed_bz=raw['observed_bz'],
            observed_bt=raw['observed_bt'],
            predicted_bx=raw['predicted_bx'],
            predicted_by=raw['predicted_by'],
            predicted_bz=raw['predicted_bz'],
            predicted_bt=raw['predicted_bt'],
            recovered_mesh_data=raw['recovered_mesh_data'],
            survey=survey,
            metadata=raw.get('metadata'),
        )

    def convert_result(self, raw_result, **kwargs):
        well = kwargs['well']
        survey = kwargs.get('survey', {})
        mesh_data = kwargs.get('mesh_data')
        return MagneticForwardResult(
            well=well,
            depths=survey['obs_points'][:, 2],
            bx=raw_result['bx'],
            by=raw_result['by'],
            bz=raw_result['bz'],
            bt=raw_result['bt'],
            mesh_data=mesh_data,
            survey=survey,
        )

    def check_dependencies(self, operation='forward'):
        if operation == 'inverse':
            if not _check_simpeg():
                return False, "SimPEG 未安装，无法执行 3C 井中磁测反演"
            return True, "ok"
        if not _check_harmonica():
            return False, "harmonica 未安装，无法进行三分量井中磁测正演"
        return True, "ok"
