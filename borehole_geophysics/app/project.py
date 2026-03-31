"""
项目管理器

统一管理：模型数据 + 钻孔 + 网格 + 正演结果

一个"项目"包含做一次建模所需的所有东西
"""

import os
import json
import numpy as np
from datetime import datetime

from app.settings import Settings
from interactive.body_manager import (
    BODY_PROPERTY_DEFAULTS,
    ModelManager,
    GeologicalBody,
)
from borehole.well import VerticalWell


RESULT_ATTRS = {
    'gravity': 'gravity_result',
    'magnetic_3c': 'magnetic_result',
}

OBSERVED_LOADERS = {
    'gravity': ('forward.result', 'ForwardResult'),
    'magnetic_3c': ('forward.magnetic_result', 'MagneticForwardResult'),
}

INVERSE_LOADERS = {
    'gravity': ('inversion.gravity_result', 'GravityInversionResult'),
    'magnetic_3c': ('inversion.magnetic_result', 'MagneticInversionResult'),
}


class Project:
    """
    项目对象 = 一次完整的建模过程
    
    包含：
        - 设置（域/钻孔/网格参数）
        - 地质体列表
        - 地层线列表
        - 钻孔
        - 网格（生成后）
        - 正演结果（计算后）
    """
    
    def __init__(self, name="untitled"):
        self.name = name
        self.settings = Settings()
        
        # 模型
        self.model_manager = ModelManager()
        if not hasattr(self.model_manager, 'layers'):
            self.model_manager.layers = []
        self._sync_settings_to_model()
        
        # 钻孔
        self.well = None
        self._create_well()
        
        # 网格（生成后才有）
        self.mesh = None
        self.mesh_data = None
        self.mesh_method = None
        self.mesh_purpose = None
        
        # 正演结果
        self.gravity_result = None
        self.magnetic_result = None
        self.current_result = None
        self.current_result_method = None
        self.result_history = []
        self.result_history_by_method = {method: [] for method in RESULT_ATTRS}
        self.result_counters = {method: 0 for method in RESULT_ATTRS}
        self.observed_results = {method: None for method in RESULT_ATTRS}
        self.inverse_result = None
        self.inverse_result_method = None
        self.inverse_latest_by_method = {method: None for method in RESULT_ATTRS}
        self.inverse_history = []
        self.inverse_history_by_method = {method: [] for method in RESULT_ATTRS}
        self.inverse_counters = {method: 0 for method in RESULT_ATTRS}
    
    def _sync_settings_to_model(self):
        """把设置同步到模型管理器"""
        domain = self.settings.get('domain')
        self.model_manager.domain = {
            'x_range': tuple(domain['x_range']),
            'y_range': tuple(domain['y_range']),
            'z_range': tuple(domain['z_range']),
        }
        self.model_manager.well_info = self.settings.get('well')
    
    def _create_well(self):
        """根据设置创建钻孔"""
        w = self.settings.get('well')
        self.well = VerticalWell(
            x=w['x'], y=w['y'],
            z_top=w['z_top'], z_bottom=w['z_bottom'],
            n_stations=w['n_stations']
        )
    
    def update_settings(self):
        """设置改变后刷新相关对象"""
        self._sync_settings_to_model()
        self._create_well()

    def get_current_method(self):
        """当前选中的地球物理方法。"""
        return self.settings.get('forward.method', 'gravity')

    def _collect_all_bodies(self):
        """合并显式地质体和由地层线生成的体。"""
        from interactive.layer_tool import layers_to_bodies
        
        domain = self.model_manager.domain
        all_bodies = list(self.model_manager.bodies)
        if getattr(self.model_manager, 'layers', None):
            layer_bodies = layers_to_bodies(
                self.model_manager.layers,
                domain['x_range'], domain['z_range']
            )
            all_bodies.extend(layer_bodies)
        return all_bodies

    def _assign_body_properties(self, body):
        """把地质体物性写入网格叶节点。"""
        for prop_name, default in BODY_PROPERTY_DEFAULTS.items():
            value = float(getattr(body, prop_name, default))
            if abs(value - float(default)) > 1e-12:
                self.mesh.assign_property(body.contains_point_3d, prop_name, value)

    def _label_result(self, method_code, result):
        """给结果打上方法和运行标签。"""
        counter = self.result_counters.get(method_code, 0) + 1
        self.result_counters[method_code] = counter
        timestamp = datetime.now().strftime('%H:%M:%S')
        setattr(result, 'method_code', method_code)
        setattr(result, 'run_index', counter)
        setattr(result, 'created_at', timestamp)
        setattr(result, 'label', f'{method_code}#{counter:02d} @ {timestamp}')

    def _label_inverse_result(self, method_code, result):
        """给反演结果打上方法和运行标签。"""
        counter = self.inverse_counters.get(method_code, 0) + 1
        self.inverse_counters[method_code] = counter
        timestamp = datetime.now().strftime('%H:%M:%S')
        setattr(result, 'method_code', method_code)
        setattr(result, 'run_index', counter)
        setattr(result, 'created_at', timestamp)
        setattr(result, 'label', f'{method_code}_inv#{counter:02d} @ {timestamp}')

    def _get_method_result_attr(self, method_code):
        """方法对应的项目属性名。"""
        return RESULT_ATTRS.get(method_code)

    def _get_method_current_result(self, method_code):
        """获取某个方法当前保存的最新结果。"""
        attr_name = self._get_method_result_attr(method_code)
        if attr_name is None:
            return None
        return getattr(self, attr_name, None)

    def _set_method_current_result(self, method_code, result):
        """设置某个方法当前保存的最新结果。"""
        attr_name = self._get_method_result_attr(method_code)
        if attr_name is not None:
            setattr(self, attr_name, result)

    def _remove_from_global_history(self, result):
        """从全局历史里移除一个结果对象。"""
        self.result_history = [item for item in self.result_history if item is not result]

    def _remove_from_inverse_global_history(self, result):
        """从全局反演历史里移除一个结果对象。"""
        self.inverse_history = [item for item in self.inverse_history if item is not result]

    def _append_to_history(self, method_code, result):
        """把结果追加到历史列表，避免重复。"""
        if result is None:
            return
        if result not in self.result_history:
            self.result_history.append(result)
        history = self.result_history_by_method.setdefault(method_code, [])
        if result not in history:
            history.append(result)

    def _append_to_inverse_history(self, method_code, result):
        """把反演结果追加到历史列表，避免重复。"""
        if result is None:
            return
        if result not in self.inverse_history:
            self.inverse_history.append(result)
        history = self.inverse_history_by_method.setdefault(method_code, [])
        if result not in history:
            history.append(result)

    def _remove_from_method_history(self, method_code, result):
        """从某个方法的历史里移除结果。"""
        history = self.result_history_by_method.get(method_code, [])
        self.result_history_by_method[method_code] = [
            item for item in history if item is not result
        ]

    def _remove_from_inverse_method_history(self, method_code, result):
        """从某个方法的反演历史里移除结果。"""
        history = self.inverse_history_by_method.get(method_code, [])
        self.inverse_history_by_method[method_code] = [
            item for item in history if item is not result
        ]

    def _store_result(self, method_code, result):
        """同步更新统一结果引用和方法专属结果。"""
        previous_for_method = self._get_method_current_result(method_code)
        if previous_for_method is not None and previous_for_method is not result:
            self._append_to_history(method_code, previous_for_method)

        self._label_result(method_code, result)
        self._remove_from_method_history(method_code, result)
        self._remove_from_global_history(result)
        self._set_method_current_result(method_code, result)
        self.current_result = result
        self.current_result_method = method_code

    def _store_inverse_result(self, method_code, result):
        """同步更新当前反演结果和反演历史。"""
        previous_for_method = self.inverse_latest_by_method.get(method_code)
        if previous_for_method is not None and previous_for_method is not result:
            self._append_to_inverse_history(method_code, previous_for_method)

        self._label_inverse_result(method_code, result)
        self._remove_from_inverse_method_history(method_code, result)
        self._remove_from_inverse_global_history(result)
        self.inverse_latest_by_method[method_code] = result
        self.inverse_result = result
        self.inverse_result_method = method_code

    def get_results_for_method(self, method_code=None, include_current=True):
        """获取某个方法的历史结果列表。"""
        method_code = method_code or self.get_current_method()
        results = list(self.result_history_by_method.get(method_code, []))
        current = self._get_method_current_result(method_code)
        if include_current and current is not None:
            results.append(current)
        return results

    def get_result_catalog(self):
        """返回所有方法的结果目录。"""
        catalog = []
        for method_code in RESULT_ATTRS:
            results = self.get_results_for_method(method_code, include_current=True)
            for index, result in enumerate(results):
                catalog.append({
                    'method_code': method_code,
                    'index': index,
                    'label': getattr(result, 'label', f'{method_code}#{index + 1}'),
                    'is_active': result is self.current_result,
                    'is_latest_for_method': result is self._get_method_current_result(method_code),
                })
        return catalog

    def set_current_result_by_index(self, method_code, index):
        """把某个方法的指定结果设为当前活动结果。"""
        results = self.get_results_for_method(method_code, include_current=True)
        if not (0 <= index < len(results)):
            raise IndexError("结果索引越界")
        result = results[index]
        latest = self._get_method_current_result(method_code)
        if latest is not None and latest is not result:
            self._append_to_history(method_code, latest)
        self._remove_from_method_history(method_code, result)
        self._remove_from_global_history(result)
        self._set_method_current_result(method_code, result)
        self.current_result = result
        self.current_result_method = method_code
        self.settings.set('forward.method', method_code)
        return result

    def delete_result_by_index(self, method_code, index):
        """删除某个方法的指定结果。"""
        history = self.result_history_by_method.get(method_code, [])
        latest = self._get_method_current_result(method_code)
        n_history = len(history)
        if not (0 <= index < n_history + (1 if latest is not None else 0)):
            raise IndexError("结果索引越界")

        if index < n_history:
            removed = history.pop(index)
        else:
            removed = latest
            promoted = history.pop(-1) if history else None
            self._remove_from_global_history(promoted)
            self._set_method_current_result(method_code, promoted)
            if self.current_result is removed:
                self.current_result = promoted
                self.current_result_method = method_code if promoted is not None else None

        self._remove_from_global_history(removed)

        if self.current_result is removed:
            self.current_result = None
            self.current_result_method = None

        return removed

    def get_inverse_results_for_method(self, method_code=None, include_current=True):
        """获取某个方法的反演结果列表。"""
        method_code = method_code or self.get_current_method()
        results = list(self.inverse_history_by_method.get(method_code, []))
        latest = self.inverse_latest_by_method.get(method_code)
        if include_current and latest is not None:
            results.append(latest)
        return results

    def get_inverse_result_catalog(self):
        """返回所有方法的反演结果目录。"""
        catalog = []
        for method_code in RESULT_ATTRS:
            results = self.get_inverse_results_for_method(method_code, include_current=True)
            for index, result in enumerate(results):
                catalog.append({
                    'method_code': method_code,
                    'index': index,
                    'label': getattr(result, 'label', f'{method_code}_inv#{index + 1}'),
                    'is_active': result is self.inverse_result,
                    'is_latest_for_method': result is self.inverse_latest_by_method.get(method_code),
                })
        return catalog

    def set_current_inverse_result_by_index(self, method_code, index):
        """把某个方法的指定反演结果设为当前活动反演结果。"""
        results = self.get_inverse_results_for_method(method_code, include_current=True)
        if not (0 <= index < len(results)):
            raise IndexError("反演结果索引越界")
        result = results[index]
        latest = self.inverse_latest_by_method.get(method_code)
        if latest is not None and latest is not result:
            self._append_to_inverse_history(method_code, latest)
        self._remove_from_inverse_method_history(method_code, result)
        self._remove_from_inverse_global_history(result)
        self.inverse_latest_by_method[method_code] = result
        self.inverse_result = result
        self.inverse_result_method = method_code
        self.settings.set('forward.method', method_code)
        return result

    def delete_inverse_result_by_index(self, method_code, index):
        """删除某个方法的指定反演结果。"""
        history = self.inverse_history_by_method.get(method_code, [])
        latest = self.inverse_latest_by_method.get(method_code)
        n_history = len(history)
        if not (0 <= index < n_history + (1 if latest is not None else 0)):
            raise IndexError("反演结果索引越界")

        if index < n_history:
            removed = history.pop(index)
        else:
            removed = latest
            promoted = history.pop(-1) if history else None
            self._remove_from_inverse_global_history(promoted)
            self.inverse_latest_by_method[method_code] = promoted
            if self.inverse_result is removed:
                self.inverse_result = promoted
                self.inverse_result_method = method_code if promoted is not None else None

        self._remove_from_inverse_global_history(removed)

        if self.inverse_result is removed:
            self.inverse_result = None
            self.inverse_result_method = None
        if self.inverse_latest_by_method.get(method_code) is removed:
            self.inverse_latest_by_method[method_code] = None

        return removed

    def get_observed_result(self, method_code=None):
        """获取某个方法对应的实测/导入数据。"""
        method_code = method_code or self.get_current_method()
        return self.observed_results.get(method_code)

    def set_observed_result(self, method_code, result):
        """登记某个方法的实测数据。"""
        setattr(result, 'method_code', method_code)
        setattr(result, 'is_observed', True)
        if not hasattr(result, 'label'):
            setattr(result, 'label', f'observed_{method_code}')
        self.observed_results[method_code] = result
        self.settings.set('forward.method', method_code)
        return result

    def load_observed_result(self, method_code, filename):
        """从 CSV 导入实测数据。"""
        import importlib

        loader_info = OBSERVED_LOADERS.get(method_code)
        if loader_info is None:
            raise ValueError(f"未注册实测数据加载器: {method_code}")
        module_name, class_name = loader_info
        module = importlib.import_module(module_name)
        result_cls = getattr(module, class_name)
        label = f"observed_{method_code}:{os.path.basename(filename)}"
        result = result_cls.load_csv(filename, well=self.well, label=label)
        return self.set_observed_result(method_code, result)

    def load_inverse_result(self, method_code, filename):
        """从 NPZ 导入反演结果。"""
        import importlib

        loader_info = INVERSE_LOADERS.get(method_code)
        if loader_info is None:
            raise ValueError(f"未注册反演结果加载器: {method_code}")
        module_name, class_name = loader_info
        module = importlib.import_module(module_name)
        result_cls = getattr(module, class_name)
        label = f"{method_code}_inv_loaded:{os.path.basename(filename)}"
        result = result_cls.load_npz(filename, well=self.well, label=label)
        self._store_inverse_result(method_code, result)
        return result
    
    # ==========================================================
    #  网格生成
    # ==========================================================
    
    def _generate_mesh_internal(self, method_code, purpose='forward'):
        """按方法自动规划并生成网格。"""
        from mesh.octree import OctreeMesh
        from mesh.strategy import (
            annotate_potential_field_mesh,
            build_potential_field_mesh_plan,
        )
        from adapters.registry import get_adapter
        
        print("\n  生成网格...")
        
        domain = self.model_manager.domain
        mesh_settings = self.settings.get('mesh')
        all_bodies = self._collect_all_bodies()
        adapter = get_adapter(method_code)
        mesh_requirements = adapter.get_mesh_requirements()

        if mesh_requirements.get('strategy_family') == 'potential_field':
            plan = build_potential_field_mesh_plan(
                domain=domain,
                well=self.well,
                bodies=all_bodies,
                mesh_settings=mesh_settings,
                method_code=method_code,
                purpose=purpose,
            )
        else:
            plan = None

        max_level = plan.max_level if plan is not None else mesh_settings['max_level']
        
        self.mesh = OctreeMesh(
            x_range=domain['x_range'],
            y_range=domain['y_range'],
            z_range=domain['z_range'],
            max_level=max_level
        )

        if plan is not None:
            print(
                f"  自动网格策略: target_cell≈{plan.target_cell_size:.1f}m, "
                f"max_level={plan.max_level}, station_spacing={plan.station_spacing:.1f}m, "
                f"purpose={purpose}"
            )
            self.mesh.refine_along_borehole(
                self.well,
                levels_and_radii=plan.borehole_refine_levels,
            )
            for body in all_bodies:
                padded = body.bounds_3d(padding=plan.body_padding_distance)
                self.mesh.refine_in_box(
                    padded['x_range'],
                    padded['y_range'],
                    padded['z_range'],
                    target_level=plan.body_padding_refine_level,
                )
                self.mesh.refine_at_boundary(
                    body.contains_point_3d,
                    target_level=plan.boundary_refine_level,
                )
                core = body.bounds_3d(padding=0.0)
                self.mesh.refine_in_box(
                    core['x_range'],
                    core['y_range'],
                    core['z_range'],
                    target_level=plan.body_core_refine_level,
                )
        else:
            refine_config = mesh_settings['borehole_refine_levels']
            levels_radii = [(int(lr[0]), float(lr[1])) for lr in refine_config]
            self.mesh.refine_along_borehole(self.well, levels_and_radii=levels_radii)

            bnd_level = mesh_settings['boundary_refine_level']
            for body in all_bodies:
                self.mesh.refine_at_boundary(body.contains_point_3d, target_level=bnd_level)
        
        self.mesh.balance()
        
        # 赋物性
        for body in all_bodies:
            self._assign_body_properties(body)
        
        self.mesh_data = self.mesh.to_arrays()
        self.mesh_method = method_code
        self.mesh_purpose = purpose
        if plan is not None:
            self.mesh_data['mesh_strategy'] = plan.to_dict()
            self.mesh_data.update(
                annotate_potential_field_mesh(
                    self.mesh_data,
                    plan=plan,
                    well=self.well,
                )
            )
        self.mesh.get_info()
        
        return self.mesh

    def generate_mesh(self, method_code=None, purpose='forward'):
        """根据当前方法生成网格。"""
        return self.generate_mesh_for_method(method_code or self.get_current_method(), purpose=purpose)

    def generate_mesh_for_method(self, method_code=None, purpose='forward'):
        """按方法需求生成网格，使用方法感知 refinement planner。"""
        method_code = method_code or self.get_current_method()
        self.settings.set('forward.method', method_code)
        return self._generate_mesh_internal(method_code, purpose=purpose)
    
    # ==========================================================
    #  正演计算
    # ==========================================================
    
    def compute_gravity(self):
        """计算重力正演"""
        return self.run_forward('gravity')

    def run_forward(self, method_code=None):
        """通过方法适配器执行正演。"""
        from adapters.registry import get_adapter
        
        method_code = method_code or self.get_current_method()
        self.settings.set('forward.method', method_code)
        
        if (
            self.mesh_data is None or
            self.mesh_method != method_code or
            self.mesh_purpose != 'forward'
        ):
            self.generate_mesh_for_method(method_code, purpose='forward')
        
        adapter = get_adapter(method_code)
        ok, msg = adapter.check_dependencies(operation='forward')
        if not ok:
            raise RuntimeError(msg)
        
        survey = adapter.create_survey(self.well, settings=self.settings)
        prepared_mesh = adapter.prepare_mesh(self.mesh_data, survey)
        prepared_model = adapter.prepare_model(self.mesh_data)
        raw_result = adapter.forward(
            prepared_mesh,
            prepared_model,
            survey,
            settings=self.settings,
            engine=self.settings.get('forward.engine', 'auto')
        )
        result = adapter.convert_result(
            raw_result,
            well=self.well,
            survey=survey,
            mesh_data=self.mesh_data,
        )
        self._store_result(method_code, result)
        
        if hasattr(result, 'get_summary'):
            result.get_summary()
        return result

    def run_inverse(self, method_code=None, **kwargs):
        """通过方法适配器执行反演。"""
        from adapters.registry import get_adapter
        
        method_code = method_code or self.get_current_method()
        adapter = get_adapter(method_code)
        ok, msg = adapter.check_dependencies(operation='inverse')
        if not ok:
            raise RuntimeError(msg)
        
        data = kwargs.pop('data', None)
        if data is None:
            data = self.get_observed_result(method_code) or self.current_result
        if data is None:
            raise RuntimeError("请先导入观测数据或准备当前结果")
        if (
            self.mesh_data is None or
            self.mesh_method != method_code or
            self.mesh_purpose != 'inverse'
        ):
            self.generate_mesh_for_method(method_code, purpose='inverse')
        
        survey = adapter.create_survey(self.well, settings=self.settings)
        prepared_mesh = adapter.prepare_mesh(self.mesh_data, survey)
        result = adapter.inverse(
            prepared_mesh,
            data,
            survey,
            settings=self.settings,
            **kwargs,
        )
        self._store_inverse_result(method_code, result)
        if hasattr(result, 'get_summary'):
            result.get_summary()
        return result
    
    # ==========================================================
    #  保存/加载
    # ==========================================================
    
    def save(self, directory=None):
        """保存整个项目"""
        if directory is None:
            directory = f"project_{self.name}"
        
        os.makedirs(directory, exist_ok=True)
        
        # 保存设置
        self.settings.filename = os.path.join(directory, 'settings.json')
        self.settings.save()
        
        # 保存模型
        model_file = os.path.join(directory, 'model.json')
        self.model_manager.save(model_file)
        
        # 保存正演结果
        if self.current_result is not None and hasattr(self.current_result, 'save_csv'):
            result_name = f'{self.current_result_method or "forward"}_result.csv'
            result_file = os.path.join(directory, result_name)
            self.current_result.save_csv(result_file)

        # 保存导入的实测数据
        for method_code, observed in self.observed_results.items():
            if observed is None or not hasattr(observed, 'save_csv'):
                continue
            observed_file = os.path.join(directory, f'observed_{method_code}.csv')
            observed.save_csv(observed_file)

        if self.inverse_result is not None and hasattr(self.inverse_result, 'save_csv'):
            inv_name = f'{self.inverse_result_method or "inverse"}_inverse_result.csv'
            inv_file = os.path.join(directory, inv_name)
            self.inverse_result.save_csv(inv_file)
        if self.inverse_result is not None and hasattr(self.inverse_result, 'save_npz'):
            inv_npz_name = f'{self.inverse_result_method or "inverse"}_inverse_result.npz'
            inv_npz_file = os.path.join(directory, inv_npz_name)
            self.inverse_result.save_npz(inv_npz_file)
        
        print(f"  ✅ 项目已保存: {os.path.abspath(directory)}/")
    
    def load(self, directory):
        """加载项目"""
        if not os.path.isdir(directory):
            print(f"  ❌ 目录不存在: {directory}")
            return False
        
        settings_file = os.path.join(directory, 'settings.json')
        if os.path.exists(settings_file):
            self.settings = Settings(settings_file)
            self.update_settings()
        
        model_file = os.path.join(directory, 'model.json')
        if os.path.exists(model_file):
            self.model_manager.load(model_file)

        for method_code in RESULT_ATTRS:
            observed_file = os.path.join(directory, f'observed_{method_code}.csv')
            if os.path.exists(observed_file):
                try:
                    self.load_observed_result(method_code, observed_file)
                except Exception as exc:
                    print(f"  ⚠️ 加载实测数据失败 ({method_code}): {exc}")

        for method_code in INVERSE_LOADERS:
            inverse_file = os.path.join(directory, f'{method_code}_inverse_result.npz')
            if os.path.exists(inverse_file):
                try:
                    self.load_inverse_result(method_code, inverse_file)
                except Exception as exc:
                    print(f"  ⚠️ 加载反演结果失败 ({method_code}): {exc}")
        
        self.name = os.path.basename(directory).replace('project_', '')
        print(f"  ✅ 项目已加载: {directory}")
        return True
    
    # ==========================================================
    #  信息
    # ==========================================================
    
    def get_summary(self):
        """打印项目摘要"""
        n_bodies = len(self.model_manager.bodies)
        n_layers = len(getattr(self.model_manager, 'layers', []))
        has_mesh = self.mesh_data is not None
        has_result = self.current_result is not None
        has_inverse = self.inverse_result is not None
        mesh_line = '❌ 未生成'
        qc_line = '—'
        if has_mesh:
            mesh_line = '✅ 已生成 (' + str(self.mesh_data["n_cells"]) + ' 格子)'
            mesh_qc = self.mesh_data.get('mesh_qc', {})
            if mesh_qc:
                qc_line = (
                    f"ROI={mesh_qc.get('roi_cells', 0)}格, "
                    f"近井={mesh_qc.get('near_well_cells', 0)}格, "
                    f"min/target={mesh_qc.get('min_cell_size', 0.0):.1f}/"
                    f"{mesh_qc.get('target_cell_size', 0.0):.1f}m"
                )
                mesh_line += f' [{self.mesh_method}/{self.mesh_purpose}]'
        
        print(f"""
    ╔══════════════════════════════════════════════╗
    ║  项目: {self.name:38s} ║
    ╠══════════════════════════════════════════════╣
    ║  地质体: {n_bodies} 个                               
    ║  地层线: {n_layers} 条                               
    ║  钻孔: ({self.well.x:.0f}, {self.well.y:.0f}), {self.well.z_top:.0f}~{self.well.z_bottom:.0f}m
    ║  网格: {mesh_line}
    ║  网格QC: {qc_line}
    ║  正演: {'✅ 已计算 (' + str(self.current_result_method) + ')' if has_result else '❌ 未计算'}
    ║  反演: {'✅ 已计算 (' + str(self.inverse_result_method) + ')' if has_inverse else '❌ 未计算'}
    ╚══════════════════════════════════════════════╝
        """)
