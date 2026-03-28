"""
项目管理器

统一管理：模型数据 + 钻孔 + 网格 + 正演结果

一个"项目"包含做一次建模所需的所有东西
"""

import os
import json
import numpy as np

from app.settings import Settings
from interactive.body_manager import ModelManager, GeologicalBody
from borehole.well import VerticalWell


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
        
        # 正演结果
        self.gravity_result = None
        self.result_history = []
    
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
    
    # ==========================================================
    #  网格生成
    # ==========================================================
    
    def generate_mesh(self):
        """根据当前设置生成网格"""
        from mesh.octree import OctreeMesh
        from interactive.layer_tool import layers_to_bodies
        
        print("\n  生成网格...")
        
        domain = self.model_manager.domain
        mesh_settings = self.settings.get('mesh')
        max_level = mesh_settings['max_level']
        
        self.mesh = OctreeMesh(
            x_range=domain['x_range'],
            y_range=domain['y_range'],
            z_range=domain['z_range'],
            max_level=max_level
        )
        
        # 钻孔加密
        refine_config = mesh_settings['borehole_refine_levels']
        levels_radii = [(int(lr[0]), float(lr[1])) for lr in refine_config]
        self.mesh.refine_along_borehole(self.well, levels_and_radii=levels_radii)
        
        # 合并所有地质体（包括地层线生成的）
        all_bodies = list(self.model_manager.bodies)
        if self.model_manager.layers:
            layer_bodies = layers_to_bodies(
                self.model_manager.layers,
                domain['x_range'], domain['z_range']
            )
            all_bodies.extend(layer_bodies)
        
        # 异常体边界加密
        bnd_level = mesh_settings['boundary_refine_level']
        for body in all_bodies:
            self.mesh.refine_at_boundary(body.contains_point_3d, target_level=bnd_level)
        
        self.mesh.balance()
        
        # 赋物性
        for body in all_bodies:
            self.mesh.assign_property(body.contains_point_3d, 'density', body.density)
            if hasattr(body, 'resistivity'):
                self.mesh.assign_property(body.contains_point_3d, 'resistivity', body.resistivity)
            if hasattr(body, 'velocity') and body.velocity != 3000:
                self.mesh.assign_property(body.contains_point_3d, 'velocity', body.velocity)
        
        self.mesh_data = self.mesh.to_arrays()
        self.mesh.get_info()
        
        return self.mesh
    
    # ==========================================================
    #  正演计算
    # ==========================================================
    
    def compute_gravity(self):
        """计算重力正演"""
        from forward.borehole_gravity import BoreholeGravityCalculator
        
        if self.mesh_data is None:
            self.generate_mesh()
        
        engine = self.settings.get('forward.engine', 'auto')
        calc = BoreholeGravityCalculator(engine=engine)
        
        result = calc.compute(self.well, self.mesh_data)
        
        if self.gravity_result is not None:
            self.result_history.append(self.gravity_result)
        self.gravity_result = result
        
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
        if self.gravity_result is not None:
            result_file = os.path.join(directory, 'gravity_result.csv')
            self.gravity_result.save_csv(result_file)
        
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
        has_result = self.gravity_result is not None
        
        print(f"""
    ╔══════════════════════════════════════════════╗
    ║  项目: {self.name:38s} ║
    ╠══════════════════════════════════════════════╣
    ║  地质体: {n_bodies} 个                               
    ║  地层线: {n_layers} 条                               
    ║  钻孔: ({self.well.x:.0f}, {self.well.y:.0f}), {self.well.z_top:.0f}~{self.well.z_bottom:.0f}m
    ║  网格: {'✅ 已生成 (' + str(self.mesh_data["n_cells"]) + ' 格子)' if has_mesh else '❌ 未生成'}
    ║  正演: {'✅ 已计算' if has_result else '❌ 未计算'}
    ╚══════════════════════════════════════════════╝
        """)