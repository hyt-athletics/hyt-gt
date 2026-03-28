"""
VSP地震专用网格适配器

地震网格的核心要求：
  1. 每个波长至少 5-10 个格子（防止数值频散）
  2. 低速区格子更小
  3. 震源/检波器位置在节点上
  4. 吸收边界层
  5. CFL条件约束时间步
  
关键公式:
  最小波长 λ_min = V_min / f_max
  最大格子 h_max = λ_min / N_ppw    (N_ppw = 每波长点数, 通常5-10)
  最大时间步 dt_max = C * h_min / V_max   (C = CFL数, 通常0.5-0.8)
"""

import numpy as np
from mesh.mesh_adapter import MeshAdapter, RefinementRule
from mesh.boundary_padding import compute_padded_domain, print_padding_info


class SeismicMeshAdapter(MeshAdapter):
    """
    地震网格适配器
    """
    
    def __init__(self, vsp_survey, velocity_model=None):
        """
        参数:
            vsp_survey: VSPSurvey 对象
            velocity_model: 速度模型字典
                           {'v_min': 最小速度, 'v_max': 最大速度,
                            'layers': [(z_top, z_bot, velocity), ...]}
        """
        super().__init__("VSP地震")
        self.survey = vsp_survey
        
        if velocity_model is None:
            velocity_model = {'v_min': 2000, 'v_max': 5000, 'layers': []}
        self.velocity_model = velocity_model
        
        self.points_per_wavelength = 8  # 每波长格子数（保守值）
        self.cfl_number = 0.5            # CFL安全系数
        self.n_pml_layers = 10           # 吸收层(PML)层数
        
        # 计算关键参数
        self._compute_mesh_parameters()
        
        self.requirements = {
            '最高频率': f'{self.survey.max_frequency:.0f} Hz',
            '最小速度': f'{self.velocity_model["v_min"]:.0f} m/s',
            '最小波长': f'{self.min_wavelength:.1f} m',
            '最大格子': f'{self.max_cell_size:.1f} m',
            '每波长点数': f'{self.points_per_wavelength}',
            'CFL时间步': f'{self.max_dt*1000:.3f} ms',
            '吸收层': f'{self.n_pml_layers} 层',
        }
    
    def _compute_mesh_parameters(self):
        """计算网格参数"""
        v_min = self.velocity_model['v_min']
        v_max = self.velocity_model['v_max']
        f_max = self.survey.max_frequency
        
        # 最小波长
        self.min_wavelength = v_min / f_max
        
        # 最大允许格子尺寸
        self.max_cell_size = self.min_wavelength / self.points_per_wavelength
        
        # CFL条件下的最大时间步
        self.max_dt = self.cfl_number * self.max_cell_size / v_max
        
        print(f"""
    ┌─ 地震网格参数计算 ───────────────────────────┐
    │ 最高频率 f_max = {f_max:.0f} Hz
    │ 最小速度 V_min = {v_min:.0f} m/s
    │ 最大速度 V_max = {v_max:.0f} m/s
    │
    │ 最小波长 λ_min = V_min/f_max = {self.min_wavelength:.1f} m
    │ 最大格子 h_max = λ_min/{self.points_per_wavelength} = {self.max_cell_size:.1f} m
    │ 最大时间步 dt = {self.max_dt*1000:.3f} ms
    │
    │ 说明：格子尺寸不能超过 {self.max_cell_size:.1f}m
    │       否则波形会失真（数值频散）
    └──────────────────────────────────────────────┘
        """)
    
    def cell_size_for_velocity(self, velocity):
        """
        根据局部速度计算所需的格子尺寸
        
        低速区 → 波长短 → 格子要小
        高速区 → 波长长 → 格子可以大
        
        参数:
            velocity: 当地速度 (m/s)
        
        返回:
            max_size: 最大允许格子尺寸 (m)
        """
        f_max = self.survey.max_frequency
        local_wavelength = velocity / f_max
        return local_wavelength / self.points_per_wavelength
    
    def get_special_points(self):
        """震源和检波器位置"""
        return self.survey.all_special_points
    
    def get_refinement_rules(self):
        """
        地震网格加密规则
        
        震源处需要最精细（脉冲源的空间解析）
        检波器处也需要精细（记录精度）
        """
        rules = []
        
        # 震源加密
        for src in self.survey.sources:
            min_size = self.max_cell_size / 2  # 震源处用全局最小的一半
            rules.append(RefinementRule(
                point=src.position,
                min_size=min_size,
                max_size=self.max_cell_size,
                influence_radius=self.min_wavelength * 3,
                priority=10,
                label=f'source_{src.label}'
            ))
        
        # 检波器加密
        for rec in self.survey.receivers:
            rules.append(RefinementRule(
                point=rec.position,
                min_size=self.max_cell_size * 0.8,
                max_size=self.max_cell_size,
                influence_radius=self.min_wavelength * 2,
                priority=5,
                label=f'receiver_{rec.label}'
            ))
        
        return rules
    
    def get_domain_with_padding(self, core_domain):
        """
        地震需要吸收边界
        
        PML层厚度 ≈ 2~3个最大波长
        """
        max_wavelength = self.velocity_model['v_max'] / self.survey.max_frequency
        pml_thickness = max_wavelength * 3
        
        padded = {}
        for axis in ['x_range', 'y_range', 'z_range']:
            c_min, c_max = core_domain[axis]
            padded[axis] = (c_min - pml_thickness, c_max + pml_thickness)
        
        # Z方向地表以上不需要太多
        padded['z_range'] = (
            max(padded['z_range'][0], -pml_thickness),
            padded['z_range'][1]
        )
        
        print(f"\n  PML吸收层厚度: {pml_thickness:.0f}m (≈3个最大波长)")
        print(f"  核心域: X={core_domain['x_range']}, "
              f"Y={core_domain['y_range']}, Z={core_domain['z_range']}")
        print(f"  扩展域: X={padded['x_range']}, "
              f"Y={padded['y_range']}, Z={padded['z_range']}")
        
        return padded, {'pml_thickness': pml_thickness}
    
    def check_mesh_quality(self, mesh_data, velocity_field=None):
        """
        检查网格是否满足地震波传播要求
        """
        centers = mesh_data['centers']
        sizes = mesh_data['sizes']
        
        if velocity_field is None:
            velocity_field = mesh_data.get('velocity',
                                           np.ones(len(centers)) * self.velocity_model['v_min'])
        
        issues = []
        warnings = []
        
        # 检查每个格子是否满足每波长点数要求
        max_sizes = np.max(sizes, axis=1)
        f_max = self.survey.max_frequency
        
        for i in range(len(centers)):
            local_v = velocity_field[i]
            local_wavelength = local_v / f_max
            required_size = local_wavelength / self.points_per_wavelength
            
            if max_sizes[i] > required_size * 1.5:  # 允许50%容差
                issues.append(i)
        
        n_violations = len(issues)
        pct = n_violations / len(centers) * 100
        
        # CFL检查
        min_size = np.min(np.min(sizes, axis=1))
        v_max = self.velocity_model['v_max']
        actual_dt = self.cfl_number * min_size / v_max
        
        print(f"""
    ┌─ 地震网格检查报告 ──────────────────────────────┐
    │ 网格单元数: {len(centers)}
    │ 格子尺寸: {np.min(max_sizes):.1f} ~ {np.max(max_sizes):.1f} m
    │
    │ 波长检查（每波长{self.points_per_wavelength}个点）:
    │   违规格子: {n_violations} / {len(centers)} ({pct:.1f}%)
    │   {'❌ 有格子太大！可能产生数值频散' if n_violations > 0 else '✅ 所有格子满足要求'}
    │
    │ CFL条件:
    │   最小格子: {min_size:.2f} m
    │   最大速度: {v_max:.0f} m/s
    │   建议时间步: {actual_dt*1000:.4f} ms
    │   {'✅ 时间步可行' if actual_dt > 1e-6 else '⚠️ 时间步极小，计算会很慢'}
    └──────────────────────────────────────────────────┘
        """)
        
        return n_violations == 0
    
    def generate_adapted_octree(self, core_domain, well=None, bodies=None):
        """
        一键生成地震适配的八叉树网格
        """
        from mesh.octree import OctreeMesh
        
        print(f"\n{'='*55}")
        print(f"  生成VSP地震适配八叉树网格")
        print(f"{'='*55}")
        
        # 扩展域
        padded_domain, pad_info = self.get_domain_with_padding(core_domain)
        
        # 确定八叉树层级
        domain_size = max(
            padded_domain['x_range'][1] - padded_domain['x_range'][0],
            padded_domain['y_range'][1] - padded_domain['y_range'][0],
            padded_domain['z_range'][1] - padded_domain['z_range'][0],
        )
        
        max_level = int(np.ceil(np.log2(domain_size / self.max_cell_size)))
        max_level = min(max_level, 8)  # 安全上限
        
        mesh = OctreeMesh(
            x_range=padded_domain['x_range'],
            y_range=padded_domain['y_range'],
            z_range=padded_domain['z_range'],
            max_level=max_level
        )
        
        actual_min_size = domain_size / (2 ** max_level)
        print(f"  最大层级: {max_level}")
        print(f"  最小格子: {actual_min_size:.1f}m (要求≤{self.max_cell_size:.1f}m)")
        
        # 全域加密到满足波长要求的层级
        # 找到满足max_cell_size的层级
        wave_level = int(np.ceil(np.log2(domain_size / self.max_cell_size)))
        wave_level = min(wave_level, max_level)
        
        # 在核心域内全部加密到wave_level
        core_center = (
            np.mean(core_domain['x_range']),
            np.mean(core_domain['y_range']),
            np.mean(core_domain['z_range']),
        )
        core_radius = domain_size  # 覆盖整个域
        mesh.refine_around_point(core_center, core_radius, wave_level)
        print(f"  波长加密到层级{wave_level}后: {mesh.n_cells} 个格子")
        
        # 震源/检波器处额外加密
        rules = self.get_refinement_rules()
        for rule in rules:
            mesh.refine_around_point(tuple(rule.point),
                                    rule.influence_radius,
                                    max_level)
        print(f"  震源/检波器加密后: {mesh.n_cells} 个格子")
        
        # 钻孔加密
        if well is not None:
            mesh.refine_along_borehole(well, levels_and_radii=[
                (max_level, 20),
                (max_level - 1, 60),
            ])
            print(f"  钻孔加密后: {mesh.n_cells} 个格子")
        
        # 异常体边界
        if bodies:
            for body in bodies:
                mesh.refine_at_boundary(body.contains_point_3d,
                                        target_level=wave_level)
        
        mesh.balance()
        print(f"  平衡后: {mesh.n_cells} 个格子")
        
        # 赋速度
        if bodies:
            for body in bodies:
                mesh.assign_property(body.contains_point_3d,
                                    'velocity', body.velocity)
        
        # 赋层状速度模型
        if self.velocity_model.get('layers'):
            for z_top, z_bot, vel in self.velocity_model['layers']:
                def layer_func(x, y, z, zt=z_top, zb=z_bot):
                    return zt <= z <= zb
                mesh.assign_property(layer_func, 'velocity', vel)
        
        # 检查
        self.check_mesh_quality(mesh.to_arrays())
        
        mesh.get_info()
        return mesh, padded_domain