"""
井地电法专用网格适配器

电法网格的核心要求：
  1. 电极位置必须是节点（conform constraint）
  2. 电极处极细网格（1/r²奇异性）
  3. 远边界扩展（模拟无穷远）
  4. 电阻率突变处加密
"""

import numpy as np
from mesh.mesh_adapter import MeshAdapter, RefinementRule
from mesh.boundary_padding import compute_padded_domain, print_padding_info


class ElectricalMeshAdapter(MeshAdapter):
    """
    电法网格适配器
    
    使用方法：
        adapter = ElectricalMeshAdapter(electrode_array)
        rules = adapter.get_refinement_rules()
        domain = adapter.get_domain_with_padding(core_domain)
        # 然后传给八叉树或四面体网格生成器
    """
    
    def __init__(self, electrode_array):
        super().__init__("井地电法 (DC Resistivity)")
        self.electrodes = electrode_array
        
        self.requirements = {
            '电极网格': '电极位置必须是节点',
            '电极处最小格子': '≤ 1m（实际建议0.5m）',
            '电极处过渡': '最少3-5层过渡网格',
            '每十倍距层数': '每十倍距离至少3个格子',
            '电阻率突变': '突变面两侧至少2-3层过渡格子',
            '边界扩展': '≥ 5-10层对数间距扩展',
            '最小质量': '四面体质量 > 0.1',
        }
    
    def get_special_points(self):
        """电极位置 = 必须在节点上的特殊点"""
        return self.electrodes.all_positions
    
    def get_refinement_rules(self):
        """
        生成加密规则
        
        电法核心规则：
          在每个电极周围，格子尺寸 ∝ 距离
          
          距离(m)    格子尺寸(m)
          0.5        0.5         ← 最小
          1          1
          5          5
          10         10
          50         25          ← 开始饱和
          100        50
          500        100         ← 背景
        """
        rules = []
        
        for i, electrode in enumerate(self.electrodes.electrodes):
            if electrode.is_source:
                # 供电电极：最强奇异性，最精细
                rules.append(RefinementRule(
                    point=electrode.position,
                    min_size=1.0,          # 电极处最小1m
                    max_size=100.0,        # 远处最大100m
                    influence_radius=500,
                    priority=10,
                    label=f'source_{electrode.label}'
                ))
            else:
                # 测量电极：也需要精细但稍逊
                rules.append(RefinementRule(
                    point=electrode.position,
                    min_size=2.0,
                    max_size=100.0,
                    influence_radius=200,
                    priority=5,
                    label=f'receiver_{electrode.label}'
                ))
        
        return rules
    
    def get_domain_with_padding(self, core_domain):
        """
        电法需要很大的边界扩展
        
        经验规则：
          边界到最远电极的距离 ≥ 最大电极间距 × 5
        """
        padded, info = compute_padded_domain(
            core_domain,
            n_pad_layers=8,          # 8层扩展
            expansion_factor=1.8     # 每层1.8倍增长
        )
        print_padding_info(core_domain, padded, info)
        return padded, info
    
    def compute_octree_levels(self, core_domain, electrode_positions):
        """
        计算八叉树加密的层级和半径
        
        返回：
            levels_and_radii: [(level, radius), ...] 从最精细到最粗
        """
        # 核心域尺寸
        domain_size = max(
            core_domain['x_range'][1] - core_domain['x_range'][0],
            core_domain['y_range'][1] - core_domain['y_range'][0],
            core_domain['z_range'][1] - core_domain['z_range'][0],
        )
        
        # 确定最大层级（使最小格子 ≈ 1m）
        max_level = int(np.ceil(np.log2(domain_size / 1.0)))
        max_level = min(max_level, 10)  # 安全上限
        
        # 生成加密层级
        levels = []
        for lv in range(max_level, max(max_level - 6, 0), -1):
            cell_size = domain_size / (2 ** lv)
            radius = cell_size * 8   # 影响半径 = 8倍格子尺寸
            levels.append((lv, radius))
        
        return levels
    
    def check_mesh_quality(self, mesh_data):
        """
        检查网格是否满足电法要求
        """
        centers = mesh_data['centers']
        sizes = mesh_data['sizes']
        
        electrode_pos = self.electrodes.all_positions
        
        issues = []
        warnings = []
        
        for i, pos in enumerate(electrode_pos):
            # 找到最近的格子
            dists = np.linalg.norm(centers - pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]
            nearest_size = np.max(sizes[nearest_idx])
            
            e = self.electrodes.electrodes[i]
            
            if nearest_dist > nearest_size:
                issues.append(
                    f"❌ 电极 {e.label}: 不在任何格子内 (距最近格子 {nearest_dist:.1f}m)")
            
            if nearest_size > 5.0 and e.is_source:
                warnings.append(
                    f"⚠️ 电极 {e.label}: 附近格子尺寸 {nearest_size:.1f}m > 5m")
        
        # 打印报告
        print(f"\n  ┌─ 电法网格检查报告 ───────────────────────────┐")
        print(f"  │ 网格单元数: {len(centers)}")
        print(f"  │ 电极数: {len(electrode_pos)}")
        
        if issues:
            for issue in issues:
                print(f"  │ {issue}")
        if warnings:
            for warn in warnings:
                print(f"  │ {warn}")
        
        if not issues and not warnings:
            print(f"  │ ✅ 所有检查通过")
        
        print(f"  └──────────────────────────────────────────────┘")
        
        return len(issues) == 0
    
    def generate_adapted_octree(self, core_domain, well=None, bodies=None):
        """
        一键生成电法适配的八叉树网格
        """
        from mesh.octree import OctreeMesh
        
        print(f"\n{'='*55}")
        print(f"  生成电法适配八叉树网格")
        print(f"{'='*55}")
        
        # 扩展域
        padded_domain, pad_info = self.get_domain_with_padding(core_domain)
        
        # 计算最大层级
        domain_size = max(
            padded_domain['x_range'][1] - padded_domain['x_range'][0],
            padded_domain['y_range'][1] - padded_domain['y_range'][0],
            padded_domain['z_range'][1] - padded_domain['z_range'][0],
        )
        max_level = min(int(np.ceil(np.log2(domain_size / 2.0))), 9)
        
        mesh = OctreeMesh(
            x_range=padded_domain['x_range'],
            y_range=padded_domain['y_range'],
            z_range=padded_domain['z_range'],
            max_level=max_level
        )
        
        print(f"  最大层级: {max_level} (最小格子 ≈ {domain_size/2**max_level:.1f}m)")
        
        # 电极处加密
        rules = self.get_refinement_rules()
        for rule in rules:
            levels_radii = []
            for lv in range(max_level, max(max_level - 5, 2), -1):
                cell_size = domain_size / (2 ** lv)
                if cell_size <= rule.min_size:
                    levels_radii.append((lv, rule.min_size * 10))
                elif cell_size <= rule.max_size:
                    radius = cell_size * 5
                    levels_radii.append((lv, min(radius, rule.influence_radius)))
            
            for lv, r in levels_radii:
                mesh.refine_around_point(tuple(rule.point), r, lv)
        
        print(f"  电极加密后: {mesh.n_cells} 个格子")
        
        # 钻孔加密
        if well is not None:
            mesh.refine_along_borehole(well, levels_and_radii=[
                (max_level - 1, 30),
                (max_level - 2, 80),
                (max_level - 3, 200),
            ])
            print(f"  钻孔加密后: {mesh.n_cells} 个格子")
        
        # 异常体边界加密
        if bodies:
            for body in bodies:
                mesh.refine_at_boundary(body.contains_point_3d,
                                        target_level=max_level - 2)
            print(f"  异常体加密后: {mesh.n_cells} 个格子")
        
        # 平衡
        mesh.balance()
        print(f"  平衡后: {mesh.n_cells} 个格子")
        
        # 赋电阻率
        if bodies:
            for body in bodies:
                mesh.assign_property(body.contains_point_3d,
                                    'resistivity', body.resistivity)
        
        # 检查
        self.check_mesh_quality(mesh.to_arrays())
        
        mesh.get_info()
        return mesh, padded_domain