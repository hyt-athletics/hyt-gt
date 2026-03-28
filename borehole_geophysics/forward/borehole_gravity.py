"""
钻孔重力计算器

把 "网格数据" + "钻孔" 传进来
自动完成：
  1. 获取钻孔观测点坐标
  2. 获取网格数据（中心、尺寸、密度）
  3. 调用正演引擎
  4. 返回结果
"""

import numpy as np
from forward.gravity_engine import forward_gravity
from forward.result import ForwardResult


class BoreholeGravityCalculator:
    """
    钻孔重力正演计算器
    
    使用方法:
        calc = BoreholeGravityCalculator()
        result = calc.compute(well, mesh_data)
        result.plot()
    """
    
    def __init__(self, engine='auto'):
        """
        参数:
            engine: 'auto', 'builtin', 'harmonica'
        """
        self.engine = engine
    
    def compute(self, well, mesh_data, component='gz'):
        """
        计算钻孔重力异常
        
        参数:
            well: VerticalWell 对象
            mesh_data: 网格数据字典，需要包含:
                       'centers': (N, 3) 棱柱中心
                       'sizes': (N, 3) 棱柱尺寸
                       'density': (N,) 密度差
            component: 'gz' (垂直分量，默认)
        
        返回:
            ForwardResult 对象
        """
        print("\n" + "=" * 50)
        print("  钻孔重力正演计算")
        print("=" * 50)
        
        # 获取观测点
        obs_points = well.stations.copy()
        print(f"  钻孔: x={well.x}, y={well.y}")
        print(f"  深度: {well.z_top} ~ {well.z_bottom} m")
        print(f"  观测点: {len(obs_points)} 个")
        
        # 获取网格数据
        centers = mesh_data['centers']
        sizes = mesh_data['sizes']
        densities = mesh_data['density']
        
        # 计算
        gz = forward_gravity(
            obs_points, centers, sizes, densities,
            engine=self.engine
        )
        
        # 封装结果
        result = ForwardResult(
            well=well,
            depths=obs_points[:, 2],
            gz=gz,
            mesh_data=mesh_data
        )
        
        return result
    
    def compute_multiple_wells(self, wells, mesh_data):
        """
        同时计算多口钻孔的重力异常
        
        参数:
            wells: VerticalWell 对象列表
            mesh_data: 网格数据
        
        返回:
            results: ForwardResult 列表
        """
        results = []
        for i, well in enumerate(wells):
            print(f"\n--- 钻孔 {i+1}/{len(wells)} ---")
            result = self.compute(well, mesh_data)
            results.append(result)
        return results
    
    def compute_sensitivity(self, well, mesh_data, cell_index):
        """
        计算单个格子对钻孔重力的贡献（灵敏度）
        
        参数:
            cell_index: 要分析的格子索引
        
        返回:
            gz_contribution: 该格子在每个观测点的贡献 (N_obs,)
        """
        obs_points = well.stations.copy()
        
        # 只取一个格子
        center = mesh_data['centers'][cell_index:cell_index+1]
        size = mesh_data['sizes'][cell_index:cell_index+1]
        density = np.array([1.0])  # 单位密度，看纯几何贡献
        
        gz = forward_gravity(
            obs_points, center, size, density,
            engine=self.engine
        )
        
        return gz