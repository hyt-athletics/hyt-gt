"""
钻孔（直井）定义
"""

import numpy as np


class VerticalWell:
    """
    直井定义
    
    直井就是一根垂直插入地下的管子
    管子上均匀分布着观测点（传感器位置）
    """
    
    def __init__(self, x, y, z_top, z_bottom, n_stations):
        """
        参数:
            x: 井口的X坐标 (m)
            y: 井口的Y坐标 (m)
            z_top: 井口深度 (m)，通常是0或很小的值
            z_bottom: 井底深度 (m)
            n_stations: 观测点个数（沿井均匀分布）
        """
        self.x = x
        self.y = y
        self.z_top = z_top
        self.z_bottom = z_bottom
        self.n_stations = n_stations
        
        # 生成观测点坐标
        z_stations = np.linspace(z_top, z_bottom, n_stations)
        self.stations = np.column_stack([
            np.full(n_stations, x),
            np.full(n_stations, y),
            z_stations
        ])
    
    def get_path_points(self, n_points=100):
        """获取钻孔路径上的密集点（用于网格加密判断）"""
        z_path = np.linspace(self.z_top, self.z_bottom, n_points)
        return np.column_stack([
            np.full(n_points, self.x),
            np.full(n_points, self.y),
            z_path
        ])
    
    def get_info(self):
        info = f"""
        ╔══════════════════════════════════════╗
        ║         直井信息                      ║
        ╠══════════════════════════════════════╣
        ║ 井口位置: ({self.x:.0f}, {self.y:.0f}) m
        ║ 深度范围: {self.z_top:.0f} ~ {self.z_bottom:.0f} m
        ║ 观测点数: {self.n_stations} 个
        ║ 观测点间距: {(self.z_bottom-self.z_top)/(self.n_stations-1):.1f} m
        ╚══════════════════════════════════════╝
        """
        print(info)