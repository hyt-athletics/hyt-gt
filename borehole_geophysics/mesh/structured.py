"""
结构化六面体网格
最简单的3D网格：把长方体空间均匀切成小长方体
"""

import numpy as np


class StructuredMesh:
    """
    结构化六面体(长方体)网格
    
    就像把一大块豆腐沿x、y、z三个方向均匀切割
    
    坐标系约定（地球物理惯例）:
        X: 东方向 (水平)
        Y: 北方向 (水平)  
        Z: 向下为正 (深度)
    """
    
    def __init__(self, x_range, y_range, z_range, nx, ny, nz):
        """
        创建结构化网格
        
        参数:
            x_range: (x最小值, x最大值)，比如 (0, 1000) 表示东西方向1000米
            y_range: (y最小值, y最大值)，比如 (0, 1000) 表示南北方向1000米
            z_range: (z最小值, z最大值)，比如 (0, 1000) 表示深度1000米
            nx: x方向切几刀，比如 20 就是切成20块
            ny: y方向切几刀
            nz: z方向切几刀
        """
        # 保存基本参数
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # 生成每个方向的切割位置
        # 比如 x_range=(0,1000), nx=5 → x_edges=[0, 200, 400, 600, 800, 1000]
        self.x_edges = np.linspace(x_range[0], x_range[1], nx + 1)
        self.y_edges = np.linspace(y_range[0], y_range[1], ny + 1)
        self.z_edges = np.linspace(z_range[0], z_range[1], nz + 1)
        
        # 总共多少个格子
        self.n_cells = nx * ny * nz
        
        # 每个格子的物性值（先全部初始化为0或背景值）
        self.density = np.zeros(self.n_cells)             # 密度差 g/cm³
        self.susceptibility = np.zeros(self.n_cells)       # 磁化率
        self.resistivity = np.ones(self.n_cells) * 100.0   # 电阻率 Ω·m
        self.velocity = np.ones(self.n_cells) * 3000.0     # 波速 m/s
        
        # 预先计算所有格子的中心坐标和尺寸（后面经常要用）
        self._compute_cell_info()
    
    def _compute_cell_info(self):
        """计算每个格子的中心坐标和尺寸"""
        # 每个方向的格子中心位置
        x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        z_centers = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        
        # 每个方向的格子尺寸
        dx = np.diff(self.x_edges)
        dy = np.diff(self.y_edges)
        dz = np.diff(self.z_edges)
        
        # 用 meshgrid 展开成三维数组，再压平成一维
        # 顺序：x变化最快，然后y，然后z（C顺序）
        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        self.cell_centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing='ij')
        self.cell_sizes = np.column_stack([DX.ravel(), DY.ravel(), DZ.ravel()])
        self.cell_volumes = DX.ravel() * DY.ravel() * DZ.ravel()
    
    def cell_index(self, ix, iy, iz):
        """
        把三维索引(ix, iy, iz)转成一维索引
        
        比如一个 5×5×5 的网格:
            第2行第3列第1层的格子 → cell_index(2, 3, 1) → 某个数字
        """
        return ix * (self.ny * self.nz) + iy * self.nz + iz
    
    def assign_property_to_region(self, geometry_func, prop_name, value):
        """
        给某个区域内的所有格子赋物性值
        
        参数:
            geometry_func: 一个函数，输入(x,y,z)坐标，返回True(在区域内)或False(不在)
            prop_name: 物性名称，如 'density'
            value: 物性值，如 0.5 (g/cm³)
        
        举例: 给一个球形矿体赋密度
            def sphere(x, y, z):
                return (x-500)**2 + (y-500)**2 + (z-500)**2 < 100**2
            mesh.assign_property_to_region(sphere, 'density', 0.5)
        """
        prop_array = getattr(self, prop_name)  # 获取对应的物性数组
        
        for i in range(self.n_cells):
            x, y, z = self.cell_centers[i]
            if geometry_func(x, y, z):
                prop_array[i] = value
    
    def get_info(self):
        """打印网格的基本信息"""
        info = f"""
        ╔══════════════════════════════════════╗
        ║        结构化网格信息                  ║
        ╠══════════════════════════════════════╣
        ║ 网格数量: {self.nx} × {self.ny} × {self.nz} = {self.n_cells} 个
        ║ X范围: {self.x_edges[0]:.0f} ~ {self.x_edges[-1]:.0f} m
        ║ Y范围: {self.y_edges[0]:.0f} ~ {self.y_edges[-1]:.0f} m
        ║ Z范围: {self.z_edges[0]:.0f} ~ {self.z_edges[-1]:.0f} m (深度)
        ║ 最小格子: {np.min(self.cell_sizes, axis=0)} m
        ║ 最大格子: {np.max(self.cell_sizes, axis=0)} m
        ╚══════════════════════════════════════╝
        """
        print(info)
        return info