"""
正演结果存储和操作

存储计算结果，提供绘图和导出功能
"""

import numpy as np


class ForwardResult:
    """
    正演结果
    
    存储一次正演计算的所有信息
    """
    
    def __init__(self, well, depths, gz, mesh_data=None):
        """
        参数:
            well: VerticalWell 对象
            depths: 观测点深度数组 (N,)
            gz: 重力异常数组 (N,) mGal
            mesh_data: 使用的网格数据（可选，用于后续分析）
        """
        self.well = well
        self.depths = np.array(depths)
        self.gz = np.array(gz)
        self.mesh_data = mesh_data
    
    @property
    def max_anomaly(self):
        return self.gz.max()
    
    @property
    def min_anomaly(self):
        return self.gz.min()
    
    @property
    def peak_depth(self):
        """最大异常对应的深度"""
        return self.depths[np.argmax(np.abs(self.gz))]
    
    def get_summary(self):
        """结果摘要"""
        summary = f"""
    ┌─ 正演结果摘要 ────────────────────┐
    │ 钻孔位置: ({self.well.x}, {self.well.y})
    │ 深度范围: {self.depths.min():.0f} ~ {self.depths.max():.0f} m
    │ 观测点数: {len(self.depths)}
    │
    │ gz 最大值: {self.max_anomaly:.4f} mGal
    │ gz 最小值: {self.min_anomaly:.4f} mGal
    │ gz 峰值深度: {self.peak_depth:.0f} m
    │ gz 均值: {self.gz.mean():.4f} mGal
    │ gz 标准差: {self.gz.std():.4f} mGal
    └──────────────────────────────────────┘
        """
        print(summary)
        return summary
    
    def plot(self, **kwargs):
        """快捷绘图"""
        from visualization.plot_gravity import plot_borehole_gravity
        plot_borehole_gravity(self, **kwargs)
    
    def save_csv(self, filename):
        """导出为CSV文件"""
        header = "depth_m,gz_mGal"
        data = np.column_stack([self.depths, self.gz])
        np.savetxt(filename, data, delimiter=',', header=header,
                   comments='', fmt='%.4f')
        print(f"  ✅ 结果已保存: {filename}")
    
    def save_npz(self, filename):
        """导出为numpy格式"""
        np.savez(filename,
                 depths=self.depths,
                 gz=self.gz,
                 well_x=self.well.x,
                 well_y=self.well.y)
        print(f"  ✅ 结果已保存: {filename}")