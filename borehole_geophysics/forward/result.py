"""
正演结果存储和操作

存储计算结果，提供绘图和导出功能
"""

from __future__ import annotations

import numpy as np

from borehole.well import VerticalWell


def _read_result_csv(filename):
    """读取结果 CSV，兼容带 # 注释头的导出格式。"""
    data = np.genfromtxt(filename, delimiter=',', names=True, comments='#', dtype=float)
    if data.size == 0:
        raise ValueError(f"CSV 文件为空: {filename}")
    if getattr(data, 'ndim', 1) == 0:
        data = np.array([data], dtype=data.dtype)
    return data


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

    @classmethod
    def from_arrays(cls, well, depths, gz, mesh_data=None, label=None, is_observed=False):
        """从数组直接构造重力结果。"""
        result = cls(well=well, depths=depths, gz=gz, mesh_data=mesh_data)
        if label is not None:
            setattr(result, 'label', label)
        setattr(result, 'is_observed', bool(is_observed))
        return result

    @classmethod
    def load_csv(cls, filename, well=None, label=None):
        """从 CSV 导入重力结果。"""
        data = _read_result_csv(filename)
        names = set(data.dtype.names or [])
        if 'depth_m' not in names or 'gz_mGal' not in names:
            raise ValueError("重力 CSV 必须包含 depth_m 和 gz_mGal 列")

        depths = np.asarray(data['depth_m'], dtype=float)
        gz = np.asarray(data['gz_mGal'], dtype=float)

        if well is None:
            x = float(np.asarray(data['x_m'])[0]) if 'x_m' in names else 0.0
            y = float(np.asarray(data['y_m'])[0]) if 'y_m' in names else 0.0
            well = VerticalWell(
                x=x,
                y=y,
                z_top=float(np.min(depths)),
                z_bottom=float(np.max(depths)),
                n_stations=len(depths),
            )

        final_label = label or f"observed_gravity:{filename.split('/')[-1]}"
        return cls.from_arrays(
            well=well,
            depths=depths,
            gz=gz,
            mesh_data=None,
            label=final_label,
            is_observed=True,
        )
