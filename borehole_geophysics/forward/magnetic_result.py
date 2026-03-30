"""
三分量井中磁测结果对象
"""

from __future__ import annotations

import numpy as np

from borehole.well import VerticalWell
from forward.result import _read_result_csv


class MagneticForwardResult:
    """统一封装三分量磁测正演结果。"""

    def __init__(self, well, depths, bx, by, bz, bt, mesh_data=None, survey=None):
        self.well = well
        self.depths = np.asarray(depths, dtype=float)
        self.bx = np.asarray(bx, dtype=float)
        self.by = np.asarray(by, dtype=float)
        self.bz = np.asarray(bz, dtype=float)
        self.bt = np.asarray(bt, dtype=float)
        self.mesh_data = mesh_data
        self.survey = survey or {}

    @property
    def max_anomaly(self):
        return float(np.max(np.abs(self.bt)))

    @property
    def peak_depth(self):
        return float(self.depths[np.argmax(np.abs(self.bt))])

    def get_summary(self):
        summary = f"""
    ┌─ 磁法结果摘要 ────────────────────┐
    │ 钻孔位置: ({self.well.x}, {self.well.y})
    │ 深度范围: {self.depths.min():.0f} ~ {self.depths.max():.0f} m
    │ 观测点数: {len(self.depths)}
    │
    │ Bx 范围: {self.bx.min():.3f} ~ {self.bx.max():.3f} nT
    │ By 范围: {self.by.min():.3f} ~ {self.by.max():.3f} nT
    │ Bz 范围: {self.bz.min():.3f} ~ {self.bz.max():.3f} nT
    │ ΔT 峰值: {self.max_anomaly:.3f} nT
    │ ΔT 峰值深度: {self.peak_depth:.0f} m
    └──────────────────────────────────────┘
        """
        print(summary)
        return summary

    def plot(self, **kwargs):
        from visualization.plot_magnetic import plot_borehole_magnetic
        plot_borehole_magnetic(self, **kwargs)

    def plot_with_model(self, model_manager=None, **kwargs):
        from visualization.plot_magnetic import plot_magnetic_with_model
        bodies = getattr(model_manager, 'bodies', None) if model_manager is not None else None
        plot_magnetic_with_model(self, bodies=bodies, **kwargs)

    def save_csv(self, filename):
        header = "depth_m,bx_nT,by_nT,bz_nT,bt_nT"
        data = np.column_stack([self.depths, self.bx, self.by, self.bz, self.bt])
        np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%.6f')
        print(f"  ✅ 结果已保存: {filename}")

    def save_npz(self, filename):
        np.savez(
            filename,
            depths=self.depths,
            bx=self.bx,
            by=self.by,
            bz=self.bz,
            bt=self.bt,
            well_x=self.well.x,
            well_y=self.well.y,
        )
        print(f"  ✅ 结果已保存: {filename}")

    @classmethod
    def from_arrays(
        cls, well, depths, bx, by, bz, bt, mesh_data=None, survey=None,
        label=None, is_observed=False
    ):
        """从数组直接构造磁法结果。"""
        result = cls(
            well=well,
            depths=depths,
            bx=bx,
            by=by,
            bz=bz,
            bt=bt,
            mesh_data=mesh_data,
            survey=survey,
        )
        if label is not None:
            setattr(result, 'label', label)
        setattr(result, 'is_observed', bool(is_observed))
        return result

    @classmethod
    def load_csv(cls, filename, well=None, label=None):
        """从 CSV 导入三分量磁法结果。"""
        data = _read_result_csv(filename)
        names = set(data.dtype.names or [])
        required = {'depth_m', 'bx_nT', 'by_nT', 'bz_nT', 'bt_nT'}
        if not required.issubset(names):
            raise ValueError("磁法 CSV 必须包含 depth_m,bx_nT,by_nT,bz_nT,bt_nT 列")

        depths = np.asarray(data['depth_m'], dtype=float)
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

        final_label = label or f"observed_magnetic:{filename.split('/')[-1]}"
        return cls.from_arrays(
            well=well,
            depths=depths,
            bx=np.asarray(data['bx_nT'], dtype=float),
            by=np.asarray(data['by_nT'], dtype=float),
            bz=np.asarray(data['bz_nT'], dtype=float),
            bt=np.asarray(data['bt_nT'], dtype=float),
            mesh_data=None,
            survey=None,
            label=final_label,
            is_observed=True,
        )
