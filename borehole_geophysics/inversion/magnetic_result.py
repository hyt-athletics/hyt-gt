"""
磁法反演结果对象
"""

from __future__ import annotations

import json
import numpy as np

from borehole.well import VerticalWell


class MagneticInversionResult:
    """封装基于外部库的 3C 井中磁测反演结果。"""

    def __init__(
        self,
        well,
        depths,
        observed_bx,
        observed_by,
        observed_bz,
        observed_bt,
        predicted_bx,
        predicted_by,
        predicted_bz,
        predicted_bt,
        recovered_mesh_data,
        survey=None,
        metadata=None,
    ):
        self.well = well
        self.depths = np.asarray(depths, dtype=float)
        self.observed_bx = np.asarray(observed_bx, dtype=float)
        self.observed_by = np.asarray(observed_by, dtype=float)
        self.observed_bz = np.asarray(observed_bz, dtype=float)
        self.observed_bt = np.asarray(observed_bt, dtype=float)
        self.predicted_bx = np.asarray(predicted_bx, dtype=float)
        self.predicted_by = np.asarray(predicted_by, dtype=float)
        self.predicted_bz = np.asarray(predicted_bz, dtype=float)
        self.predicted_bt = np.asarray(predicted_bt, dtype=float)
        self.recovered_mesh_data = recovered_mesh_data
        self.survey = survey or {}
        self.metadata = metadata or {}

    @property
    def recovered_susceptibility(self):
        return self.recovered_mesh_data.get('susceptibility')

    @property
    def residual_bt(self):
        return self.predicted_bt - self.observed_bt

    @property
    def rmse_bt(self):
        return float(np.sqrt(np.mean(self.residual_bt ** 2)))

    def get_summary(self):
        susceptibility = np.asarray(self.recovered_susceptibility, dtype=float)
        update_mode = self.metadata.get('update_mode', 'all_active')
        n_update = self.metadata.get('n_update_cells', len(susceptibility))
        n_fixed = self.metadata.get('n_fixed_cells', 0)
        summary = f"""
    ┌─ 磁法反演结果摘要 ────────────────────┐
    │ 钻孔位置: ({self.well.x}, {self.well.y})
    │ 深度范围: {self.depths.min():.0f} ~ {self.depths.max():.0f} m
    │ 观测点数: {len(self.depths)}
    │
    │ 观测 ΔT 范围: {self.observed_bt.min():.3f} ~ {self.observed_bt.max():.3f} nT
    │ 预测 ΔT 范围: {self.predicted_bt.min():.3f} ~ {self.predicted_bt.max():.3f} nT
    │ ΔT RMSE: {self.rmse_bt:.3f} nT
    │ recovered χ: {susceptibility.min():.5f} ~ {susceptibility.max():.5f} SI
    │ 更新模式: {update_mode}
    │ 更新/固定单元: {n_update} / {n_fixed}
    │ 约束范围: {self.metadata.get('lower_bound', 'n/a')} ~ {self.metadata.get('upper_bound', 'n/a')} SI
    │ 正则: {self.metadata.get('regularization_mode', 'smooth')} / {self.metadata.get('reference_model_mode', 'property')}
    │ 单元数: {len(susceptibility)}
    │ 假设: {self.metadata.get('assumptions', 'n/a')}
    └──────────────────────────────────────────────┘
        """
        print(summary)
        return summary

    def save_csv(self, filename):
        header = "depth_m,observed_bx_nT,observed_by_nT,observed_bz_nT,observed_bt_nT,predicted_bx_nT,predicted_by_nT,predicted_bz_nT,predicted_bt_nT,residual_bt_nT"
        data = np.column_stack([
            self.depths,
            self.observed_bx,
            self.observed_by,
            self.observed_bz,
            self.observed_bt,
            self.predicted_bx,
            self.predicted_by,
            self.predicted_bz,
            self.predicted_bt,
            self.residual_bt,
        ])
        np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%.6f')
        print(f"  ✅ 磁法反演结果已保存: {filename}")

    def save_npz(self, filename):
        np.savez(
            filename,
            depths=self.depths,
            observed_bx=self.observed_bx,
            observed_by=self.observed_by,
            observed_bz=self.observed_bz,
            observed_bt=self.observed_bt,
            predicted_bx=self.predicted_bx,
            predicted_by=self.predicted_by,
            predicted_bz=self.predicted_bz,
            predicted_bt=self.predicted_bt,
            centers=self.recovered_mesh_data['centers'],
            sizes=self.recovered_mesh_data['sizes'],
            susceptibility=self.recovered_mesh_data['susceptibility'],
            update_mask=self.recovered_mesh_data.get('update_mask'),
            fixed_mask=self.recovered_mesh_data.get('fixed_mask'),
            well_x=self.well.x,
            well_y=self.well.y,
            metadata_json=json.dumps(self.metadata, ensure_ascii=False),
        )
        print(f"  ✅ 磁法反演结果已保存: {filename}")

    def plot(self, **kwargs):
        from visualization.plot_magnetic import plot_magnetic_inversion_result
        plot_magnetic_inversion_result(self, **kwargs)

    def plot_slice(self, **kwargs):
        from visualization.plot_magnetic import plot_magnetic_inverse_susceptibility_slice
        plot_magnetic_inverse_susceptibility_slice(self, **kwargs)

    @classmethod
    def load_npz(cls, filename, well=None, label=None):
        data = np.load(filename)
        depths = np.asarray(data['depths'], dtype=float)
        if well is None:
            well = VerticalWell(
                x=float(data['well_x']),
                y=float(data['well_y']),
                z_top=float(np.min(depths)),
                z_bottom=float(np.max(depths)),
                n_stations=len(depths),
            )
        result = cls(
            well=well,
            depths=depths,
            observed_bx=np.asarray(data['observed_bx'], dtype=float),
            observed_by=np.asarray(data['observed_by'], dtype=float),
            observed_bz=np.asarray(data['observed_bz'], dtype=float),
            observed_bt=np.asarray(data['observed_bt'], dtype=float),
            predicted_bx=np.asarray(data['predicted_bx'], dtype=float),
            predicted_by=np.asarray(data['predicted_by'], dtype=float),
            predicted_bz=np.asarray(data['predicted_bz'], dtype=float),
            predicted_bt=np.asarray(data['predicted_bt'], dtype=float),
            recovered_mesh_data={
                'centers': np.asarray(data['centers'], dtype=float),
                'sizes': np.asarray(data['sizes'], dtype=float),
                'susceptibility': np.asarray(data['susceptibility'], dtype=float),
                'update_mask': np.asarray(data['update_mask'], dtype=bool) if 'update_mask' in data else None,
                'fixed_mask': np.asarray(data['fixed_mask'], dtype=bool) if 'fixed_mask' in data else None,
                'n_cells': len(np.asarray(data['susceptibility'])),
            },
            metadata={
                **(
                    json.loads(str(data['metadata_json']))
                    if 'metadata_json' in data and str(data['metadata_json']).strip()
                    else {}
                ),
                'loaded_from': filename,
            },
        )
        if label is not None:
            setattr(result, 'label', label)
        return result
