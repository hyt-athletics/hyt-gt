"""
重力反演结果对象
"""

from __future__ import annotations

import numpy as np
import json

from borehole.well import VerticalWell


class GravityInversionResult:
    """封装基于外部库的重力反演结果。"""

    def __init__(
        self,
        well,
        depths,
        observed_gz,
        predicted_gz,
        recovered_mesh_data,
        metadata=None,
    ):
        self.well = well
        self.depths = np.asarray(depths, dtype=float)
        self.observed_gz = np.asarray(observed_gz, dtype=float)
        self.predicted_gz = np.asarray(predicted_gz, dtype=float)
        self.recovered_mesh_data = recovered_mesh_data
        self.metadata = metadata or {}

    @property
    def residual(self):
        return self.predicted_gz - self.observed_gz

    @property
    def rmse(self):
        return float(np.sqrt(np.mean(self.residual ** 2)))

    @property
    def recovered_density(self):
        return self.recovered_mesh_data.get('density')

    def get_summary(self):
        density = np.asarray(self.recovered_density, dtype=float)
        update_mode = self.metadata.get('update_mode', 'all_active')
        n_update = self.metadata.get('n_update_cells', len(density))
        n_fixed = self.metadata.get('n_fixed_cells', 0)
        summary = f"""
    ┌─ 重力反演结果摘要 ────────────────────┐
    │ 钻孔位置: ({self.well.x}, {self.well.y})
    │ 深度范围: {self.depths.min():.0f} ~ {self.depths.max():.0f} m
    │ 观测点数: {len(self.depths)}
    │
    │ 观测 gz 范围: {self.observed_gz.min():.4f} ~ {self.observed_gz.max():.4f} mGal
    │ 预测 gz 范围: {self.predicted_gz.min():.4f} ~ {self.predicted_gz.max():.4f} mGal
    │ RMSE: {self.rmse:.4f} mGal
    │ recovered density: {density.min():.4f} ~ {density.max():.4f} g/cc
    │ 更新模式: {update_mode}
    │ 更新/固定单元: {n_update} / {n_fixed}
    │ 约束范围: {self.metadata.get('lower_bound', 'n/a')} ~ {self.metadata.get('upper_bound', 'n/a')} g/cc
    │ 正则: {self.metadata.get('regularization_mode', 'smooth')} / {self.metadata.get('reference_model_mode', 'property')}
    │ 单元数: {len(density)}
    └──────────────────────────────────────────┘
        """
        print(summary)
        return summary

    def save_csv(self, filename):
        header = "depth_m,observed_gz_mGal,predicted_gz_mGal,residual_mGal"
        data = np.column_stack([
            self.depths,
            self.observed_gz,
            self.predicted_gz,
            self.residual,
        ])
        np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%.6f')
        print(f"  ✅ 反演结果已保存: {filename}")

    def save_npz(self, filename):
        np.savez(
            filename,
            depths=self.depths,
            observed_gz=self.observed_gz,
            predicted_gz=self.predicted_gz,
            residual=self.residual,
            centers=self.recovered_mesh_data['centers'],
            sizes=self.recovered_mesh_data['sizes'],
            density=self.recovered_mesh_data['density'],
            update_mask=self.recovered_mesh_data.get('update_mask'),
            fixed_mask=self.recovered_mesh_data.get('fixed_mask'),
            well_x=self.well.x,
            well_y=self.well.y,
            metadata_json=json.dumps(self.metadata, ensure_ascii=False),
        )
        print(f"  ✅ 反演结果已保存: {filename}")

    def plot(self, **kwargs):
        from visualization.plot_gravity import plot_gravity_inversion_result
        plot_gravity_inversion_result(self, **kwargs)

    def plot_slice(self, **kwargs):
        from visualization.plot_gravity import plot_gravity_inverse_density_slice
        plot_gravity_inverse_density_slice(self, **kwargs)

    @classmethod
    def load_npz(cls, filename, well=None, label=None):
        """从 NPZ 反演结果文件恢复对象。"""
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
            observed_gz=np.asarray(data['observed_gz'], dtype=float),
            predicted_gz=np.asarray(data['predicted_gz'], dtype=float),
            recovered_mesh_data={
                'centers': np.asarray(data['centers'], dtype=float),
                'sizes': np.asarray(data['sizes'], dtype=float),
                'density': np.asarray(data['density'], dtype=float),
                'update_mask': np.asarray(data['update_mask'], dtype=bool) if 'update_mask' in data else None,
                'fixed_mask': np.asarray(data['fixed_mask'], dtype=bool) if 'fixed_mask' in data else None,
                'n_cells': len(np.asarray(data['density'])),
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
