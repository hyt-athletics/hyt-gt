"""三维井间电磁波正演

输入/输出规范见 manifest.json。
"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm
from core.mesh_utils import make_tensor_mesh_3d


class EmForward3d(BaseAlgorithm):
    """三维井间电磁波正演（频率域）。"""

    def run(
        self,
        conductivity: np.ndarray,
        tx_locations: np.ndarray,
        rx_locations: np.ndarray,
        frequency: float = 1e3,
        dx: float = 10.0,
        dy: float = 10.0,
        dz: float = 5.0,
    ) -> AlgorithmResult:
        conductivity = np.asarray(conductivity, dtype=float)
        tx_locations = np.asarray(tx_locations, dtype=float)
        rx_locations = np.asarray(rx_locations, dtype=float)

        if conductivity.ndim != 3:
            raise ValueError("conductivity 必须为三维数组")
        if tx_locations.ndim != 2 or tx_locations.shape[1] < 3:
            raise ValueError("tx_locations 必须为 shape (n, >=3) 的二维数组")
        if rx_locations.ndim != 2 or rx_locations.shape[1] < 3:
            raise ValueError("rx_locations 必须为 shape (n, >=3) 的二维数组")

        # ── 构建三维张量网格 ──────────────────────────────────────
        all_pts = np.vstack([tx_locations[:, :3], rx_locations[:, :3]])
        x_extent = float(all_pts[:, 0].max() - all_pts[:, 0].min()) + 4 * dx
        y_extent = float(all_pts[:, 1].max() - all_pts[:, 1].min()) + 4 * dy
        z_extent = float(np.abs(all_pts[:, 2]).max()) + 4 * dz

        mesh = make_tensor_mesh_3d(
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
            dx=dx, dy=dy, dz=dz,
        )

        sigma = np.nan_to_num(conductivity, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.clip(sigma, 1e-6, None)

        tx_xyz = tx_locations[:, :3]
        rx_xyz = rx_locations[:, :3]
        tx_strength = (
            np.abs(tx_locations[:, 3])
            if tx_locations.shape[1] > 3
            else np.ones(tx_xyz.shape[0], dtype=float)
        )

        bounds_min = np.array(
            [
                float(all_pts[:, 0].min()) - 2.0 * dx,
                float(all_pts[:, 1].min()) - 2.0 * dy,
                float(all_pts[:, 2].min()) - 2.0 * dz,
            ],
            dtype=float,
        )
        bounds_max = np.array(
            [
                float(all_pts[:, 0].max()) + 2.0 * dx,
                float(all_pts[:, 1].max()) + 2.0 * dy,
                float(all_pts[:, 2].max()) + 2.0 * dz,
            ],
            dtype=float,
        )
        bounds_span = np.maximum(bounds_max - bounds_min, 1e-9)

        diff = rx_xyz[None, :, :] - tx_xyz[:, None, :]
        distance = np.linalg.norm(diff, axis=2)
        safe_distance = np.maximum(distance, max(min(dx, dy, dz) * 0.5, 1e-6))

        n_samples = int(np.clip(np.ceil(np.max(safe_distance) / max(min(dx, dy, dz), 1e-6)) + 1, 12, 48))
        t = np.linspace(0.0, 1.0, n_samples, dtype=float)
        sample_points = tx_xyz[:, None, None, :] + diff[:, :, None, :] * t[None, None, :, None]

        nx, ny, nz = sigma.shape
        normalized = (sample_points - bounds_min) / bounds_span
        ix = np.clip(np.rint(normalized[..., 0] * (nx - 1)).astype(np.intp), 0, nx - 1)
        iy = np.clip(np.rint(normalized[..., 1] * (ny - 1)).astype(np.intp), 0, ny - 1)
        iz = np.clip(np.rint(normalized[..., 2] * (nz - 1)).astype(np.intp), 0, nz - 1)

        sigma_path = sigma[ix, iy, iz]
        sigma_eff = np.mean(sigma_path, axis=2)
        sigma_var = np.mean(np.abs(sigma_path - sigma_eff[:, :, None]), axis=2)

        mu0 = 4e-7 * np.pi
        omega = 2.0 * np.pi * max(float(frequency), 1e-9)
        attenuation = np.sqrt(np.pi * max(float(frequency), 1e-9) * mu0 * sigma_eff)
        geometric_decay = 1.0 / (safe_distance * safe_distance + dx * dy)
        heterogeneity_gain = 1.0 + sigma_var / np.maximum(sigma_eff, 1e-6)

        response = (
            tx_strength[:, None]
            * sigma_eff
            * heterogeneity_gain
            * geometric_decay
            * np.exp(-attenuation * safe_distance)
            * np.sqrt(omega * mu0)
        )

        return AlgorithmResult(
            outputs={
                "response": response,
                "mesh_file": mesh,
            },
        )
