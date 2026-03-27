"""SimPEG TEM 3D 圆柱网格正演算法。

使用 SimPEG 的 TDEM 模块在圆柱形网格（CylMesh）上进行正演，
适用于地面回线 TEM 装置（中心回线或重叠回线）。
圆柱形网格利用了同心回线源的轴对称性，计算效率高。

参考：
  SimPEG TDEM 教程: https://docs.simpeg.xyz/content/tutorials/08-tdem/
  Haber et al. (2004), On optimization techniques for solving nonlinear inverse problems
"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class SimPEGTEMForward(BaseAlgorithm):
    """SimPEG TEM 3D 正演：圆柱网格，中心回线装置。"""

    def run(
        self,
        layer_resistivity: np.ndarray,
        layer_thickness: np.ndarray,
        loop_radius: float = 20.0,
        peak_current: float = 1.0,
        t_min: float = -5.0,
        t_max: float = -2.5,
        n_times: int = 25,
    ) -> AlgorithmResult:
        import discretize
        from simpeg.electromagnetics import time_domain as tdem
        from simpeg import maps

        rho = np.asarray(layer_resistivity, dtype=float).ravel()
        thick = np.asarray(layer_thickness, dtype=float).ravel()

        if len(thick) != len(rho) - 1:
            raise ValueError(
                f"layer_thickness 长度({len(thick)})应比 layer_resistivity 层数({len(rho)})少1。"
            )

        times = np.logspace(t_min, t_max, n_times)

        # ── 圆柱形网格（利用轴对称性）────────────────────────────────
        mesh, sigma_model, sigma_map = self._build_cyl_mesh(rho, thick)

        # ── 发射源：圆形线圈（离散为多个点偶极子）────────────────────
        n_seg = 20
        theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        loop_x = loop_radius * np.cos(theta)
        loop_y = loop_radius * np.sin(theta)
        loop_locs = np.c_[loop_x, loop_y, np.zeros(n_seg)]

        # 使用 CircularLoop 源（SimPEG 内置）
        rx_loc = np.array([[0.0, 0.0, 0.0]])    # 中心接收
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(
            rx_loc, times, orientation="z"
        )
        src = tdem.sources.CircularLoop(
            [rx],
            location=np.array([0.0, 0.0, 0.0]),
            radius=loop_radius,
            current=peak_current,
            n_turns=1,
        )
        survey = tdem.Survey([src])

        # ── 正演模拟 ─────────────────────────────────────────────────
        # Time steps: logarithmic distribution matching observation times
        time_steps = [(times[0] / 5, 5), (times[n_times // 4] / 5, 5),
                      (times[n_times // 2] / 5, 5), (times[-1] / 5, 5)]

        sim = tdem.Simulation3DElectricField(
            mesh,
            survey=survey,
            sigmaMap=sigma_map,
            t0=times[0] / 2,
            time_steps=time_steps,
        )

        dbdt = sim.dpred(sigma_model)

        # 视电阻率
        rho_app = self._apparent_resistivity(np.abs(dbdt), times, loop_radius)

        return AlgorithmResult(
            outputs={
                "dbdt": dbdt,
                "times": times,
                "rho_app": rho_app,
            }
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_cyl_mesh(rho, thick):
        import discretize
        from simpeg import maps

        # 径向：核心细网格 + 外部渐变
        dr = 5.0
        hr = [(dr, 10), (dr * 2, 10, 1.5)]
        # 垂直：地表向下
        dz = min(thick) / 3 if thick.size > 0 else 5.0
        hz = [(dz, 5, -1.3), (dz, 10), (dz * 3, 5, 1.3)]

        MeshClass = getattr(discretize, "CylindricalMesh", None) or discretize.CylMesh
        mesh = MeshClass([hr, 1, hz], origin="00N")

        # 将层状模型映射到网格
        total_depth = np.cumsum(thick)
        sigma_layers = 1.0 / rho

        zcc = -mesh.gridCC[:, 2]   # 深度（正向下）
        sigma_grid = np.full(mesh.n_cells, sigma_layers[-1])
        for i, depth in enumerate(total_depth):
            sigma_grid[zcc < depth] = sigma_layers[i]

        sigma_map = maps.IdentityMap(mesh)
        return mesh, sigma_grid, sigma_map

    @staticmethod
    def _apparent_resistivity(dbdt, times, radius):
        mu0 = 4 * np.pi * 1e-7
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (mu0 / (20 * np.pi)) ** (2 / 3) * (
                (np.pi * radius**2 * mu0) / (times * dbdt)
            ) ** (2 / 3)
        return np.where(np.isfinite(rho) & (rho > 0), rho, np.nan)
