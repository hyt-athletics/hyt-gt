"""SimPEG DC 电阻率 2D 反演算法。

使用 SimPEG 的 DC 模块进行 2D 直流电阻率正反演（Simulation2DNodal）。
支持 Wenner、Dipole-Dipole、Schlumberger 等任意排列。

参考：
  SimPEG DC 教程: https://docs.simpeg.xyz/content/tutorials/06-dc/
  Oldenburg & Li (1994), Inversion of induced polarization data
"""
import numpy as np

from algorithms.base import AlgorithmResult, BaseAlgorithm


class SimPEGDCInversion(BaseAlgorithm):
    """SimPEG DC 电阻率 2D 反演（Simulation2DNodal + Tikhonov 正则化）。"""

    def run(
        self,
        apparent_resistivity: np.ndarray,
        electrode_positions: np.ndarray,
        abmn_indices: np.ndarray,
        rho_background: float = 100.0,
        alpha_s: float = 1e-3,
        alpha_x: float = 1.0,
        alpha_z: float = 1.0,
        max_iter: int = 8,
    ) -> AlgorithmResult:
        from simpeg.electromagnetics.static import resistivity as dc
        from simpeg import (
            maps,
            optimization,
            regularization,
            data_misfit,
            inverse_problem,
            inversion,
            directives,
            data as simpeg_data,
        )
        from discretize import TensorMesh

        rho_obs = np.asarray(apparent_resistivity, dtype=float).ravel()
        elec = np.asarray(electrode_positions, dtype=float)
        abmn = np.asarray(abmn_indices, dtype=int)

        n_data = len(rho_obs)
        if abmn.shape != (n_data, 4):
            raise ValueError(
                f"abmn_indices shape 应为 ({n_data}, 4)，实际为 {abmn.shape}"
            )

        # ── 构建 SimPEG 调查（电极坐标以剖面中心为原点）────────────────
        x_center_pre = (elec[:, 0].min() + elec[:, 0].max()) / 2
        elec_centered = elec.copy()
        elec_centered[:, 0] -= x_center_pre
        source_list = self._build_sources(abmn, elec_centered, rho_obs)
        survey = dc.Survey(source_list)

        # ── 构建 2D 张量网格 ─────────────────────────────────────────
        x_min_c = elec_centered[:, 0].min()
        x_max_c = elec_centered[:, 0].max()
        L = x_max_c - x_min_c
        dx = L / 40
        dz = L / 30

        hx = [(dx * 3, 5, -1.4), (dx, 40), (dx * 3, 5, 1.4)]
        hz = [(dz, 3, 1.4), (dz, 20)]
        mesh = TensorMesh([hx, hz], origin="CN")

        active = mesh.gridCC[:, 1] < 0.0   # 地面以下

        n_active = int(active.sum())
        log_rho_map = (
            maps.InjectActiveCells(mesh, active, np.log(rho_background))
            * maps.ExpMap()
        )

        # ── 正演模拟器 ───────────────────────────────────────────────
        sim = dc.Simulation2DNodal(
            mesh,
            survey=survey,
            rhoMap=log_rho_map,
        )

        # 初始化视电阻率几何因子（必须在 dpred/inversion 前调用）
        survey.set_geometric_factor(space_type="halfspace")

        dobj = simpeg_data.Data(survey=survey, dobs=rho_obs, relative_error=0.05)

        # ── 反演 ────────────────────────────────────────────────────
        dmisfit = data_misfit.L2DataMisfit(data=dobj, simulation=sim)

        reg = regularization.WeightedLeastSquares(
            mesh,
            active_cells=active,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_z=alpha_z,
        )

        opt = optimization.InexactGaussNewton(maxIter=max_iter, maxIterCG=20)
        inv_prob = inverse_problem.BaseInvProblem(dmisfit, reg, opt)

        beta_sch = directives.BetaSchedule(coolingFactor=2, coolingRate=2)
        target = directives.TargetMisfit(chifact=1.0)

        inv = inversion.BaseInversion(
            inv_prob, directiveList=[beta_sch, target]
        )

        m0 = np.log(rho_background) * np.ones(n_active)
        # 预热：强制计算视电阻率几何因子缓存
        _ = sim.dpred(m0)
        # 设置初始 beta（手动代替 BetaEstimate_ByEig）
        inv_prob.beta = 1e2
        m_rec = inv.run(m0)

        rho_model = np.exp(m_rec)
        predicted = np.exp(sim.dpred(m_rec))
        phi_d = float(dmisfit(m_rec))

        warnings = []
        if phi_d > n_data * 2:
            warnings.append(
                f"phi_d={phi_d:.1f} 偏大，建议增加 max_iter 或调整正则化参数。"
            )

        return AlgorithmResult(
            outputs={
                "resistivity_model": rho_model,
                "predicted": predicted,
                "misfit": phi_d,
            },
            warnings=warnings,
        )

    @staticmethod
    def _build_sources(abmn, elec_locs, dobs):
        from simpeg.electromagnetics.static import resistivity as dc

        # SimPEG DC Simulation2DNodal 使用 2D 坐标 [x, z]
        # elec_locs[:, 0] = x,  elec_locs[:, 1] = z（通常 z=0 地表）
        source_list = []
        for i, (a, b, m, n) in enumerate(abmn):
            a_loc = elec_locs[a]   # shape (2,): [x, z]
            b_loc = elec_locs[b]
            m_loc = elec_locs[m].reshape(1, 2)
            n_loc = elec_locs[n].reshape(1, 2)
            rx = dc.receivers.Dipole(
                locations_m=m_loc,
                locations_n=n_loc,
                data_type="apparent_resistivity",
            )
            src = dc.sources.Dipole([rx], location_a=a_loc, location_b=b_loc)
            source_list.append(src)
        return source_list
