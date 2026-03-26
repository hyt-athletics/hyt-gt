"""Generate synthetic test datasets for all methods.

Creates TEST-WELL borehole in workspace with:
  - logging/raw/: GR.csv (synthetic gamma ray curve)
  - tem/raw/:     resistivity.csv, depths.csv, emf.csv, times.csv
  - ert/raw/:     apparent_resistivity.csv, electrode_positions.csv, abmn_indices.csv

Usage:
    .venv/bin/python tests/build_test_datasets.py
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from core.data_store import DataStore

WORKSPACE = Path.home() / "geophys-workspace"
BH_NAME = "TEST-WELL"

store = DataStore(WORKSPACE)
store.init_workspace()
store.create_borehole(BH_NAME)

print(f"Workspace: {WORKSPACE}")
print(f"Borehole:  {BH_NAME}")
print("=" * 60)

# ── LOGGING ───────────────────────────────────────────────────────────
raw_log, _ = store.ensure_method_dirs(BH_NAME, "logging")

rng = np.random.default_rng(42)
n_pts = 500
depth = np.linspace(0, 500, n_pts)

# Synthetic GR: shale/sand alternation + noise
gr_clean = 30 + 60 * (np.sin(2 * np.pi * depth / 50) > 0).astype(float)
gr_clean += 20 * np.sin(2 * np.pi * depth / 200)
gr = gr_clean + rng.standard_normal(n_pts) * 8

np.savetxt(raw_log / "GR.csv", gr, delimiter=",")
np.savetxt(raw_log / "depth.csv", depth, delimiter=",")
print(f"[logging] GR.csv: {n_pts} points, range [{gr.min():.0f}, {gr.max():.0f}] API")

# ── TEM ───────────────────────────────────────────────────────────────
raw_tem, _ = store.ensure_method_dirs(BH_NAME, "tem")

# 3-layer model: air / 200 Ohm.m overburden / 10 Ohm.m conductor / 1000 Ohm.m basement
res_model = np.array([2e14, 200.0, 10.0, 1000.0])
dep_model = np.array([0.0, 50.0, 100.0])

np.savetxt(raw_tem / "resistivity.csv", res_model, delimiter=",")
np.savetxt(raw_tem / "depths.csv", dep_model, delimiter=",")

# For simpeg_tem_forward: layer_thickness = n_layers - 1 values
# res_model has 4 layers (air, 200, 10, 1000), so need 3 thicknesses
layer_thickness = np.array([0.001, 50.0, 50.0])  # air(~0) + 50m overburden + 50m conductor
np.savetxt(raw_tem / "layer_thickness.csv", layer_thickness, delimiter=",")

# Generate TEM forward response using empymod
try:
    import empymod
    times = np.logspace(-5, -2, 30)
    emf = empymod.dipole(
        src=[0, 0, 0.001],
        rec=[0, 0, 0.001],
        depth=dep_model.tolist(),
        res=res_model.tolist(),
        freqtime=times,
        signal=-1,
        verb=0,
    )
    # Add 5% noise
    emf_noisy = emf * (1 + 0.05 * rng.standard_normal(len(emf)))
    np.savetxt(raw_tem / "emf.csv", emf_noisy, delimiter=",")
    np.savetxt(raw_tem / "times.csv", times, delimiter=",")
    print(f"[tem] Forward computed: {len(times)} time gates, empymod 3-layer model")
except ImportError:
    # Fallback: synthetic exponential decay
    times = np.logspace(-5, -2, 30)
    emf_noisy = 1e-6 * np.exp(-times * 1e3) * (1 + 0.05 * rng.standard_normal(30))
    np.savetxt(raw_tem / "emf.csv", emf_noisy, delimiter=",")
    np.savetxt(raw_tem / "times.csv", times, delimiter=",")
    print("[tem] empymod not available, using synthetic exponential decay")

# ── ERT ───────────────────────────────────────────────────────────────
raw_ert, _ = store.ensure_method_dirs(BH_NAME, "ert")

# Wenner array: 20 electrodes, 5m spacing
n_elec = 20
spacing = 5.0
elec_x = np.arange(n_elec) * spacing
elec_positions = np.column_stack([elec_x, np.zeros(n_elec)])  # [x, z]

# Generate Wenner ABMN indices
abmn_list = []
for a_sep in range(1, n_elec // 3 + 1):
    for i in range(n_elec - 3 * a_sep):
        a = i
        m = i + a_sep
        n = i + 2 * a_sep
        b = i + 3 * a_sep
        abmn_list.append([a, b, m, n])
abmn = np.array(abmn_list, dtype=int)

# Synthetic apparent resistivity: 2-layer model (100 / 20 Ohm.m at 15m depth)
# Use geometric factor approximation for Wenner
rho_app = np.zeros(len(abmn))
for i, (a, b, m, n) in enumerate(abmn):
    a_pos = elec_x[a]
    b_pos = elec_x[b]
    m_pos = elec_x[m]
    n_pos = elec_x[n]
    sep = abs(m_pos - a_pos)
    # Deeper separations see lower resistivity (conductor at 15m)
    depth_factor = 1.0 - 0.7 * np.clip(sep / 40.0, 0, 1)
    rho_app[i] = 100.0 * depth_factor + rng.standard_normal() * 2

np.savetxt(raw_ert / "apparent_resistivity.csv", rho_app, delimiter=",")
np.savetxt(raw_ert / "electrode_positions.csv", elec_positions, delimiter=",")
np.savetxt(raw_ert / "abmn_indices.csv", abmn, delimiter=",", fmt="%d")
print(f"[ert] Wenner array: {n_elec} electrodes, {len(abmn)} measurements")

# ── EM ────────────────────────────────────────────────────────────────
raw_em, _ = store.ensure_method_dirs(BH_NAME, "em")

# 3D conductivity model (small: 10x10x10)
nx_em, ny_em, nz_em = 10, 10, 10
sigma_3d = np.ones((nx_em, ny_em, nz_em)) * 0.01  # 100 Ohm.m background
sigma_3d[3:7, 3:7, 4:8] = 0.1  # conductor block (10 Ohm.m)
np.savez_compressed(raw_em / "conductivity.npz", data=sigma_3d)

# TX: 3 sources along borehole 1 (x=20, y=25)
tx_locs = np.array([
    [20.0, 25.0, -10.0],
    [20.0, 25.0, -25.0],
    [20.0, 25.0, -40.0],
])
np.savetxt(raw_em / "tx_locations.csv", tx_locs, delimiter=",")

# RX: 5 receivers along borehole 2 (x=30, y=25)
rx_locs = np.array([
    [30.0, 25.0, -5.0],
    [30.0, 25.0, -15.0],
    [30.0, 25.0, -25.0],
    [30.0, 25.0, -35.0],
    [30.0, 25.0, -45.0],
])
np.savetxt(raw_em / "rx_locations.csv", rx_locs, delimiter=",")
print(f"[em] 3D conductivity ({nx_em}x{ny_em}x{nz_em}), 3 TX, 5 RX")

# ── IP (reuse ERT data + add chargeability) ───────────────────────────
raw_ip, _ = store.ensure_method_dirs(BH_NAME, "ip")

# Copy ERT survey geometry to IP
for fname in ("apparent_resistivity.csv", "electrode_positions.csv", "abmn_indices.csv"):
    import shutil
    shutil.copy(raw_ert / fname, raw_ip / fname)

# Synthetic chargeability (higher in conductor zone)
chg_app = np.zeros(len(abmn))
for i, (a, b, m, n) in enumerate(abmn):
    sep = abs(elec_x[m] - elec_x[a])
    chg_app[i] = 5.0 + 20.0 * np.clip(sep / 40.0, 0, 1) + rng.standard_normal() * 1
np.savetxt(raw_ip / "apparent_chargeability.csv", chg_app, delimiter=",")
print(f"[ip] Chargeability added: {len(chg_app)} measurements")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test datasets ready. Workspace structure:")
bh_dir = WORKSPACE / "boreholes" / BH_NAME
for method_dir in sorted(bh_dir.iterdir()):
    if not method_dir.is_dir():
        continue
    raw_dir = method_dir / "raw"
    if raw_dir.exists():
        files = sorted(raw_dir.iterdir())
        print(f"  {method_dir.name}/raw/")
        for f in files:
            size = f.stat().st_size
            print(f"    {f.name:35s} {size/1024:6.1f} KB")
print("=" * 60)
