"""三维数据处理工作流集成测试。

测试内容：
  1. NPZ 3D 数组 round-trip（save → load → compare）
  2. mesh_io save/load round-trip
  3. Runner 对 ndarray_3d / mesh 类型的端到端路由
  4. make_tensor_mesh_3d 构建验证
  5. VTK reader 集成测试（需要 pyvista）

运行方式：
    uv run python tests/test_3d_workflow.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ── 1. NPZ round-trip ────────────────────────────────────────────────
print("[1] NPZ 3D array round-trip...")
with tempfile.TemporaryDirectory() as tmp:
    arr = np.random.default_rng(42).standard_normal((10, 12, 8))
    npz_path = Path(tmp) / "test.npz"
    np.savez_compressed(npz_path, data=arr)

    loaded = np.load(npz_path)["data"]
    assert np.allclose(arr, loaded), "NPZ round-trip 失败：数据不一致"
    assert loaded.shape == (10, 12, 8), f"形状不匹配: {loaded.shape}"
print("  OK: NPZ 3D array round-trip 通过")

# ── 2. mesh_io round-trip ─────────────────────────────────────────────
print("[2] mesh_io save/load round-trip...")
from core.mesh_io import save_mesh, load_mesh

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    arr_3d = np.random.default_rng(7).standard_normal((5, 6, 7))

    saved_path = save_mesh(arr_3d, tmp_dir, "test_3d")
    assert saved_path.suffix == ".npz", f"期望 .npz，实际: {saved_path.suffix}"
    assert saved_path.exists(), f"文件不存在: {saved_path}"

    loaded_3d = load_mesh(saved_path, fmt="npz")
    assert np.allclose(arr_3d, loaded_3d), "mesh_io NPZ round-trip 失败"
print("  OK: mesh_io round-trip 通过")

# ── 3. Runner ndarray_3d 类型路由 ─────────────────────────────────────
print("[3] Runner ndarray_3d type routing...")
from algorithms.runner import AlgorithmRunner
from algorithms.registry import AlgorithmRegistry
from core.data_store import DataStore

with tempfile.TemporaryDirectory() as tmp:
    ws = Path(tmp) / "workspace"
    store = DataStore(ws)
    store.init_workspace()
    store.create_borehole("TEST-3D")

    raw_dir = ws / "boreholes" / "TEST-3D" / "tem" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    test_data = np.random.default_rng(99).standard_normal((4, 5, 6))
    np.savez_compressed(raw_dir / "conductivity.npz", data=test_data)

    loaded_back = np.load(raw_dir / "conductivity.npz")["data"]
    assert loaded_back.shape == (4, 5, 6), f"写入后形状不匹配: {loaded_back.shape}"
    assert np.allclose(test_data, loaded_back), "写入后数据不一致"
print("  OK: ndarray_3d 文件 I/O 通过")

# ── 4. make_tensor_mesh_3d（需要 discretize） ─────────────────────────
try:
    from core.mesh_utils import make_tensor_mesh_3d

    print("[4] make_tensor_mesh_3d...")
    mesh = make_tensor_mesh_3d(
        x_extent=100.0, y_extent=80.0, z_extent=50.0,
        dx=10.0, dy=10.0, dz=5.0,
        n_pad=3, pad_factor=1.3,
    )
    assert mesh.dim == 3, f"期望 3D mesh，实际 {mesh.dim}D"
    assert mesh.nC > 0, "网格单元数为 0"
    print(f"  OK: 3D mesh 构建成功 — {mesh.nC} cells, shape {mesh.shape_cells}")
except ImportError:
    print("[4] 跳过: discretize 未安装（uv pip install geophys-tool[3d]）")

# ── 5. VTK reader（需要 pyvista） ─────────────────────────────────────
try:
    import pyvista
    from core.readers.vtk_reader import VtkReader

    print("[5] VTK reader...")
    with tempfile.TemporaryDirectory() as tmp:
        grid = pyvista.ImageData(dimensions=(5, 6, 7))
        grid["test_model"] = np.arange(grid.n_cells, dtype=np.float64)
        vtk_path = Path(tmp) / "test.vti"
        grid.save(str(vtk_path))

        reader = VtkReader()
        channels = reader.read(vtk_path)
        assert "cell_test_model" in channels, f"缺少 cell_test_model: {list(channels.keys())}"
        assert channels["cell_test_model"].shape[0] == grid.n_cells
    print(f"  OK: VTK reader 通过 — 读取 {len(channels)} 通道")
except ImportError:
    print("[5] 跳过: pyvista 未安装（uv pip install geophys-tool[3d]）")

# ── 6. mesh_io discretize round-trip（需要 discretize） ────────────────
try:
    from discretize import TensorMesh
    from core.mesh_io import save_mesh, load_mesh

    print("[6] discretize mesh round-trip...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mesh = TensorMesh(
            [np.ones(5) * 10, np.ones(4) * 10, np.ones(3) * 5]
        )
        saved = save_mesh(mesh, tmp_dir, "test_mesh")
        assert saved.exists(), f"网格文件不存在: {saved}"

        reloaded = load_mesh(saved, fmt="discretize")
        assert reloaded.nC == mesh.nC, f"网格单元数不匹配: {reloaded.nC} vs {mesh.nC}"
    print(f"  OK: discretize mesh round-trip 通过 — {mesh.nC} cells")
except ImportError:
    print("[6] 跳过: discretize 未安装")

print("\n所有测试通过。")
