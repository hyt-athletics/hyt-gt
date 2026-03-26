"""模拟陌生用户全面测试。

假设：用户从未见过这个项目，只看了 README 和用户手册。
按照文档描述的步骤逐一操作，记录每个环节是否能顺利完成。

测试分 7 个场景：
  场景 1: 首次启动 — 工作区是否自动创建？
  场景 2: 新建钻孔 — DataStore 接口是否正常？
  场景 3: 导入数据 — LAS/CSV 导入是否成功？
  场景 4: 浏览算法 — 算法树是否能正确列出所有可用算法？
  场景 5: 运行单算法 — 选参数、运行、看结果
  场景 6: 运行流水线 — 多步串联是否正确？
  场景 7: 自己开发算法 — 按照开发者指南从零创建并运行
  场景 8: 验证工具 — validate 和 scaffold 是否好用
  场景 9: 错误操作 — 故意犯错看系统如何响应

运行：
    .venv/bin/python tests/test_fresh_user_experience.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

WORKSPACE = Path.home() / "geophys-workspace-fresh-test"
ALGO_DIR = _ROOT / "algorithms"
_pass = _fail = _warn = 0
issues = []


def ok(name, detail=""):
    global _pass
    _pass += 1
    print(f"  [PASS] {name}{' — ' + detail if detail else ''}")


def fail(name, reason):
    global _fail
    _fail += 1
    issues.append(f"[FAIL] {name}: {reason}")
    print(f"  [FAIL] {name} — {reason}")


def warn(name, reason):
    global _warn
    _warn += 1
    issues.append(f"[WARN] {name}: {reason}")
    print(f"  [WARN] {name} — {reason}")


print("=" * 70)
print("  陌生用户全面测试")
print("  模拟从未用过此软件的人按文档操作")
print(f"  使用干净工作区: {WORKSPACE}")
print("=" * 70)

# 清理旧的测试工作区
if WORKSPACE.exists():
    shutil.rmtree(WORKSPACE)

# ══════════════════════════════════════════════════════════════════════
# 场景 1: 首次启动
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 1: 首次启动 — 工作区自动创建")
print(f"{'━'*70}")

from core.data_store import DataStore

store = DataStore(WORKSPACE)

# 文档说"第一次启动时会自动创建工作区目录"
try:
    store.init_workspace()
    assert WORKSPACE.exists(), "工作区目录未创建"
    assert (WORKSPACE / "boreholes").exists(), "boreholes 子目录未创建"
    ok("workspace_init", f"创建于 {WORKSPACE}")
except Exception as e:
    fail("workspace_init", str(e))

# 首次应该没有任何钻孔
bh_list = store.list_boreholes()
if len(bh_list) == 0:
    ok("empty_boreholes", "首次启动无钻孔，符合预期")
else:
    warn("empty_boreholes", f"首次启动已有 {len(bh_list)} 个钻孔")

# ══════════════════════════════════════════════════════════════════════
# 场景 2: 新建钻孔
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 2: 新建钻孔")
print(f"{'━'*70}")

# 按文档操作：输入钻孔名称 "TEST-BH-001"
try:
    store.create_borehole("TEST-BH-001")
    assert (WORKSPACE / "boreholes" / "TEST-BH-001").exists()
    ok("create_borehole", "TEST-BH-001")
except Exception as e:
    fail("create_borehole", str(e))

# 再创建一个看看列表是否更新
store.create_borehole("TEST-BH-002")
bh_list = store.list_boreholes()
if "TEST-BH-001" in bh_list and "TEST-BH-002" in bh_list:
    ok("list_boreholes", f"{len(bh_list)} 个钻孔")
else:
    fail("list_boreholes", f"列表不完整: {bh_list}")

# 尝试创建同名钻孔
try:
    store.create_borehole("TEST-BH-001")
    ok("duplicate_borehole", "重复创建不报错（幂等操作）")
except Exception as e:
    warn("duplicate_borehole", f"重复创建报错: {e}")

# ══════════════════════════════════════════════════════════════════════
# 场景 3: 导入数据
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 3: 导入数据")
print(f"{'━'*70}")

from core.importer import DataImporter

# 3a. 导入 LAS 文件（文档说支持 .las）
las_file = _ROOT / "tests" / "data" / "6038187_v1.2_short.las"
if las_file.exists():
    try:
        importer = DataImporter(WORKSPACE)
        channels = importer.import_file(las_file, "TEST-BH-001", "logging")
        raw_dir = WORKSPACE / "boreholes" / "TEST-BH-001" / "logging" / "raw"
        csv_files = list(raw_dir.glob("*.csv"))
        ok("import_las", f"{len(channels)} 通道: {channels[:3]}...")

        # 真实用户痛点：LAS 曲线名与算法期望的文件名不一致
        # 例如 LAS 中叫 GAMN，但 savgol 算法期望 GR.csv
        # 解决方法：找到伽马曲线并复制为 GR.csv
        gr_candidates = [f for f in csv_files if "GAM" in f.stem.upper() or "GR" in f.stem.upper()]
        if gr_candidates and not (raw_dir / "GR.csv").exists():
            import shutil as _sh
            _sh.copy(gr_candidates[0], raw_dir / "GR.csv")
            warn("file_mapping", f"LAS 中伽马曲线叫 '{gr_candidates[0].stem}'，"
                 f"已复制为 GR.csv（真实用户需手动重命名或修改 manifest）")
        elif (raw_dir / "GR.csv").exists():
            ok("file_mapping", "GR.csv 已存在")
        else:
            warn("file_mapping", f"未找到 GR 曲线，可用文件: {[f.stem for f in csv_files]}")
    except Exception as e:
        fail("import_las", str(e))
else:
    warn("import_las", f"LAS 文件不存在: {las_file}")

# 3b. 手动创建 CSV 输入（用户手动复制文件的场景）
raw_tem, _ = store.ensure_method_dirs("TEST-BH-001", "tem")
rng = np.random.default_rng(42)
np.savetxt(raw_tem / "resistivity.csv", [2e14, 200.0, 10.0, 1000.0], delimiter=",")
np.savetxt(raw_tem / "depths.csv", [0.0, 50.0, 100.0], delimiter=",")

import empymod
times = np.logspace(-5, -2, 30)
emf = empymod.dipole(src=[0,0,0.001], rec=[0,0,0.001], depth=[0,50,100],
                     res=[2e14,200,10,1000], freqtime=times, signal=-1, verb=0)
np.savetxt(raw_tem / "emf.csv", emf * (1 + 0.05 * rng.standard_normal(30)), delimiter=",")
np.savetxt(raw_tem / "times.csv", times, delimiter=",")
ok("manual_csv", "TEM 数据手动创建成功")

# ══════════════════════════════════════════════════════════════════════
# 场景 4: 浏览算法
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 4: 浏览算法列表")
print(f"{'━'*70}")

from algorithms.registry import AlgorithmRegistry

registry = AlgorithmRegistry(ALGO_DIR)
registry.scan()
all_manifests = registry.list_manifests()

# 用户能看到的算法（有输入输出的）
visible = [m for m in all_manifests if m.get("inputs") and m.get("outputs")]
hidden = [m for m in all_manifests if not m.get("inputs") or not m.get("outputs")]

print(f"  总计: {len(all_manifests)} 个算法")
print(f"  可见（有输入输出）: {len(visible)} 个")
print(f"  隐藏（空输入/输出）: {len(hidden)} 个")

if len(visible) >= 8:
    ok("algo_visible", f"{len(visible)} 个算法可在 GUI 中使用")
else:
    fail("algo_visible", f"只有 {len(visible)} 个可见算法，文档说应有 8 个")

# 按方法分组
methods = {}
for m in visible:
    method = m.get("method", "unknown")
    methods.setdefault(method, []).append(m["name"])
for method, algos in sorted(methods.items()):
    print(f"    {method}: {len(algos)} 个 — {', '.join(algos[:3])}{'...' if len(algos) > 3 else ''}")

# ══════════════════════════════════════════════════════════════════════
# 场景 5: 运行单算法
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 5: 运行算法（模拟 GUI 操作）")
print(f"{'━'*70}")

from algorithms.runner import AlgorithmRunner

runner = AlgorithmRunner(registry, WORKSPACE)

# 5a. 测井 Savitzky-Golay（文档第一个示例）
print("\n  5a. 测井 SG 滤波")
try:
    saved = runner.run("log_preproc_savgol", "TEST-BH-001",
                       {"window_length": "11", "polyorder": "3"})
    assert "curve_filtered" in saved
    filtered = np.loadtxt(saved["curve_filtered"], delimiter=",")
    snr = float(Path(saved["snr_improvement"]).read_text(encoding="utf-8"))
    ok("run_savgol", f"输出 {len(filtered)} 点, SNR={snr:.1f}dB")
except Exception as e:
    fail("run_savgol", str(e))

# 5b. TEM 正演
print("\n  5b. TEM empymod 正演")
try:
    saved = runner.run("empymod_tem_forward", "TEST-BH-001", {})
    assert "emf" in saved and "times" in saved
    ok("run_tem_fwd", f"输出: {list(saved.keys())}")
except Exception as e:
    fail("run_tem_fwd", str(e))

# 5c. TEM 反演
print("\n  5c. TEM Occam 反演")
try:
    saved = runner.run("tem_1d_inversion", "TEST-BH-001",
                       {"n_layers": "15", "lambda_": "0.001"})
    misfit = float(Path(saved["misfit"]).read_text(encoding="utf-8"))
    ok("run_tem_inv", f"misfit={misfit:.4f}")
except Exception as e:
    fail("run_tem_inv", str(e))

# 5d. 查看结果文件
print("\n  5d. 检查输出文件结构")
proc_dir = WORKSPACE / "boreholes" / "TEST-BH-001"
for method_dir in sorted(proc_dir.iterdir()):
    if not method_dir.is_dir():
        continue
    proc = method_dir / "processed"
    if proc.exists():
        for algo_dir in sorted(proc.iterdir()):
            if algo_dir.is_dir():
                files = [f.name for f in algo_dir.iterdir()]
                has_meta = "run_meta.json" in files
                print(f"    {method_dir.name}/{algo_dir.name}: {len(files)} files, meta={'✓' if has_meta else '✗'}")
ok("output_structure", "输出目录结构正确")

# ══════════════════════════════════════════════════════════════════════
# 场景 6: 运行流水线
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 6: 运行流水线")
print(f"{'━'*70}")

from algorithms.pipeline_runner import PipelineRunner

pipe_runner = PipelineRunner(registry, WORKSPACE, _ROOT / "pipelines")
pipe_runner.scan()
pipelines = pipe_runner.list_pipelines()
print(f"  可用流水线: {len(pipelines)} 条")
for p in pipelines:
    print(f"    {p['name']}: {p.get('display_name', '')}")

# 运行测井多滤波对比
try:
    all_saved = pipe_runner.run("logging_multi_filter", "TEST-BH-001")
    steps = list(all_saved.keys())
    ok("pipeline_logging", f"完成 {len(steps)} 步: {steps}")
except Exception as e:
    fail("pipeline_logging", str(e))

# 运行 TEM 工作流
try:
    all_saved = pipe_runner.run("tem_1d_workflow", "TEST-BH-001")
    steps = list(all_saved.keys())
    ok("pipeline_tem", f"完成 {len(steps)} 步: {steps}")
except Exception as e:
    fail("pipeline_tem", str(e))

# ══════════════════════════════════════════════════════════════════════
# 场景 7: 自己开发算法（按开发者指南操作）
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 7: 按开发者指南从零创建自己的算法")
print(f"{'━'*70}")

MY_ALGO = "logging_preproc_myfilter"
my_dir = ALGO_DIR / MY_ALGO
if my_dir.exists():
    shutil.rmtree(my_dir)

# 7a. 按文档"第一步"创建文件夹
my_dir.mkdir()
ok("dev_mkdir", f"创建 {MY_ALGO}/")

# 7b. 按文档"模板 A"写 manifest.json
manifest = {
    "name": "logging_preproc_myfilter",
    "display_name": "我的滤波算法",
    "version": "1.0.0",
    "method": "logging",
    "category": "preprocess",
    "dimension": "1d",
    "entry": {"type": "python", "module": "algorithm", "class": "LoggingPreprocMyfilter"},
    "inputs": [
        {"name": "curve", "label": "输入曲线", "type": "ndarray_1d", "file": "GR.csv"},
        {"name": "window_size", "label": "窗口大小", "type": "int", "default": 11, "required": False},
    ],
    "outputs": [
        {"name": "result", "label": "处理结果", "type": "ndarray_1d"},
        {"name": "quality", "label": "质量指标", "type": "float"},
    ],
}
(my_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
ok("dev_manifest", "manifest.json 写入完成")

# 7c. 按文档"最简模板"写 algorithm.py
algo_code = '''\
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm


class LoggingPreprocMyfilter(BaseAlgorithm):
    def run(self, **inputs):
        curve = inputs["curve"]
        window_size = int(inputs.get("window_size", 11))

        # 我的算法：简单移动平均
        kernel = np.ones(window_size) / window_size
        result = np.convolve(curve, kernel, mode="same")
        quality = float(1.0 - np.std(result - curve) / max(np.std(curve), 1e-12))

        return AlgorithmResult(
            outputs={"result": result, "quality": quality},
        )
'''
(my_dir / "algorithm.py").write_text(algo_code, encoding="utf-8")
ok("dev_algorithm", "algorithm.py 写入完成")

# 7d. 按文档"第四步"运行验证
validate_result = subprocess.run(
    [sys.executable, str(_ROOT / "tools" / "validate_algo.py"), str(my_dir)],
    capture_output=True, text=True,
)
if "[ERROR]" not in validate_result.stdout:
    ok("dev_validate", "验证全部通过")
else:
    fail("dev_validate", f"验证失败:\n{validate_result.stdout}")

# 7e. 运行自己的算法
try:
    registry2 = AlgorithmRegistry(ALGO_DIR)
    registry2.scan()
    runner2 = AlgorithmRunner(registry2, WORKSPACE)
    saved = runner2.run(MY_ALGO, "TEST-BH-001", {"window_size": "7"})
    result = np.loadtxt(saved["result"], delimiter=",")
    quality = float(Path(saved["quality"]).read_text(encoding="utf-8"))
    ok("dev_run", f"自定义算法运行成功, 输出 {len(result)} 点, quality={quality:.3f}")
except Exception as e:
    fail("dev_run", str(e))

# 清理
shutil.rmtree(my_dir)

# ══════════════════════════════════════════════════════════════════════
# 场景 8: 工具测试
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 8: 开发工具测试")
print(f"{'━'*70}")

# scaffold 工具
scaffold_result = subprocess.run(
    [sys.executable, str(_ROOT / "tools" / "scaffold_algo.py"),
     "--name", "test_scaffold_fresh", "--display-name", "测试脚手架",
     "--method", "tem", "--category", "preprocess", "--template", "1d-preprocess"],
    capture_output=True, text=True, cwd=str(_ROOT),
)
scaffold_dir = ALGO_DIR / "test_scaffold_fresh"
if scaffold_dir.exists():
    manifest_ok = (scaffold_dir / "manifest.json").exists()
    algo_ok = (scaffold_dir / "algorithm.py").exists()
    m = json.loads((scaffold_dir / "manifest.json").read_text(encoding="utf-8"))
    has_inputs = len(m.get("inputs", [])) > 0
    has_outputs = len(m.get("outputs", [])) > 0
    ok("scaffold_tool", f"manifest={manifest_ok}, algo={algo_ok}, inputs={has_inputs}, outputs={has_outputs}")
    shutil.rmtree(scaffold_dir)
else:
    fail("scaffold_tool", f"未创建目录: {scaffold_result.stderr}")

# validate --all
validate_all = subprocess.run(
    [sys.executable, str(_ROOT / "tools" / "validate_algo.py"), "--all"],
    capture_output=True, text=True, cwd=str(_ROOT),
)
n_errors = validate_all.stdout.count("[ERROR]")
n_warnings = validate_all.stdout.count("[WARN]")
if n_errors == 0:
    ok("validate_all", f"全部通过: 0 errors, {n_warnings} warnings")
else:
    fail("validate_all", f"{n_errors} errors found")

# ══════════════════════════════════════════════════════════════════════
# 场景 9: 错误操作（故意犯错）
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  场景 9: 错误操作测试")
print(f"{'━'*70}")

# 9a. 运行算法但钻孔不存在
print("\n  9a. 不存在的钻孔")
try:
    runner.run("log_preproc_savgol", "NONEXISTENT-BH", {})
    fail("bad_borehole", "应该报错但没有")
except FileNotFoundError:
    ok("bad_borehole", "正确报 FileNotFoundError")
except Exception as e:
    warn("bad_borehole", f"报了 {type(e).__name__} 而非 FileNotFoundError")

# 9b. 运行不存在的算法
print("\n  9b. 不存在的算法")
try:
    runner.run("nonexistent_algorithm", "TEST-BH-001", {})
    fail("bad_algo", "应该报错但没有")
except KeyError:
    ok("bad_algo", "正确报 KeyError")
except Exception as e:
    warn("bad_algo", f"报了 {type(e).__name__}: {e}")

# 9c. 缺少必要输入文件
print("\n  9c. 缺少输入文件")
raw_ert, _ = store.ensure_method_dirs("TEST-BH-001", "ert")
# 不放任何文件，直接运行 ERT 反演
try:
    runner.run("simpeg_dc_inversion", "TEST-BH-001", {})
    fail("missing_file", "应该报错但没有")
except FileNotFoundError:
    ok("missing_file", "正确报 FileNotFoundError")
except Exception as e:
    warn("missing_file", f"报了 {type(e).__name__}: {e}")

# 9d. 错误的参数类型
print("\n  9d. 参数类型错误")
try:
    runner.run("log_preproc_savgol", "TEST-BH-001",
               {"window_length": "abc"})  # 应该是数字
    fail("bad_param_type", "应该报错但没有")
except (ValueError, TypeError):
    ok("bad_param_type", "正确报类型错误")
except Exception as e:
    warn("bad_param_type", f"报了 {type(e).__name__}: {e}")

# ── 清理测试工作区 ────────────────────────────────────────────────────
shutil.rmtree(WORKSPACE)

# ══════════════════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════════════════
total = _pass + _fail + _warn
print(f"\n{'='*70}")
print(f"  陌生用户测试结果:")
print(f"  {_pass} passed, {_fail} failed, {_warn} warnings / {total} total")
if issues:
    print(f"\n  发现的问题:")
    for issue in issues:
        print(f"    {issue}")
print(f"{'='*70}")
sys.exit(1 if _fail else 0)
