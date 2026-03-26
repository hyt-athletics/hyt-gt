"""将算法包中的 algorithm.py 编译为 Cython 扩展模块（.pyd / .so）。

编译后交付平台时只需提供 manifest.json + algorithm.pyd，不暴露 .py 源码。

用法：
    python tools/compile_algo.py <算法包目录>
    python tools/compile_algo.py algorithms/log_preproc_savgol
    python tools/compile_algo.py --all   # 编译 algorithms/ 下所有包

前置条件：
    uv add --optional compile cython setuptools
    （或）pip install "cython>=3.0" "setuptools>=68"
    Windows 还需安装 Visual Studio Build Tools（含 MSVC 编译器）
    Linux/Mac 需要 gcc/clang
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ALGO_ROOT = _ROOT / "algorithms"
_MODULE = "algorithm"


def compile_one(algo_dir: Path) -> None:
    src = algo_dir / f"{_MODULE}.py"
    if not src.exists():
        print(f"[跳过] {algo_dir.name}: 未找到 {_MODULE}.py")
        return

    print(f"[编译] {algo_dir.name} ...")
    with tempfile.TemporaryDirectory() as tmp:
        setup_py = Path(tmp) / "setup.py"
        setup_py.write_text(
            f"""\
from setuptools import setup
from Cython.Build import cythonize
setup(
    ext_modules=cythonize(
        r"{src}",
        compiler_directives={{"language_level": "3"}},
        quiet=True,
    ),
    script_args=["build_ext", "--inplace", "--build-lib", r"{tmp}"],
)
""",
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, str(setup_py)],
            capture_output=True,
            text=True,
            cwd=str(algo_dir),
        )
        if result.returncode != 0:
            print(f"[失败] {algo_dir.name}:\n{result.stderr}", file=sys.stderr)
            return

    # 清理 Cython 生成的 .c 文件
    c_file = algo_dir / f"{_MODULE}.c"
    if c_file.exists():
        c_file.unlink()
    build_dir = algo_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # 确认产物
    compiled = list(algo_dir.glob(f"{_MODULE}.pyd")) + list(
        algo_dir.glob(f"{_MODULE}.cpython-*.so")
    )
    if compiled:
        print(f"[完成] {algo_dir.name}: {compiled[0].name}")
    else:
        print(f"[警告] {algo_dir.name}: 编译完成但未找到输出文件", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="将算法 .py 编译为 Cython 扩展")
    parser.add_argument("target", nargs="?", help="算法包目录路径")
    parser.add_argument("--all", action="store_true", help="编译 algorithms/ 下所有包")
    args = parser.parse_args()

    if args.all:
        for d in sorted(_ALGO_ROOT.iterdir()):
            if d.is_dir() and (d / "manifest.json").exists():
                compile_one(d)
    elif args.target:
        compile_one(Path(args.target).resolve())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
