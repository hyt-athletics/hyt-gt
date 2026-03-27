"""从 lasio 官方测试库下载示例 LAS 文件到 tests/data/。

用法：
    python tests/download_sample_las.py

下载的文件可直接通过 GUI "导入 LAS" 按钮导入到工作区。
"""
import sys
import urllib.request
from pathlib import Path

_URL = (
    "https://raw.githubusercontent.com/kinverarity1/lasio"
    "/main/tests/examples/6038187_v1.2_short.las"
)
_OUT = Path(__file__).parent / "data" / "6038187_v1.2_short.las"


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    if _OUT.exists():
        print(f"文件已存在，跳过下载：{_OUT}")
        return
    print(f"正在下载 {_URL} ...")
    try:
        urllib.request.urlretrieve(_URL, _OUT)
        print(f"已保存到 {_OUT}")
    except Exception as e:
        print(f"下载失败：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
