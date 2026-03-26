import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .base import AlgorithmResult, BaseAlgorithm

# C 函数参数角色标识
_ROLE_INPUT = "input"
_ROLE_OUTPUT = "output"


class CLibAlgorithm(BaseAlgorithm):
    """将 C 动态库函数包装为 BaseAlgorithm。

    C 函数约定（manifest["c_signature"] 描述）：
        int func_name(
            double* input_arr, int n,   // 每个数组输入：数据指针 + 长度
            ...
            double* output_arr,         // 输出数组（调用方预分配）
            ...
        )
    返回 0 表示成功，非零为错误码。

    manifest["c_signature"] 示例：
    {
        "args": [
            {"name": "data", "role": "input"},
            {"name": "result", "role": "output", "size_from": "data"}
        ]
    }
    """

    def __init__(self, lib_path: Path, manifest: dict):
        if not lib_path.exists():
            raise FileNotFoundError(
                f"C 动态库不存在: {lib_path}\n"
                f"Windows 请提供 .dll，Linux 提供 .so，macOS 提供 .dylib。"
            )
        if sys.platform == "win32":
            os.add_dll_directory(str(lib_path.parent))
        self._lib = ctypes.CDLL(str(lib_path))
        self._manifest = manifest
        self._func_name = manifest["entry"]["function"]

    def run(self, **inputs: Any) -> AlgorithmResult:
        sig = self._manifest.get("c_signature", {})
        fn = getattr(self._lib, self._func_name)
        fn.restype = ctypes.c_int

        args: list[Any] = []
        output_arrays: dict[str, np.ndarray] = {}

        for arg_spec in sig.get("args", []):
            name = arg_spec["name"]
            role = arg_spec["role"]

            if role == _ROLE_INPUT:
                arr = np.ascontiguousarray(inputs[name], dtype=np.float64)
                ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                args.append(ptr)
                args.append(ctypes.c_int(arr.size))

            elif role == _ROLE_OUTPUT:
                # 输出数组大小：固定值 or 与某输入同大小
                if "size" in arg_spec:
                    n = int(arg_spec["size"])
                elif "size_from" in arg_spec:
                    src = inputs[arg_spec["size_from"]]
                    n = np.asarray(src).size
                else:
                    raise ValueError(
                        f"c_signature arg '{name}' (output) 必须指定 size 或 size_from"
                    )
                out_arr = np.zeros(n, dtype=np.float64)
                output_arrays[name] = out_arr
                args.append(out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        ret = fn(*args)
        if ret != 0:
            raise RuntimeError(
                f"C 函数 '{self._func_name}' 返回错误码 {ret}。"
                f"请检查输入数据或联系算法开发者。"
            )

        return AlgorithmResult(outputs=output_arrays)
