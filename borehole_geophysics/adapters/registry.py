"""
方法适配器注册表
"""

from __future__ import annotations

from adapters.gravity import GravityAdapter
from adapters.magnetic_3c import Magnetic3CAdapter


ADAPTER_REGISTRY = {
    'gravity': GravityAdapter,
    'magnetic_3c': Magnetic3CAdapter,
}


def get_adapter(method_code):
    """按方法代码创建适配器实例。"""
    if method_code not in ADAPTER_REGISTRY:
        raise KeyError(f"未注册的方法: {method_code}")
    return ADAPTER_REGISTRY[method_code]()
