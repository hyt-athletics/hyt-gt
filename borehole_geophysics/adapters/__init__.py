"""
多方法适配器层
"""

from adapters.base import MethodAdapter
from adapters.registry import ADAPTER_REGISTRY, get_adapter

__all__ = ['MethodAdapter', 'ADAPTER_REGISTRY', 'get_adapter']
