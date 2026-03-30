"""
多方法适配器基类

平台负责建模、网格、人机交互和可视化；
具体正反演算法交给外部库或外部程序。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MethodAdapter(ABC):
    """所有地球物理方法适配器的统一接口。"""

    method_name = "未命名方法"
    method_code = "unknown"
    required_properties = []
    optional_properties = []

    @abstractmethod
    def create_survey(self, well, **kwargs):
        """创建观测系统描述。"""

    @abstractmethod
    def get_mesh_requirements(self):
        """返回方法对应的网格要求。"""

    @abstractmethod
    def prepare_mesh(self, mesh_data, survey):
        """把统一网格格式转成外部引擎所需格式。"""

    @abstractmethod
    def prepare_model(self, mesh_data):
        """把统一模型格式转成外部引擎所需格式。"""

    @abstractmethod
    def forward(self, mesh, model, survey, **kwargs):
        """调用外部正演引擎。"""

    @abstractmethod
    def inverse(self, mesh, data, survey, **kwargs):
        """调用外部反演引擎。"""

    @abstractmethod
    def convert_result(self, raw_result, **kwargs):
        """把外部结果转成项目内部结果对象。"""

    def check_dependencies(self, operation='forward'):
        """检查外部依赖是否满足。"""
        return True, "ok"
