"""流水线执行器：按顺序调用多个算法步骤，自动传递中间结果。

流水线定义文件格式（pipelines/<name>.json）：
{
    "name": "tem_standard_workflow",
    "display_name": "TEM 标准处理流程",
    "method": "tem",
    "steps": [
        {
            "algo": "tem_preproc_wavelet",
            "output_key": "emf_clean"        // 可选：给输出起别名供后续步骤引用
        },
        {
            "algo": "tem_forward_2d",
            "input_map": {"emf": "emf_clean"} // 将上一步的 emf_clean 映射为本步的 emf 输入
        }
    ]
}
"""
import json
from pathlib import Path
from typing import Any

from .registry import AlgorithmRegistry
from .runner import AlgorithmRunner


class PipelineRunner:
    def __init__(
        self,
        registry: AlgorithmRegistry,
        workspace_dir: Path,
        pipelines_dir: Path,
    ):
        self._registry = registry
        self._runner = AlgorithmRunner(registry, workspace_dir)
        self._pipelines_dir = pipelines_dir
        self._pipelines: dict[str, dict] = {}

    def scan(self) -> list[dict]:
        """扫描 pipelines/ 目录，加载所有流水线定义。"""
        results = []
        if not self._pipelines_dir.exists():
            return results
        for p in self._pipelines_dir.glob("*.json"):
            try:
                pipeline = json.loads(p.read_text(encoding="utf-8"))
                self._pipelines[pipeline["name"]] = pipeline
                results.append(pipeline)
            except Exception as e:
                print(f"[pipeline] 跳过 {p.name}：解析失败 — {e}")
        return results

    def list_pipelines(self) -> list[dict]:
        return list(self._pipelines.values())

    def get_pipeline(self, name: str) -> dict:
        if name not in self._pipelines:
            raise KeyError(f"流水线未找到: {name}。请先调用 scan()。")
        return self._pipelines[name]

    def run(
        self,
        pipeline_name: str,
        borehole_name: str,
        user_params: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, dict[str, Path]]:
        """执行流水线，返回每个步骤的结果文件字典。

        Args:
            pipeline_name: 流水线名称
            borehole_name: 钻孔名称
            user_params: 可选，各步骤的额外标量参数
                         格式：{algo_name: {param_name: value_str}}

        Returns:
            {algo_name: {output_name: file_path}}
        """
        pipeline = self.get_pipeline(pipeline_name)
        user_params = user_params or {}

        # 中间结果缓存：key → 文件路径
        result_cache: dict[str, Path] = {}
        all_saved: dict[str, dict[str, Path]] = {}

        for step in pipeline["steps"]:
            algo_name = step["algo"]
            input_map: dict[str, str] = step.get("input_map", {})
            output_key: str | None = step.get("output_key")

            # 构建此步骤的 user_params（标量参数）
            params = user_params.get(algo_name, {})

            # 将流水线中间结果注入为文件引用
            # input_map: {algo输入名: cache_key}
            # 通过在工作区创建符号/拷贝来传递，这里直接修改 manifest 的 file 字段
            # 实际做法：临时覆盖 runner 的文件查找路径
            saved = self._runner.run(
                algo_name=algo_name,
                borehole_name=borehole_name,
                user_params=params,
                file_overrides={
                    algo_input: result_cache[cache_key]
                    for algo_input, cache_key in input_map.items()
                    if cache_key in result_cache
                },
            )

            all_saved[algo_name] = saved

            # 更新缓存：默认用 output_name 作为 key；output_key 提供别名
            for out_name, path in saved.items():
                result_cache[out_name] = path
            if output_key and saved:
                # 取第一个输出作为 output_key 的值
                first_path = next(iter(saved.values()))
                result_cache[output_key] = first_path

        return all_saved
