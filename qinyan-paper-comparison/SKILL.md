---
name: qinyan-paper-comparison
description: "沁言学术论文对比分析 - 通过沁言学术OpenAPI检索并分析多篇论文，从研究目标、方法论、实验设计、核心结果等维度进行系统化对比，生成结构化对比分析表格和报告。触发词：论文对比、对比分析、paper comparison、对比论文、方法对比、compare papers、论文比较"
version: 1.0.0
---

# 沁言学术 - 论文对比分析

## 概述

本技能通过沁言学术OpenAPI检索文献并调用论文分析接口，对多篇论文从研究目标、方法论、实验设计、核心结果、优缺点等维度进行系统化对比分析，生成结构化的对比报告。适用于文献综述中的方法对比、选型评估、论文审阅等场景。

**学术诚信声明：** 对比分析基于论文实际内容，不捏造或夸大任何论文的结果。如无法获取某篇论文的完整信息，将明确标注信息来源的局限性。

## 前置条件

使用前请确保：
1. 已设置 `QINYAN_API_KEY` 环境变量（前往 https://platform.qinyanai.com/ 申请）
2. 环境中可用 `curl` 和 `python3`

```bash
export QINYAN_API_KEY="your-api-key-here"
```

## 工作流程

### 第一步：确定对比论文

获取用户提供的论文列表。论文可通过以下方式指定：
- 直接提供论文标题
- 提供DOI
- 提供论文URL/PDF链接
- 描述研究方向，由系统检索后推荐

### 第二步：检索论文信息

使用 `scripts/search.sh` 检索各论文的完整元数据：

```bash
bash scripts/search.sh <source> '<json_payload>'
# source: google | wanfang | pubmed | arxiv
```

**各数据库参数：**

| 数据源 | 命令 | 核心参数 |
|--------|------|----------|
| Google Scholar | `bash scripts/search.sh google '<json>'` | query, max_results(≤20), author |
| 万方 | `bash scripts/search.sh wanfang '<json>'` | query, max_results(≤100) |
| PubMed | `bash scripts/search.sh pubmed '<json>'` | query(英文), max_results(≤200) |
| ArXiv | `bash scripts/search.sh arxiv '<json>'` | query(英文), max_results(≤100), author |

```bash
# 示例：检索待对比的论文
bash scripts/search.sh google '{"query": "BERT pre-training deep bidirectional transformers Devlin", "max_results": 5}'
bash scripts/search.sh google '{"query": "GPT-3 language models few-shot learners Brown", "max_results": 5}'
bash scripts/search.sh arxiv '{"query": "LLaMA open efficient foundation language models Touvron", "max_results": 5}'
```

### 第三步：深度分析各论文

使用 `scripts/analyze.sh` 对每篇论文进行深度分析：

```bash
bash scripts/analyze.sh '<json_payload>'
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| title | string | 是 | 论文标题 |
| authors | string[] | 是 | 作者列表 |
| abstract | string | 否 | 摘要 |
| doi | string | 否 | DOI |
| source_url | string | 否 | 论文页面URL |
| pdf_url | string | 否 | PDF直链（优先级最高） |
| language | string | 否 | 输出语言："中文"或"en" |

```bash
# 对每篇论文调用分析
bash scripts/analyze.sh '{"title": "BERT: Pre-training of Deep Bidirectional Transformers", "authors": ["Jacob Devlin"], "doi": "10.18653/v1/N19-1423", "language": "中文"}'
bash scripts/analyze.sh '{"title": "Language Models are Few-Shot Learners", "authors": ["Tom Brown"], "doi": "10.48550/arXiv.2005.14165", "language": "中文"}'
```

### 第四步：生成对比分析报告

基于各论文的分析结果，进行多维度对比。

### 第五步（可选）：Python生成对比表格

可通过Python脚本生成格式化的对比表格：

```python
#!/usr/bin/env python3
"""论文对比表格生成"""

def generate_comparison_table(papers_info):
    """生成Markdown对比表格"""
    dimensions = ['研究目标', '核心方法', '数据集', '主要结果', '创新点', '局限性']

    # 表头
    header = "| 对比维度 | " + " | ".join(p['short_name'] for p in papers_info) + " |"
    separator = "|" + "---|" * (len(papers_info) + 1)

    print(header)
    print(separator)

    for dim in dimensions:
        row = f"| **{dim}** | "
        row += " | ".join(p.get(dim, '-') for p in papers_info)
        row += " |"
        print(row)

# 使用示例
papers = [
    {"short_name": "BERT", "研究目标": "双向预训练", "核心方法": "MLM+NSP", ...},
    {"short_name": "GPT-3", "研究目标": "少样本学习", "核心方法": "自回归+规模化", ...},
]
generate_comparison_table(papers)
```

## 输出格式

```
# 论文对比分析报告

## 1. 对比论文概览

| 编号 | 论文标题 | 作者 | 年份 | 发表来源 |
|------|----------|------|------|----------|
| [1]  | ...      | ...  | ...  | ...      |
| [2]  | ...      | ...  | ...  | ...      |
| [3]  | ...      | ...  | ...  | ...      |

## 2. 多维度对比分析

### 2.1 研究目标对比
[各论文研究目标的异同分析]

### 2.2 方法论对比
[各论文技术方法的详细对比]

### 2.3 实验设计对比
[数据集、评估指标、基线方法的对比]

### 2.4 核心结果对比
[各论文主要实验结果的对比]

### 2.5 创新点对比
[各论文主要贡献和创新点的对比]

## 3. 对比总结表

| 对比维度 | 论文A | 论文B | 论文C |
|----------|-------|-------|-------|
| 研究目标 | ...   | ...   | ...   |
| 核心方法 | ...   | ...   | ...   |
| 数据集   | ...   | ...   | ...   |
| 主要结果 | ...   | ...   | ...   |
| 创新点   | ...   | ...   | ...   |
| 局限性   | ...   | ...   | ...   |

## 4. 综合评述
[对比分析的总结性评价，各论文的相对优劣势，适用场景建议]

## 参考文献
[对比涉及的所有论文完整引用]
```

## 对比维度说明

| 维度 | 分析内容 |
|------|----------|
| 研究目标 | 要解决的核心问题和研究动机 |
| 方法论 | 技术方法、模型架构、算法设计 |
| 实验设计 | 数据集、评估指标、对比基线 |
| 核心结果 | 主要实验结果、性能指标 |
| 创新点 | 相比已有工作的主要贡献 |
| 局限性 | 方法的不足和适用范围限制 |
| 计算资源 | 训练/推理所需的计算资源（如适用） |
| 可复现性 | 是否开源、数据是否公开 |

## 示例工作流

```
用户: 帮我对比分析 BERT、GPT-3 和 LLaMA 这三篇论文

步骤1: 确认论文 → BERT, GPT-3, LLaMA

步骤2: 分别检索三篇论文的完整信息
- Google Scholar 检索各论文元数据

步骤3: 分别调用 analyze.sh 获取深度分析

步骤4: 多维度对比分析

步骤5: 生成结构化对比报告和总结表
```
