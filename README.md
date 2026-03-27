# 井中探测数据处理与反演子系统 v0.3

## 这是什么

一个用于**井中地球物理数据处理**的桌面软件平台。

你可以用它来：

- 导入测井、瞬变电磁、电阻率等多种格式的数据
- 一键运行各种预处理、正演、反演算法
- 查看二维曲线图和三维模型
- 将多个算法串成自动流水线

算法研究员只需提交两个文件（配置文件 + 算法代码），系统就能自动识别并在界面中调用。

---

## 支持的功能

| 功能            | 说明                                                                   |
| --------------- | ---------------------------------------------------------------------- |
| 5 种勘探方法    | 电阻率法(ERT)、激发极化法(IP)、电磁法(EM)、瞬变电磁法(TEM)、金属矿测井 |
| 8 个已实现算法  | 测井滤波(3个) + TEM正反演(3个) + ERT反演 + EM正演                      |
| 27 个待填充插槽 | 算法研究员可按标准接口提交，无需修改系统代码                           |
| 多格式数据导入  | LAS 测井 / SEG-Y 地震 / CSV 文本 / VTK 网格 / 自定义二进制             |
| 流水线          | 多步骤自动串联处理（JSON 配置）                                        |
| 三维可视化      | PyVista 体渲染、等值面、正交剖面、钻孔轨迹                             |
| 算法代码保护    | 可编译为 .pyd(Windows) / .so(Linux/Mac)，不暴露源码                    |
| Windows 友好    | 中文界面、UTF-8 编码、一键验证脚本                                     |

---

## 已实现的算法

| 方法 | 算法                  | 功能                     | 来源               |
| ---- | --------------------- | ------------------------ | ------------------ |
| 测井 | `log_preproc_savgol`  | Savitzky-Golay 平滑滤波  | scipy.signal       |
| 测井 | `log_preproc_wavelet` | 小波阈值去噪             | PyWavelets         |
| 测井 | `log_preproc_kalman`  | Kalman-RTS 双向平滑      | numpy/scipy        |
| TEM  | `empymod_tem_forward` | 一维正演（水平层状介质） | empymod            |
| TEM  | `simpeg_tem_forward`  | 三维圆柱网格正演         | SimPEG             |
| TEM  | `tem_1d_inversion`    | 一维 Occam 反演          | scipy.optimize     |
| ERT  | `simpeg_dc_inversion` | 二维电阻率反演           | SimPEG             |
| EM   | `em_forward_3d`       | 三维井间电磁正演         | numpy + discretize |

---

## 安装方法

### 第一步：安装 Python

需要 Python 3.11 或更高版本。推荐从 [python.org](https://www.python.org/downloads/) 下载安装。

> Windows 用户：安装时勾选「Add Python to PATH」。

### 第二步：下载项目

```
git clone <仓库地址>
cd geophys-tool
```

### 第三步：安装依赖

```
pip install uv
uv sync
```

如需三维可视化功能，额外安装：

```
uv sync --extra 3d
```

---

## 使用方法

### 启动软件

```
python main.py
```

软件界面分为四个区域：

```
┌──────────┬───────────────────────┬──────────────┐
│          │                       │              │
│ 项目树    │    结果显示区          │  算法面板     │
│（左侧）   │  （中央：2D/3D）      │ （右侧）     │
│          │                       │              │
├──────────┴───────────────────────┴──────────────┤
│                    执行日志                       │
└─────────────────────────────────────────────────┘
```

### 基本操作流程

1. **新建钻孔** — 点击工具栏「+ 新建钻孔」，输入名称
2. **导入数据** — 点击「导入数据」，选择 LAS/CSV/SEG-Y 文件，选择方法类型
3. **选择算法** — 在右侧面板展开算法树，点击想要运行的算法
4. **设置参数** — 调整参数（可用默认值），点击「运行算法」
5. **查看结果** — 中央画布自动显示结果曲线或三维模型

### 运行流水线

流水线可以自动串联多个算法。在工具栏点击「运行流水线」，选择预定义的流水线即可。

内置流水线：

| 流水线             | 步骤                             |
| ------------------ | -------------------------------- |
| 测井三算法滤波对比 | SG 滤波 → 小波去噪 → Kalman 平滑 |
| TEM 一维工作流     | empymod 正演 → Occam 反演        |

---

## 给算法研究员

如果你想把自己的算法集成到这个平台，只需要做三件事：

1. 在 `algorithms/` 下新建一个文件夹
2. 写一个 `manifest.json`（告诉系统你的算法需要什么输入、产出什么输出）
3. 写一个 `algorithm.py`（你的算法代码）

详细步骤请看 **[算法开发者指南](docs/developer_guide.md)**。

快速验证你的算法包：

```
python tools\validate_algo.py algorithms\你的算法名
```

Windows 用户也可以双击 `tools\verify.bat`。

---

## 开发工具

| 工具                     | 用途               | 用法                                                                                                                   |
| ------------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `tools/scaffold_algo.py` | 自动生成算法模板   | `python tools/scaffold_algo.py --name xxx --display-name xxx --method tem --category forward --template 1d-preprocess` |
| `tools/validate_algo.py` | 验证算法包是否正确 | `python tools/validate_algo.py algorithms/xxx` 或 `--all`                                                              |
| `tools/compile_algo.py`  | 编译算法为动态库   | `python tools/compile_algo.py algorithms/xxx`                                                                          |
| `tools/verify.bat`       | Windows 一键验证   | 双击或拖拽算法文件夹                                                                                                   |

---

## 项目目录结构

```
geophys-tool/
├── main.py                 # 软件入口
├── pyproject.toml          # Python 依赖配置
│
├── src/                    # 系统源码（一般不需要修改）
│   ├── core/               #   数据管理、文件导入、网格工具
│   ├── algorithms/         #   算法调度引擎（注册、运行、流水线）
│   └── ui/                 #   界面（主窗口、2D画布、3D可视化）
│
├── algorithms/             # 算法模块目录（35个）
│   └── 算法名/
│       ├── manifest.json   #   算法配置（输入输出定义）
│       └── algorithm.py    #   算法代码
│
├── pipelines/              # 流水线配置（JSON）
├── tools/                  # 开发工具（脚手架、验证、编译）
├── tests/                  # 测试和演示脚本
│   └── data/               #   测试数据和输出图片
└── docs/                   # 文档
    ├── developer_guide.md  #   算法开发者指南
    └── user_guide.md       #   用户使用手册
```

---

## 测试验证

系统已通过 37 项自动化测试，覆盖全部 8 个已实现算法：

```
python tests/test_e2e_workflow.py              # 端到端全流程（11项）
python tests/test_third_party_integration.py   # 第三方算法集成（11项）
python tests/test_algo_variety.py              # 6种算法封装模式
python tests/test_opensource_and_compiled.py    # 开源库封装 + 动态库编译（9项）
```

演示脚本：

```
python tests/demo_3d_inversion_compiled.py     # 三维反演 + Cython编译全流程
python tests/demo_full_pipeline.py             # 综合9子图演示
python tests/build_geo_model_3d.py             # 精细地质模型构建
```

---

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                  PyQt6 图形界面                           │
│   项目树 │ 2D/3D 可视化画布 │ 算法面板 │ 执行日志        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│           AlgorithmRunner / PipelineRunner               │
│     （自动加载输入 → 调用算法 → 保存输出 → 验证结果）      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              算法插件层  algorithms/                       │
│    每个算法 = manifest.json（配置）+ algorithm.py（代码）   │
│    支持 Python 源码 / Cython 编译产物(.pyd/.so) / C 动态库 │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│           工作区  ~/geophys-workspace/                    │
│     boreholes/钻孔名/方法/raw/（输入）                    │
│                          processed/（输出）               │
└─────────────────────────────────────────────────────────┘
```
