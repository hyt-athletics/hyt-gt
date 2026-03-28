
# Borehole Geophysics Platform (BHGPS)

钻孔地球物理交互建模与正演平台 v1.0

## 快速开始

```bash
# 安装依赖
pip install numpy matplotlib scipy pyvista

# 可选（推荐，性能提升100-400倍）
pip install numba

# 可选（成熟正演库）
pip install harmonica

# 启动程序
python main.py
```

## 功能模块

| 模块 | 说明 |
|------|------|
| 交互式建模 | 鼠标画矿体（多边形/椭圆/矩形/地层线），拖拽移动，顶点编辑，撤销重做 |
| 网格剖分 | 结构化网格、八叉树自适应、2D Delaunay、约束Delaunay、边翻转、2.5D三棱柱、3D四面体 |
| 重力正演 | 棱柱解析公式(Nagy 1966)，自研+NumPy向量化+Numba JIT+harmonica |
| 网格适配 | 井地电法专用网格（电极奇异性）、VSP地震专用网格（波长约束） |
| 可视化 | matplotlib 2D、PyVista 3D、VTK导出（ParaView） |

## 操作流程

1. `python main.py` → 选择 [1] 交互式建模
2. 按 P → 鼠标点击画矿体 → 右键完成 → 输入密度
3. 按 C → 自动网格+正演+显示曲线
4. 按 V → 3D查看
5. Ctrl+S → 保存模型

## 交互快捷键

| 键 | 功能 | 键 | 功能 |
|----|------|----|------|
| P | 画多边形 | Ctrl+Z | 撤销 |
| E | 画椭圆 | Ctrl+Y | 重做 |
| R | 画矩形 | Ctrl+C | 复制 |
| L | 画地层线 | Ctrl+V | 粘贴 |
| S | 选择 | Ctrl+S | 保存 |
| M | 移动 | Delete | 删除 |
| T | 顶点编辑 | Esc | 取消 |
| 双击 | 编辑物性 | H | 帮助 |
| C | 正演计算 | V | 3D查看 |
| X | VTK导出 | G | 网格吸附 |

## 项目结构

```
borehole_geophysics/
├── main.py              # 主入口
├── app/                 # 应用层（菜单/项目/工作流）
├── mesh/                # 网格剖分模块
├── forward/             # 正演计算模块
├── interactive/         # 交互建模模块
├── visualization/       # 可视化模块
├── geometry/            # 几何体定义
├── borehole/            # 钻孔定义
├── export/              # VTK导出
├── acceleration/        # 性能加速
├── config/              # 观测系统配置
└── demos/              # 各模块演示脚本
```

## 演示脚本

各模块的独立演示脚本，位于 `demos/` 文件夹中：

| 脚本 | 说明 |
|------|------|
| run_step1.py | 结构化网格 + 可视化 |
| run_step2.py | 八叉树自适应网格 |
| run_step3.py | 2D Delaunay三角剖分 |
| run_step4.py | 约束Delaunay三角剖分 |
| run_step5.py | 2.5D三棱柱网格 |
| run_step6.py | 3D四面体网格 |
| run_step7.py | 重力正演（基础） |
| run_step8.py | 重力正演（NumPy加速） |
| run_step9.py | 重力正演（Numba加速） |
| run_step10.py | 井地电法网格 |
| run_step11.py | VSP地震网格 |
| run_step12.py | 网格质量评估 |
| run_step13.py | 网格优化 |
| run_step14.py | 性能基准测试 |

运行演示脚本：
```bash
python borehole_geophysics/demos/run_step1.py
```
