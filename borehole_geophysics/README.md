
# Borehole Geophysics Platform (BHGPS)

钻孔地球物理交互建模与多方法正反演平台 v1.3

## 快速开始

```bash
# 安装依赖
pip install numpy matplotlib scipy pyvista

# 可选（推荐，性能提升100-400倍）
pip install numba

# 可选（成熟正演库，3C磁测需要）
pip install harmonica

# 可选（外部反演库，重力/磁法反演 MVP 需要）
pip install simpeg discretize

# 启动程序
python main.py
```

## 功能模块

| 模块 | 说明 |
|------|------|
| 交互式建模 | 鼠标画矿体（多边形/椭圆/矩形/地层线），拖拽移动，顶点编辑，撤销重做，多物性录入 |
| 网格剖分 | 结构化网格、八叉树自适应、2D Delaunay、约束Delaunay、边翻转、2.5D三棱柱、3D四面体 |
| 重力正演 | 统一适配器入口，支持 builtin / NumPy / Numba / harmonica |
| 3C井中磁测 | 统一适配器入口，调用 harmonica 计算 Bx/By/Bz/ΔT |
| 重力反演 | 调用 SimPEG，支持 active volume、bounds、update region、smooth/compact 正则模板 |
| 3C磁测反演 MVP | 调用 SimPEG，支持磁化率反演、smooth/compact 正则模板，当前仅限标量磁化率 + 感应磁化 |
| 网格适配 | 井地电法专用网格（电极奇异性）、VSP地震专用网格（波长约束） |
| 结果管理与导出 | 正演/反演历史管理、CSV/NPZ/VTK 导出、ParaView 主文件打包 |
| 可视化 | matplotlib 2D、PyVista 3D、VTK导出（ParaView） |

## 操作流程

1. `python main.py` → 选择 [1] 交互式建模
2. 按 P → 鼠标点击画矿体 → 右键完成 → 输入密度/磁化率等物性
3. 在 [6] 设置中选择方法 `gravity` 或 `magnetic_3c`
4. 按 C → 自动网格+正演+显示曲线
5. 在 [4] 导入/导出 导入实测曲线，或直接用当前正演结果做 smoke test
6. 在 [2] 正演计算 → [6] 外部反演计算，运行当前方法的外部反演
7. 按 V → 3D查看
8. Ctrl+S → 保存模型

## 当前反演边界

- 重力反演：`SimPEG` 密度反演工作流，支持 `active volume / bounds / update region / smooth / compact / property reference`
- 3C 磁测反演：`SimPEG` 磁化率反演 MVP，支持 `smooth / compact` 正则，当前仅支持标量磁化率与感应磁化
- 暂不支持剩磁反演、矢量磁化率模型、复杂商业软件私有求解器复刻

## 当前实现重点

- 主线方法已经明确为井中重磁位场工作流，不把主求解链切到有限差分或 CNN/PINN。
- 正演继续优先使用积分核与外部库，反演继续优先使用 `SimPEG` 的高斯牛顿系工作流。
- 网格策略已经区分 `forward / inverse` 两种用途，支持 `ROI / active volume / near-well refinement / mesh QC`。

## 下一阶段

1. 把 `property reference` 扩展成更接近商业软件的 `petrophysical prior / hard-soft constraint voxel`。
2. 把磁法的剩磁参数接进约束、可视化和结果解释层，即使第一阶段求解器仍以感应磁化为主。
3. 继续强化 survey-aware mesh template 和井中重磁的 QC 面板。

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
├── adapters/            # 方法适配器层
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
| run_step15.py | 3C井中磁测演示 |

运行演示脚本：
```bash
python borehole_geophysics/demos/run_step1.py
```
