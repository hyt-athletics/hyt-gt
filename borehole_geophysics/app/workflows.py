"""
工作流

把各个模块串成完整的操作流程
每个工作流 = 一个菜单选项
"""

import os
import sys
import numpy as np


def workflow_interactive_modeling(project):
    """工作流1：交互式建模"""
    from interactive.drawing_tool_v2 import InteractiveDrawingToolV2
    
    def on_compute():
        project.generate_mesh()
        result = project.compute_gravity()
        from visualization.plot_gravity import plot_gravity_with_model
        plot_gravity_with_model(result, bodies=project.model_manager.bodies)
    
    def on_export():
        if project.mesh is None:
            project.generate_mesh()
        from export.vtk_export import VTKExporter
        output_dir = project.settings.get('export.vtk_output_dir', 'vtk_output')
        exporter = VTKExporter(output_dir=output_dir)
        exporter.export_all(
            octree_mesh=project.mesh,
            well=project.well,
            bodies=project.model_manager.bodies,
            gravity_result=project.gravity_result,
        )
    
    tool = InteractiveDrawingToolV2(
        model_manager=project.model_manager,
        on_compute=on_compute,
        on_export=on_export,
    )
    tool.run()


def workflow_gravity_forward(project):
    """工作流2：重力正演"""
    n_bodies = len(project.model_manager.bodies)
    n_layers = len(getattr(project.model_manager, 'layers', []))
    
    if n_bodies == 0 and n_layers == 0:
        print("\n  ⚠️ 没有地质体！请先通过[1]交互式建模创建模型")
        input("  按Enter返回...")
        return
    
    print(f"\n  当前模型: {n_bodies} 个地质体, {n_layers} 条地层线")
    
    # 生成网格
    project.generate_mesh()
    
    # 正演计算
    result = project.compute_gravity()
    
    # 显示结果
    from visualization.plot_gravity import plot_gravity_with_model
    plot_gravity_with_model(result, bodies=project.model_manager.bodies)
    
    # 保存结果
    ans = input("\n  保存结果到CSV？(y/n) [y]: ").strip().lower()
    if ans != 'n':
        result.save_csv('gravity_result.csv')


def workflow_mesh_tools(project):
    """工作流3：网格工具"""
    while True:
        print("""
    ┌─ 网格工具 ─────────────────────────────┐
    │                                         │
    │  [1] 八叉树网格演示                      │
    │  [2] 2D Delaunay三角剖分演示             │
    │  [3] 2.5D三棱柱网格演示                  │
    │  [4] 3D四面体网格演示                    │
    │  [5] 电法专用网格演示                    │
    │  [6] 地震专用网格演示                    │
    │  [0] 返回主菜单                          │
    │                                         │
    └──────────────────────────────────────────┘
        """)
        
        choice = input("  请选择 [0-6]: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            _run_demo_script('run_step2')
        elif choice == '2':
            _run_demo_script('run_step3')
        elif choice == '3':
            _run_demo_script('run_step5')
        elif choice == '4':
            _run_demo_script('run_step7')
        elif choice == '5':
            _run_demo_script('run_step13', func='demo_electrical')
        elif choice == '6':
            _run_demo_script('run_step13', func='demo_seismic')
        else:
            print("  无效选择")


def workflow_import_export(project):
    """工作流4：导入/导出"""
    while True:
        print("""
    ┌─ 导入/导出 ────────────────────────────┐
    │                                         │
    │  [1] 加载模型 (.json)                   │
    │  [2] 保存模型 (.json)                   │
    │  [3] 导出VTK文件 (ParaView用)           │
    │  [4] 导出正演结果 (.csv)                │
    │  [5] 保存整个项目 (文件夹)               │
    │  [6] 加载整个项目                        │
    │  [0] 返回主菜单                          │
    │                                         │
    └──────────────────────────────────────────┘
        """)
        
        choice = input("  请选择 [0-6]: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            fn = _ask_filename("加载模型文件", "model.json", mode='open')
            if fn:
                project.model_manager.load(fn)
        elif choice == '2':
            fn = _ask_filename("保存模型文件", "model.json", mode='save')
            if fn:
                project.model_manager.save(fn)
        elif choice == '3':
            _export_vtk(project)
        elif choice == '4':
            if project.gravity_result:
                fn = _ask_filename("保存CSV", "gravity_result.csv", mode='save')
                if fn:
                    project.gravity_result.save_csv(fn)
            else:
                print("  ⚠️ 还没有正演结果，请先计算")
        elif choice == '5':
            name = input("  项目名称 [my_project]: ").strip() or "my_project"
            project.name = name
            project.save()
        elif choice == '6':
            dirname = input("  项目文件夹路径: ").strip()
            if dirname:
                project.load(dirname)
        else:
            print("  无效选择")


def workflow_benchmark(project):
    """工作流5：性能测试"""
    from acceleration.benchmark import run_gravity_benchmark, run_geometry_benchmark
    
    print("\n  选择测试项目：")
    print("    [1] 重力正演基准测试")
    print("    [2] 几何计算基准测试")
    print("    [3] 全部测试")
    
    choice = input("  请选择 [1-3]: ").strip()
    
    if choice == '1':
        run_gravity_benchmark()
    elif choice == '2':
        run_geometry_benchmark()
    elif choice == '3':
        run_gravity_benchmark()
        run_geometry_benchmark()
    
    input("\n  按Enter返回...")


def workflow_settings(project):
    """工作流6：设置"""
    while True:
        s = project.settings
        
        print(f"""
    ┌─ 设置 ──────────────────────────────────────────┐
    │                                                  │
    │  [1] 模型域                                      │
    │      X: {s.get('domain.x_range')}  Y: {s.get('domain.y_range')}  Z: {s.get('domain.z_range')}
    │                                                  │
    │  [2] 钻孔                                        │
    │      位置: ({s.get('well.x')}, {s.get('well.y')})  深度: {s.get('well.z_top')}~{s.get('well.z_bottom')}m
    │      观测点: {s.get('well.n_stations')} 个
    │                                                  │
    │  [3] 网格                                        │
    │      类型: {s.get('mesh.type')}  最大层级: {s.get('mesh.max_level')}
    │                                                  │
    │  [4] 正演引擎                                    │
    │      当前: {s.get('forward.engine')}
    │                                                  │
    │  [5] 查看所有设置                                │
    │  [6] 恢复默认设置                                │
    │  [7] 保存设置                                    │
    │  [0] 返回主菜单                                  │
    │                                                  │
    └──────────────────────────────────────────────────┘
        """)
        
        choice = input("  请选择 [0-7]: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            _edit_domain(s)
            project.update_settings()
        elif choice == '2':
            _edit_well(s)
            project.update_settings()
        elif choice == '3':
            _edit_mesh(s)
        elif choice == '4':
            _edit_engine(s)
        elif choice == '5':
            s.print_all()
        elif choice == '6':
            s._load_defaults()
            project.update_settings()
            print("  ✅ 已恢复默认设置")
        elif choice == '7':
            s.save()


def workflow_help():
    """工作流7：帮助"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    使用帮助                                       ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  典型工作流程:                                                     ║
    ║  ───────────                                                      ║
    ║    1. 选择 [1] 交互式建模                                          ║
    ║       → 窗口弹出后按 P 画多边形矿体                                ║
    ║       → 右键完成 → 输入名称和密度                                  ║
    ║       → 按 C 计算重力 → 查看曲线                                   ║
    ║       → 关闭窗口                                                   ║
    ║                                                                   ║
    ║    2. 选择 [2] 重力正演                                            ║
    ║       → 自动生成网格并计算                                          ║
    ║       → 查看重力曲线和模型截面                                      ║
    ║                                                                   ║
    ║    3. 选择 [4] 导出                                                ║
    ║       → 导出VTK文件 → 用ParaView打开                               ║
    ║                                                                   ║
    ║  交互建模快捷键:                                                   ║
    ║  ─────────────                                                    ║
    ║    P=多边形  E=椭圆  R=矩形  L=地层线                              ║
    ║    S=选择  M=移动  T=顶点编辑  双击=改物性                          ║
    ║    Ctrl+Z=撤销  Ctrl+Y=重做  Ctrl+C/V=复制粘贴                     ║
    ║    C=正演  V=3D查看  X=VTK导出  G=网格吸附                         ║
    ║                                                                   ║
    ║  已支持的网格类型:                                                  ║
    ║  ───────────────                                                  ║
    ║    • 结构化六面体    • 八叉树自适应                                  ║
    ║    • 2D Delaunay    • 约束Delaunay                                ║
    ║    • 2.5D三棱柱     • 3D四面体                                    ║
    ║    • 电法专用网格    • 地震专用网格                                  ║
    ║                                                                   ║
    ║  正演引擎:                                                        ║
    ║  ─────────                                                        ║
    ║    • builtin: 纯Python（慢但可靠）                                 ║
    ║    • numpy:   NumPy向量化（快10~50倍）                             ║
    ║    • numba:   Numba JIT（快100~400倍，需pip install numba）        ║
    ║    • harmonica: Fatiando库（需pip install harmonica）              ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    input("  按Enter返回...")


# =============================================================
#  设置编辑辅助函数
# =============================================================

def _edit_domain(s):
    print("\n  编辑模型域范围（直接回车保持不变）:")
    for axis in ['x_range', 'y_range', 'z_range']:
        current = s.get(f'domain.{axis}')
        val = input(f"    {axis} [{current[0]}, {current[1]}]: ").strip()
        if val:
            try:
                parts = val.replace(',', ' ').replace('[', '').replace(']', '').split()
                s.set(f'domain.{axis}', [float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                print(f"    ⚠️ 格式错误，保持原值")


def _edit_well(s):
    print("\n  编辑钻孔参数（直接回车保持不变）:")
    for key in ['x', 'y', 'z_top', 'z_bottom', 'n_stations']:
        current = s.get(f'well.{key}')
        val = input(f"    {key} [{current}]: ").strip()
        if val:
            try:
                s.set(f'well.{key}', float(val) if '.' in val else int(val))
            except ValueError:
                print(f"    ⚠️ 格式错误")


def _edit_mesh(s):
    print("\n  编辑网格参数:")
    val = input(f"    最大层级 [{s.get('mesh.max_level')}]: ").strip()
    if val:
        try:
            s.set('mesh.max_level', int(val))
        except ValueError:
            pass


def _edit_engine(s):
    current = s.get('forward.engine')
    print(f"\n  当前引擎: {current}")
    print("  可选: auto / builtin / numpy / numba / harmonica")
    val = input("  选择引擎: ").strip()
    if val in ('auto', 'builtin', 'numpy', 'numba', 'harmonica'):
        s.set('forward.engine', val)
        print(f"  ✅ 引擎已切换为: {val}")


def _ask_filename(title, default, mode='open'):
    """请求文件名"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if mode == 'open':
            fn = filedialog.askopenfilename(title=title, filetypes=[('JSON', '*.json'), ('All', '*.*')])
        else:
            fn = filedialog.asksaveasfilename(title=title, defaultextension='.json',
                                              initialfile=default)
        root.destroy()
        return fn if fn else None
    except Exception:
        fn = input(f"  文件名 [{default}]: ").strip() or default
        return fn


def _export_vtk(project):
    """VTK导出"""
    if project.mesh is None:
        print("  先生成网格...")
        project.generate_mesh()
    
    from export.vtk_export import VTKExporter
    output_dir = project.settings.get('export.vtk_output_dir', 'vtk_output')
    exporter = VTKExporter(output_dir=output_dir)
    exporter.export_all(
        octree_mesh=project.mesh,
        well=project.well,
        bodies=project.model_manager.bodies,
        gravity_result=project.gravity_result,
    )


def _run_demo_script(module_name, func=None):
    """运行演示脚本"""
    try:
        mod = __import__(module_name)
        if func:
            getattr(mod, func)()
        else:
            mod.main()
    except ImportError as e:
        print(f"  ⚠️ 无法加载 {module_name}: {e}")
    except Exception as e:
        print(f"  ⚠️ 运行出错: {e}")
        import traceback
        traceback.print_exc()