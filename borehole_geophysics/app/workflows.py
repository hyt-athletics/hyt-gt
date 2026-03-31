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
        method_code = project.get_current_method()
        project.generate_mesh_for_method(method_code)
        result = project.run_forward(method_code)
        _show_forward_result_with_model(result, project)
    
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
            magnetic_result=project.magnetic_result,
            gravity_inverse_result=project.inverse_result if project.inverse_result_method == 'gravity' else None,
            magnetic_inverse_result=project.inverse_result if project.inverse_result_method == 'magnetic_3c' else None,
        )
    
    tool = InteractiveDrawingToolV2(
        model_manager=project.model_manager,
        on_compute=on_compute,
        on_export=on_export,
    )
    tool.run()


def workflow_forward_menu(project):
    """工作流2：正演方法菜单"""
    while True:
        print("""
    ┌─ 正演计算 ─────────────────────────────┐
    │                                         │
    │  [1] 井中重力正演                        │
    │  [2] 3C井中磁测正演                      │
    │  [3] 对比当前方法历史结果                │
    │  [4] 结果管理                            │
    │  [5] 当前方法 正演/实测 对比             │
    │  [6] 外部反演计算                        │
    │  [7] 反演结果管理                        │
    │  [0] 返回主菜单                          │
    │                                         │
    └──────────────────────────────────────────┘
        """)
        
        choice = input("  请选择 [0-7]: ").strip()
        if choice == '0':
            break
        if choice == '1':
            workflow_gravity_forward(project)
            input("\n  按Enter返回正演菜单...")
            continue
        if choice == '2':
            workflow_magnetic_forward(project)
            input("\n  按Enter返回正演菜单...")
            continue
        if choice == '3':
            workflow_compare_results(project)
            input("\n  按Enter返回正演菜单...")
            continue
        if choice == '4':
            workflow_result_manager(project)
            input("\n  按Enter返回正演菜单...")
            continue
        if choice == '5':
            workflow_compare_with_observed(project)
            input("\n  按Enter返回正演菜单...")
            continue
        if choice == '6':
            workflow_inverse_shell(project)
            input("\n  按Enter返回正演菜单...")
            continue
        if choice == '7':
            workflow_inverse_result_manager(project)
            input("\n  按Enter返回正演菜单...")
            continue
        print("  无效选择")


def workflow_gravity_forward(project):
    """工作流2-1：重力正演"""
    n_bodies = len(project.model_manager.bodies)
    n_layers = len(getattr(project.model_manager, 'layers', []))
    
    if n_bodies == 0 and n_layers == 0:
        print("\n  ⚠️ 没有地质体！请先通过[1]交互式建模创建模型")
        input("  按Enter返回...")
        return
    
    print(f"\n  当前模型: {n_bodies} 个地质体, {n_layers} 条地层线")
    
    # 生成网格
    project.settings.set('forward.method', 'gravity')
    project.generate_mesh_for_method('gravity')
    
    # 正演计算
    result = project.run_forward('gravity')
    
    # 显示结果
    _show_forward_result_with_model(result, project)
    
    # 保存结果
    ans = input("\n  保存结果到CSV？(y/n) [y]: ").strip().lower()
    if ans != 'n':
        result.save_csv('gravity_result.csv')


def workflow_magnetic_forward(project):
    """工作流2-2：3C井中磁测正演。"""
    n_bodies = len(project.model_manager.bodies)
    n_layers = len(getattr(project.model_manager, 'layers', []))
    
    if n_bodies == 0 and n_layers == 0:
        print("\n  ⚠️ 没有地质体！请先通过[1]交互式建模创建模型")
        return
    
    print(f"\n  当前模型: {n_bodies} 个地质体, {n_layers} 条地层线")
    project.settings.set('forward.method', 'magnetic_3c')
    
    try:
        project.generate_mesh_for_method('magnetic_3c')
        result = project.run_forward('magnetic_3c')
        _show_forward_result_with_model(result, project)
        ans = input("\n  保存结果到CSV？(y/n) [y]: ").strip().lower()
        if ans != 'n':
            result.save_csv('magnetic_result.csv')
    except Exception as exc:
        print(f"\n  ❌ 磁法工作流失败: {exc}")


def workflow_compare_results(project):
    """对比当前方法的历史结果。"""
    method_code = project.get_current_method()
    results = project.get_results_for_method(method_code, include_current=True)
    if len(results) < 2:
        print(f"\n  ⚠️ 方法 {method_code} 至少需要两次结果才能对比")
        return
    
    print(f"\n  对比方法: {method_code}")
    print(f"  可用结果数: {len(results)}")
    for result in results:
        print(f"    - {getattr(result, 'label', '未命名结果')}")
    
    if method_code == 'gravity':
        from visualization.plot_gravity import plot_gravity_comparison
        plot_gravity_comparison(results)
        return
    
    if method_code == 'magnetic_3c':
        from visualization.plot_magnetic import plot_magnetic_comparison
        plot_magnetic_comparison(results)
        return
    
    print("  ⚠️ 当前方法尚未接入结果对比视图")


def workflow_result_manager(project):
    """项目级结果管理。"""
    filter_mode = 'all'
    while True:
        catalog = _get_filtered_result_catalog(project, filter_mode)
        if not catalog:
            print("\n  ⚠️ 当前筛选条件下没有结果")
            raw = input("  输入 f 切换筛选，直接回车返回: ").strip().lower()
            if raw == 'f':
                filter_mode = _ask_result_filter_mode(project, filter_mode)
                continue
            return

        print(f"\n  结果目录 [{_describe_result_filter(project, filter_mode)}]:")
        for item in catalog:
            flags = []
            if item['is_active']:
                flags.append('ACTIVE')
            if item['is_latest_for_method']:
                flags.append('LATEST')
            flag_text = f" [{'|'.join(flags)}]" if flags else ""
            print(f"    [{item['method_code']}:{item['index']}] {item['label']}{flag_text}")

        raw = input(
            "\n  选择结果 (method:index)，输入 f 切换筛选，直接回车返回: "
        ).strip()
        if not raw:
            return
        if raw.lower() == 'f':
            filter_mode = _ask_result_filter_mode(project, filter_mode)
            continue

        try:
            method_code, index_text = raw.split(':', 1)
            index = int(index_text)
        except ValueError:
            print("  ⚠️ 输入格式错误")
            continue

        results = project.get_results_for_method(method_code, include_current=True)
        if not (0 <= index < len(results)):
            print("  ⚠️ 结果不存在")
            continue
        result = results[index]

        while True:
            print(f"\n  已选择: {getattr(result, 'label', '未命名结果')}")
            print("  [1] 查看摘要")
            print("  [2] 设为当前结果")
            print("  [3] 绘图")
            print("  [4] 导出CSV")
            print("  [5] 导出NPZ")
            print("  [6] 导出VTK")
            print("  [7] 删除结果")
            print("  [0] 返回结果目录")

            action = input("  请选择操作 [0-7]: ").strip()
            if action == '0':
                break
            if action == '1':
                if hasattr(result, 'get_summary'):
                    result.get_summary()
                continue
            if action == '2':
                project.set_current_result_by_index(method_code, index)
                print(f"  ✅ 已切换当前结果为: {getattr(result, 'label', '未命名结果')}")
                break
            if action == '3':
                _show_forward_result_with_model(result, project)
                continue
            if action == '4':
                default_name = f"{_result_filename_stub(result, method_code)}.csv"
                fn = _ask_filename("保存结果CSV", default_name, mode='save')
                if fn and hasattr(result, 'save_csv'):
                    result.save_csv(fn)
                continue
            if action == '5':
                default_name = f"{_result_filename_stub(result, method_code)}.npz"
                fn = _ask_filename("保存结果NPZ", default_name, mode='save')
                if fn and hasattr(result, 'save_npz'):
                    result.save_npz(fn)
                continue
            if action == '6':
                default_dir = _result_filename_stub(result, method_code)
                output_dir = input(f"  导出目录 [{default_dir}]: ").strip() or default_dir
                _export_selected_result_vtk(project, method_code, result, output_dir)
                continue
            if action == '7':
                removed = project.delete_result_by_index(method_code, index)
                print(f"  ✅ 已删除结果: {getattr(removed, 'label', '未命名结果')}")
                break
            print("  ⚠️ 无效操作")


def workflow_compare_with_observed(project):
    """对比当前方法的正演结果和导入实测数据。"""
    method_code = project.get_current_method()
    modeled = project.current_result
    observed = project.get_observed_result(method_code)

    if modeled is None or getattr(modeled, 'method_code', method_code) != method_code:
        print(f"\n  ⚠️ 当前方法 {method_code} 还没有正演结果")
        return
    if observed is None:
        print(f"\n  ⚠️ 当前方法 {method_code} 还没有导入实测数据")
        return

    if method_code == 'gravity':
        from visualization.plot_gravity import plot_gravity_model_vs_observed
        plot_gravity_model_vs_observed(modeled, observed)
        return

    if method_code == 'magnetic_3c':
        from visualization.plot_magnetic import plot_magnetic_model_vs_observed
        plot_magnetic_model_vs_observed(modeled, observed)
        return

    print("  ⚠️ 当前方法尚未接入实测对比视图")


def workflow_inverse_shell(project):
    """通过外部库执行当前方法的反演。"""
    method_code = project.get_current_method()
    observed = project.get_observed_result(method_code)
    if observed is None:
        print(f"\n  ⚠️ 当前方法 {method_code} 还没有导入实测数据")
        print("  先到 [4] 导入/导出 → 导入实测曲线")
        return

    print(f"\n  外部反演计算: {method_code}")
    print("  当前阶段不自研反演核，只通过统一适配器调用外部库。")
    print("  已加载的实测数据:")
    print(f"    {getattr(observed, 'label', 'observed')}")
    _print_inverse_setting_summary(project.settings, method_code)

    try:
        result = project.run_inverse(method_code=method_code)
        assumptions = getattr(result, 'metadata', {}).get('assumptions')
        if assumptions:
            print(f"  反演假设: {assumptions}")
        if hasattr(result, 'plot'):
            ans = input("  显示反演结果图？(y/n) [y]: ").strip().lower()
            if ans != 'n':
                result.plot()
        if hasattr(result, 'save_csv'):
            ans = input("  导出反演结果CSV？(y/n) [y]: ").strip().lower()
            if ans != 'n':
                default_name = f"{method_code}_inverse_result.csv"
                fn = _ask_filename("保存反演结果CSV", default_name, mode='save')
                if fn:
                    result.save_csv(fn)
    except Exception as exc:
        print(f"  ❌ 外部反演入口执行失败: {exc}")


def workflow_inverse_result_manager(project):
    """项目级反演结果管理。"""
    filter_mode = 'all'
    while True:
        catalog = _get_filtered_inverse_catalog(project, filter_mode)
        if not catalog:
            print("\n  ⚠️ 当前筛选条件下没有反演结果")
            raw = input("  输入 f 切换筛选，直接回车返回: ").strip().lower()
            if raw == 'f':
                filter_mode = _ask_result_filter_mode(project, filter_mode)
                continue
            return

        print(f"\n  反演结果目录 [{_describe_result_filter(project, filter_mode)}]:")
        for item in catalog:
            flags = []
            if item['is_active']:
                flags.append('ACTIVE')
            if item['is_latest_for_method']:
                flags.append('LATEST')
            flag_text = f" [{'|'.join(flags)}]" if flags else ""
            print(f"    [{item['method_code']}:{item['index']}] {item['label']}{flag_text}")

        raw = input(
            "\n  选择反演结果 (method:index)，输入 f 切换筛选，直接回车返回: "
        ).strip()
        if not raw:
            return
        if raw.lower() == 'f':
            filter_mode = _ask_result_filter_mode(project, filter_mode)
            continue

        try:
            method_code, index_text = raw.split(':', 1)
            index = int(index_text)
        except ValueError:
            print("  ⚠️ 输入格式错误")
            continue

        results = project.get_inverse_results_for_method(method_code, include_current=True)
        if not (0 <= index < len(results)):
            print("  ⚠️ 反演结果不存在")
            continue
        result = results[index]

        while True:
            print(f"\n  已选择: {getattr(result, 'label', '未命名反演结果')}")
            print("  [1] 查看摘要")
            print("  [2] 设为当前反演结果")
            print("  [3] 绘图")
            print("  [4] 导出CSV")
            print("  [5] 导出NPZ")
            print("  [6] 导出VTK")
            print("  [7] 删除结果")
            print("  [0] 返回结果目录")

            action = input("  请选择操作 [0-7]: ").strip()
            if action == '0':
                break
            if action == '1':
                if hasattr(result, 'get_summary'):
                    result.get_summary()
                continue
            if action == '2':
                project.set_current_inverse_result_by_index(method_code, index)
                print(f"  ✅ 已切换当前反演结果为: {getattr(result, 'label', '未命名反演结果')}")
                break
            if action == '3':
                if hasattr(result, 'plot'):
                    result.plot()
                continue
            if action == '4':
                default_name = f"{_result_filename_stub(result, method_code)}.csv"
                fn = _ask_filename("保存反演结果CSV", default_name, mode='save')
                if fn and hasattr(result, 'save_csv'):
                    result.save_csv(fn)
                continue
            if action == '5':
                default_name = f"{_result_filename_stub(result, method_code)}.npz"
                fn = _ask_filename("保存反演结果NPZ", default_name, mode='save')
                if fn and hasattr(result, 'save_npz'):
                    result.save_npz(fn)
                continue
            if action == '6':
                default_dir = _result_filename_stub(result, method_code)
                output_dir = input(f"  导出目录 [{default_dir}]: ").strip() or default_dir
                _export_selected_inverse_result_vtk(project, method_code, result, output_dir)
                continue
            if action == '7':
                removed = project.delete_inverse_result_by_index(method_code, index)
                print(f"  ✅ 已删除反演结果: {getattr(removed, 'label', '未命名反演结果')}")
                break
            print("  ⚠️ 无效操作")


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
    │  [5] 导入实测曲线 (.csv)                │
    │  [6] 导出当前反演结果 (.csv)            │
    │  [7] 导出当前反演结果 (VTK)             │
    │  [8] 保存整个项目 (文件夹)               │
    │  [9] 加载整个项目                        │
    │  [0] 返回主菜单                          │
    │                                         │
    └──────────────────────────────────────────┘
        """)
        
        choice = input("  请选择 [0-9]: ").strip()
        
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
            if project.current_result and hasattr(project.current_result, 'save_csv'):
                default_name = f"{project.current_result_method or 'forward'}_result.csv"
                fn = _ask_filename("保存CSV", default_name, mode='save')
                if fn:
                    project.current_result.save_csv(fn)
            else:
                print("  ⚠️ 还没有正演结果，请先计算")
        elif choice == '5':
            _import_observed_csv(project)
        elif choice == '6':
            if project.inverse_result and hasattr(project.inverse_result, 'save_csv'):
                default_name = f"{project.inverse_result_method or 'inverse'}_inverse_result.csv"
                fn = _ask_filename("保存反演结果CSV", default_name, mode='save')
                if fn:
                    project.inverse_result.save_csv(fn)
            else:
                print("  ⚠️ 还没有反演结果，请先运行外部反演")
        elif choice == '7':
            _export_current_inverse_vtk(project)
        elif choice == '8':
            name = input("  项目名称 [my_project]: ").strip() or "my_project"
            project.name = name
            project.save()
        elif choice == '9':
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
        inverse_method = s.get('forward.method')
        inverse_settings = s.get(f'inversion.{inverse_method}', {})
        inverse_label = _inverse_settings_label(inverse_method, inverse_settings)
        
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
    │  [5] 当前方法                                    │
    │      当前: {s.get('forward.method')}
    │                                                  │
    │  [6] 磁场参数                                    │
    │      B0: {s.get('magnetic.b0_strength')} nT, I={s.get('magnetic.b0_inclination')}°, D={s.get('magnetic.b0_declination')}°
    │                                                  │
    │  [7] 方法反演约束                                │
    │      {inverse_label}
    │                                                  │
    │  [8] 查看所有设置                                │
    │  [9] 恢复默认设置                                │
    │  [10] 保存设置                                   │
    │  [0] 返回主菜单                                  │
    │                                                  │
    └──────────────────────────────────────────────────┘
        """)
        
        choice = input("  请选择 [0-10]: ").strip()
        
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
            _edit_method(s)
        elif choice == '6':
            _edit_magnetic_settings(s)
        elif choice == '7':
            _edit_method_inversion_settings(s, inverse_method)
        elif choice == '8':
            s.print_all()
        elif choice == '9':
            s._load_defaults()
            project.update_settings()
            print("  ✅ 已恢复默认设置")
        elif choice == '10':
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
    ║       → 右键完成 → 输入名称和多方法物性                            ║
    ║       → 按 C 计算当前方法 → 查看曲线                               ║
    ║       → 关闭窗口                                                   ║
    ║                                                                   ║
    ║    2. 选择 [6] 设置                                                ║
    ║       → 切换当前方法 gravity / magnetic_3c                         ║
    ║       → 磁法可设置 B0 / 倾角 / 偏角                                ║
    ║                                                                   ║
    ║    3. 选择 [2] 正演计算                                            ║
    ║       → 进入正演子菜单                                              ║
    ║       → 选择重力或 3C井中磁测                                      ║
    ║                                                                   ║
    ║    4. 在正演子菜单选择 [3]                                          ║
    ║       → 对比当前方法的历史结果                                     ║
    ║                                                                   ║
    ║    5. 在 [4] 导入/导出 中导入实测 CSV                               ║
    ║       → 再到正演子菜单选择 [5] 做正演/实测对比                      ║
    ║                                                                   ║
    ║    6. 在正演子菜单选择 [4]                                          ║
    ║       → 管理结果历史、切换当前结果、导出或删除                     ║
    ║                                                                   ║
    ║    7. 在正演子菜单选择 [6]                                          ║
    ║       → 进入外部反演计算入口                                       ║
    ║                                                                   ║
    ║    8. 在 [6] 设置 → 方法反演约束                                    ║
    ║       → 选择 balanced / compact_body / wide_search 模板             ║
    ║       → 再微调 bounds 和 update_region                             ║
    ║                                                                   ║
    ║    9. 在正演子菜单选择 [7]                                          ║
    ║       → 管理反演结果历史、导出 VTK/CSV/NPZ                         ║
    ║                                                                   ║
    ║    10. 选择 [4] 导出                                               ║
    ║       → 导出模型/正演/反演 VTK 文件 → 用ParaView打开               ║
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
    ║  当前外部求解器:                                                   ║
    ║  ─────────────                                                   ║
    ║    • builtin: 纯Python（慢但可靠）                                 ║
    ║    • numpy:   NumPy向量化（快10~50倍）                             ║
    ║    • numba:   Numba JIT（快100~400倍，需pip install numba）        ║
    ║    • harmonica: Fatiando库（3C井中磁测使用）                       ║
    ║    • SimPEG: 重力反演 + 3C磁测反演 MVP                              ║
    ║                                                                   ║
    ║  当前磁法反演边界:                                                 ║
    ║  ─────────────                                                   ║
    ║    • 仅支持标量磁化率                                              ║
    ║    • 仅支持感应磁化                                                ║
    ║    • 暂不支持剩磁反演与矢量磁化率                                  ║
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

    current_target = s.get('mesh.target_cell_size')
    val = input(f"    目标格子尺寸 target_cell_size [{current_target}，回车=自动]: ").strip()
    if val:
        try:
            s.set('mesh.target_cell_size', float(val))
        except ValueError:
            print("    ⚠️ target_cell_size 格式错误")
    elif val == '':
        s.set('mesh.target_cell_size', None)

    current_mode = s.get('mesh.active_model.mode', 'auto')
    print(f"    active_model 模式: {current_mode}")
    val = input("    active_model 模式 [auto/manual，回车保持]: ").strip().lower()
    if val in ('auto', 'manual'):
        s.set('mesh.active_model.mode', val)

    if s.get('mesh.active_model.mode', 'auto') == 'manual':
        print("    编辑 active_model.bounds（格式: min max，直接回车保持）:")
        for axis in ['x_range', 'y_range', 'z_range']:
            current = s.get(f'mesh.active_model.bounds.{axis}')
            raw = input(f"      {axis} [{current[0]}, {current[1]}]: ").strip()
            if not raw:
                continue
            try:
                parts = raw.replace(',', ' ').replace('[', '').replace(']', '').split()
                s.set(f'mesh.active_model.bounds.{axis}', [float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                print(f"      ⚠️ {axis} 格式错误，保持原值")


METHOD_INVERSION_PRESETS = {
    'gravity': {
        'balanced': {
            'lower_bound': -1.5,
            'upper_bound': 1.5,
            'update_region.mode': 'all_active',
            'regularization.mode': 'smooth',
            'regularization.reference_model': 'property',
            'description': '默认平衡模板，适合初始解释。',
            'property_label': 'density',
            'unit': 'g/cc',
        },
        'compact_body': {
            'lower_bound': -0.8,
            'upper_bound': 1.2,
            'update_region.mode': 'property_only',
            'regularization.mode': 'compact',
            'regularization.reference_model': 'property',
            'description': '优先约束到已知异常附近，适合紧致矿体。',
            'property_label': 'density',
            'unit': 'g/cc',
        },
        'wide_search': {
            'lower_bound': -2.5,
            'upper_bound': 2.5,
            'update_region.mode': 'all_active',
            'regularization.mode': 'smooth',
            'regularization.reference_model': 'zero',
            'description': '放宽搜索范围，适合早期扫描。',
            'property_label': 'density',
            'unit': 'g/cc',
        },
    },
    'magnetic_3c': {
        'balanced': {
            'lower_bound': 0.0,
            'upper_bound': 0.08,
            'update_region.mode': 'all_active',
            'regularization.mode': 'smooth',
            'regularization.reference_model': 'property',
            'description': '默认平衡模板，适合井中磁法初始约束。',
            'property_label': 'susceptibility',
            'unit': 'SI',
        },
        'compact_body': {
            'lower_bound': 0.0,
            'upper_bound': 0.05,
            'update_region.mode': 'property_only',
            'regularization.mode': 'compact',
            'regularization.reference_model': 'property',
            'description': '优先收敛到磁性体附近，适合紧致磁异常。',
            'property_label': 'susceptibility',
            'unit': 'SI',
        },
        'wide_search': {
            'lower_bound': 0.0,
            'upper_bound': 0.15,
            'update_region.mode': 'all_active',
            'regularization.mode': 'smooth',
            'regularization.reference_model': 'zero',
            'description': '扩展搜索范围，适合早期磁法扫描。',
            'property_label': 'susceptibility',
            'unit': 'SI',
        },
    },
}


def _get_inversion_presets(method_code):
    return METHOD_INVERSION_PRESETS.get(method_code, {})


def _get_inversion_property_label(method_code):
    presets = _get_inversion_presets(method_code)
    sample = next(iter(presets.values()), None)
    if sample is None:
        return 'property', ''
    return sample.get('property_label', 'property'), sample.get('unit', '')


def _apply_method_inversion_preset(s, method_code, preset_name):
    presets = _get_inversion_presets(method_code)
    preset = presets.get(preset_name)
    if not preset:
        print(f"  ⚠️ {method_code} 未知 preset: {preset_name}")
        return False
    base = f'inversion.{method_code}'
    s.set(f'{base}.preset', preset_name)
    s.set(f'{base}.lower_bound', float(preset['lower_bound']))
    s.set(f'{base}.upper_bound', float(preset['upper_bound']))
    s.set(f'{base}.update_region.mode', preset['update_region.mode'])
    if 'regularization.mode' in preset:
        s.set(f'{base}.regularization.mode', preset['regularization.mode'])
    if 'regularization.reference_model' in preset:
        s.set(f'{base}.regularization.reference_model', preset['regularization.reference_model'])
    print(f"  ✅ 已应用 {method_code} 反演模板: {preset_name} — {preset['description']}")
    return True


def _apply_gravity_inversion_preset(s, preset_name):
    return _apply_method_inversion_preset(s, 'gravity', preset_name)


def _inverse_settings_label(method_code, settings_dict):
    prop_label, unit = _get_inversion_property_label(method_code)
    bounds = f"{settings_dict.get('lower_bound')}~{settings_dict.get('upper_bound')}"
    unit_text = f" {unit}" if unit else ""
    update_mode = settings_dict.get('update_region', {}).get('mode', 'all_active')
    preset = settings_dict.get('preset', 'balanced')
    reg = settings_dict.get('regularization', {})
    reg_mode = reg.get('mode', 'smooth')
    ref_mode = reg.get('reference_model', 'property')
    return (
        f"preset={preset}  {prop_label}={bounds}{unit_text}  "
        f"reg={reg_mode}/{ref_mode}  update_region={update_mode}"
    )


def _print_inverse_setting_summary(settings, method_code):
    inv = settings.get(f'inversion.{method_code}', {})
    prop_label, unit = _get_inversion_property_label(method_code)
    update_region = inv.get('update_region', {})
    title = "当前反演约束"
    print(f"  {title}:")
    print(f"    preset: {inv.get('preset', 'balanced')}")
    print(f"    {prop_label} bounds: {inv.get('lower_bound')} ~ {inv.get('upper_bound')} {unit}".rstrip())
    regularization = inv.get('regularization', {})
    print(
        "    regularization: "
        f"{regularization.get('mode', 'smooth')} / "
        f"{regularization.get('reference_model', 'property')}"
    )
    print(f"    update_region: {update_region.get('mode', 'all_active')}")
    note = _get_method_inverse_note(method_code)
    if note:
        print(f"    note: {note}")


def _get_method_inverse_note(method_code):
    if method_code == 'gravity':
        return "SimPEG 密度反演 MVP，按 active volume / bounds / update region 求解"
    if method_code == 'magnetic_3c':
        return "SimPEG 磁化率反演 MVP，仅标量磁化率 + 感应磁化，不含剩磁"
    return ""


def _edit_method_inversion_settings(s, method_code):
    print(f"\n  编辑 {method_code} 反演约束:")
    presets = _get_inversion_presets(method_code)
    current = s.get(f'inversion.{method_code}.preset', 'balanced')
    print(f"    当前 preset: {current}")
    print(f"    可选 preset: {' / '.join(presets.keys())}")
    raw = input("    选择 preset [回车保持]: ").strip()
    if raw in presets:
        _apply_method_inversion_preset(s, method_code, raw)

    inv_base = f'inversion.{method_code}'
    prop_label, unit = _get_inversion_property_label(method_code)
    lower_current = s.get(f'{inv_base}.lower_bound')
    upper_current = s.get(f'{inv_base}.upper_bound')
    unit_suffix = f" ({unit})" if unit else ''
    raw = input(f"    {prop_label} lower_bound{unit_suffix} [{lower_current}]: ").strip()
    if raw:
        try:
            s.set(f'{inv_base}.lower_bound', float(raw))
        except ValueError:
            print("    ⚠️ lower_bound 格式错误")
    raw = input(f"    {prop_label} upper_bound{unit_suffix} [{upper_current}]: ").strip()
    if raw:
        try:
            s.set(f'{inv_base}.upper_bound', float(raw))
        except ValueError:
            print("    ⚠️ upper_bound 格式错误")

    update_mode = s.get(f'{inv_base}.update_region.mode', 'all_active')
    print(f"    update_region 模式: {update_mode}")
    raw = input("    update_region 模式 [all_active/property_only/manual，回车保持]: ").strip().lower()
    if raw in ('all_active', 'property_only', 'manual'):
        s.set(f'{inv_base}.update_region.mode', raw)

    if s.get(f'{inv_base}.update_region.mode', 'all_active') == 'manual':
        print("    编辑 update_region.bounds（格式: min max，直接回车保持）:")
        for axis in ['x_range', 'y_range', 'z_range']:
            current = s.get(f'{inv_base}.update_region.bounds.{axis}')
            raw = input(f"      {axis} [{current[0]}, {current[1]}]: ").strip()
            if not raw:
                continue
            try:
                parts = raw.replace(',', ' ').replace('[', '').replace(']', '').split()
                s.set(f'{inv_base}.update_region.bounds.{axis}', [float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                print(f"      ⚠️ {axis} 格式错误，保持原值")

    regularization = s.get(f'{inv_base}.regularization', {})
    current_reg_mode = regularization.get('mode', 'smooth')
    raw = input(f"    regularization 模式 [smooth/compact，当前 {current_reg_mode}]: ").strip().lower()
    if raw in ('smooth', 'compact'):
        s.set(f'{inv_base}.regularization.mode', raw)

    current_ref_mode = regularization.get('reference_model', 'property')
    raw = input(
        f"    reference_model [property/zero，当前 {current_ref_mode}]: "
    ).strip().lower()
    if raw in ('property', 'zero'):
        s.set(f'{inv_base}.regularization.reference_model', raw)


def _edit_engine(s):
    current = s.get('forward.engine')
    method_code = s.get('forward.method')
    print(f"\n  当前方法: {method_code}")
    print(f"  当前引擎: {current}")
    print("  可选: auto / builtin / numpy / numba / harmonica")
    val = input("  选择引擎: ").strip()
    if val in ('auto', 'builtin', 'numpy', 'numba', 'harmonica'):
        s.set('forward.engine', val)
        print(f"  ✅ 引擎已切换为: {val}")


def _edit_method(s):
    current = s.get('forward.method')
    print(f"\n  当前方法: {current}")
    print("  可选: gravity / magnetic_3c")
    val = input("  选择方法: ").strip()
    if val in ('gravity', 'magnetic_3c'):
        s.set('forward.method', val)
        print(f"  ✅ 当前方法已切换为: {val}")


def _edit_magnetic_settings(s):
    print("\n  编辑磁场参数（直接回车保持不变）:")
    for key in ['b0_strength', 'b0_inclination', 'b0_declination']:
        current = s.get(f'magnetic.{key}')
        val = input(f"    {key} [{current}]: ").strip()
        if not val:
            continue
        try:
            s.set(f'magnetic.{key}', float(val))
        except ValueError:
            print(f"    ⚠️ {key} 格式错误")


def _ask_filename(title, default, mode='open'):
    """请求文件名"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        if mode == 'open':
            fn = filedialog.askopenfilename(title=title, filetypes=[('JSON', '*.json'), ('All', '*.*')])
        else:
            ext = os.path.splitext(default)[1] or '.dat'
            fn = filedialog.asksaveasfilename(
                title=title,
                defaultextension=ext,
                initialfile=default
            )
        root.destroy()
        return fn if fn else None
    except Exception:
        fn = input(f"  文件名 [{default}]: ").strip() or default
        return fn


def _ask_method_code(project, purpose="选择方法"):
    """询问方法代码。"""
    current = project.get_current_method()
    print(f"\n  {purpose}")
    print(f"  [1] gravity{' (当前)' if current == 'gravity' else ''}")
    print(f"  [2] magnetic_3c{' (当前)' if current == 'magnetic_3c' else ''}")
    raw = input("  请选择 [1-2，直接回车=当前方法]: ").strip()
    mapping = {'1': 'gravity', '2': 'magnetic_3c'}
    return mapping.get(raw, current)


def _import_observed_csv(project):
    """导入实测曲线 CSV。"""
    method_code = _ask_method_code(project, "选择要导入实测数据的方法")
    default_name = f"observed_{method_code}.csv"
    fn = _ask_filename("导入实测CSV", default_name, mode='open')
    if not fn:
        return
    try:
        observed = project.load_observed_result(method_code, fn)
        print(f"  ✅ 已导入实测数据: {getattr(observed, 'label', 'observed')}")
        if hasattr(observed, 'get_summary'):
            observed.get_summary()
    except Exception as exc:
        print(f"  ❌ 导入实测数据失败: {exc}")


def _sanitize_result_label(label):
    """把结果标签转换成适合文件名的形式。"""
    sanitized = str(label).replace(' ', '_').replace(':', '-').replace('/', '-')
    sanitized = sanitized.replace('\\', '-').replace('#', '_')
    return sanitized


def _result_filename_stub(result, method_code):
    """结果导出时的默认文件名前缀。"""
    label = getattr(result, 'label', method_code)
    return _sanitize_result_label(label)


def _describe_result_filter(project, filter_mode):
    """结果筛选器说明文字。"""
    if filter_mode == 'all':
        return '全部方法'
    if filter_mode == 'current':
        return f"当前方法 {project.get_current_method()}"
    return filter_mode


def _get_filtered_result_catalog(project, filter_mode):
    """按筛选条件返回结果目录。"""
    catalog = project.get_result_catalog()
    if filter_mode == 'all':
        return catalog
    if filter_mode == 'current':
        method_code = project.get_current_method()
        return [item for item in catalog if item['method_code'] == method_code]
    return [item for item in catalog if item['method_code'] == filter_mode]


def _get_filtered_inverse_catalog(project, filter_mode):
    """按筛选条件返回反演结果目录。"""
    catalog = project.get_inverse_result_catalog()
    if filter_mode == 'all':
        return catalog
    if filter_mode == 'current':
        method_code = project.get_current_method()
        return [item for item in catalog if item['method_code'] == method_code]
    return [item for item in catalog if item['method_code'] == filter_mode]


def _ask_result_filter_mode(project, current_mode):
    """请求切换结果筛选器。"""
    print(f"\n  当前筛选: {_describe_result_filter(project, current_mode)}")
    print("  [1] 全部方法")
    print(f"  [2] 当前方法 ({project.get_current_method()})")
    print("  [3] gravity")
    print("  [4] magnetic_3c")
    raw = input("  请选择筛选 [1-4，直接回车保持不变]: ").strip()
    mapping = {
        '1': 'all',
        '2': 'current',
        '3': 'gravity',
        '4': 'magnetic_3c',
    }
    return mapping.get(raw, current_mode)


def _export_selected_result_vtk(project, method_code, result, output_dir):
    """导出单条历史结果到独立的 VTK 目录。"""
    from export.vtk_export import VTKExporter

    exporter = VTKExporter(output_dir=output_dir)
    exporter.export_all(
        mesh_data=getattr(result, 'mesh_data', None),
        well=project.well,
        bodies=project.model_manager.bodies,
        gravity_result=result if method_code == 'gravity' else None,
        magnetic_result=result if method_code == 'magnetic_3c' else None,
    )


def _export_selected_inverse_result_vtk(project, method_code, result, output_dir):
    """导出单条历史反演结果到独立的 VTK 目录。"""
    from export.vtk_export import VTKExporter

    exporter = VTKExporter(output_dir=output_dir)
    exporter.export_all(
        mesh_data=getattr(result, 'recovered_mesh_data', None),
        well=project.well,
        bodies=project.model_manager.bodies,
        gravity_inverse_result=result if method_code == 'gravity' else None,
        magnetic_inverse_result=result if method_code == 'magnetic_3c' else None,
    )


def _export_current_inverse_vtk(project):
    """导出当前反演结果到独立 VTK 目录。"""
    if project.inverse_result is None or project.inverse_result_method is None:
        print("  ⚠️ 还没有反演结果，请先运行外部反演")
        return
    default_dir = _result_filename_stub(
        project.inverse_result, project.inverse_result_method
    ) + "_vtk"
    output_dir = input(f"  导出目录 [{default_dir}]: ").strip() or default_dir
    _export_selected_inverse_result_vtk(
        project,
        project.inverse_result_method,
        project.inverse_result,
        output_dir,
    )


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
        magnetic_result=project.magnetic_result,
        gravity_inverse_result=project.inverse_result if project.inverse_result_method == 'gravity' else None,
        magnetic_inverse_result=project.inverse_result if project.inverse_result_method == 'magnetic_3c' else None,
    )


def _show_forward_result_with_model(result, project):
    """按结果类型选择展示方式。"""
    if result is None:
        return
    if hasattr(result, 'plot_with_model'):
        result.plot_with_model(project.model_manager)
        return
    if hasattr(result, 'gz'):
        from visualization.plot_gravity import plot_gravity_with_model
        plot_gravity_with_model(result, bodies=project.model_manager.bodies)
        return
    if hasattr(result, 'plot'):
        result.plot()


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
