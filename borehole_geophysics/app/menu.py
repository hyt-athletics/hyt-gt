"""
主菜单系统
"""

import os
import sys


BANNER = r"""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     ██████╗  ██╗  ██╗  ██████╗  ██████╗  ███████╗     ║
    ║     ██╔══██╗ ██║  ██║ ██╔════╝  ██╔══██╗ ██╔════╝     ║
    ║     ██████╔╝ ███████║ ██║  ███╗ ██████╔╝ ███████╗     ║
    ║     ██╔══██╗ ██╔══██║ ██║   ██║ ██╔═══╝  ╚════██║     ║
    ║     ██████╔╝ ██║  ██║ ╚██████╔╝ ██║      ███████║     ║
    ║     ╚═════╝  ╚═╝  ╚═╝  ╚═════╝  ╚═╝      ╚══════╝    ║
    ║                                                       ║
    ║     Borehole Geophysics Platform  v1.0                ║
    ║     钻孔地球物理交互建模与正演平台                       ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
"""

MENU = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     [1]  🎨  交互式建模        画矿体 + 地层线        ║
    ║     [2]  📊  重力正演          网格 → 计算 → 曲线     ║
    ║     [3]  🔬  网格工具          各类网格演示            ║
    ║     [4]  📦  导入/导出         模型 / VTK / CSV       ║
    ║     [5]  ⚡  性能测试          基准测试               ║
    ║     [6]  ⚙️   设置             域/钻孔/引擎参数       ║
    ║     [7]  📖  帮助              操作说明               ║
    ║     [0]  退出                                        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
"""


def check_dependencies():
    """检查依赖库的安装状态"""
    deps = {
        'numpy': ('必须', True),
        'matplotlib': ('必须', True),
        'scipy': ('推荐', False),
        'pyvista': ('推荐（3D可视化）', False),
        'numba': ('推荐（性能加速）', False),
        'harmonica': ('可选（正演库）', False),
    }
    
    print("\n  依赖库检查:")
    all_ok = True
    
    for name, (desc, required) in deps.items():
        try:
            __import__(name)
            status = "✅"
        except ImportError:
            status = "❌" if required else "⬜"
            if required:
                all_ok = False
        print(f"    {status} {name:15s} — {desc}")
    
    if not all_ok:
        print("\n  ❌ 缺少必要依赖！请运行:")
        print("     pip install numpy matplotlib scipy pyvista")
        return False
    
    print()
    return True


def print_project_status(project):
    """在菜单上方显示当前项目状态"""
    n_bodies = len(project.model_manager.bodies)
    n_layers = len(getattr(project.model_manager, 'layers', []))
    has_mesh = project.mesh_data is not None
    has_result = project.gravity_result is not None
    
    status_parts = []
    if n_bodies > 0:
        status_parts.append(f"{n_bodies}个地质体")
    if n_layers > 0:
        status_parts.append(f"{n_layers}条地层线")
    if has_mesh:
        status_parts.append(f"网格✅")
    if has_result:
        status_parts.append(f"正演✅")
    
    if status_parts:
        status = " | ".join(status_parts)
    else:
        status = "空项目 — 请选择[1]开始建模"
    
    print(f"    📁 项目: {project.name} — {status}")


def main_loop(project):
    """主菜单循环"""
    from app.workflows import (
        workflow_interactive_modeling,
        workflow_gravity_forward,
        workflow_mesh_tools,
        workflow_import_export,
        workflow_benchmark,
        workflow_settings,
        workflow_help,
    )
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
        
        print(BANNER)
        print_project_status(project)
        print(MENU)
        
        choice = input("    请选择 [0-7]: ").strip()
        
        if choice == '0':
            # 退出前询问保存
            if len(project.model_manager.bodies) > 0:
                ans = input("\n    保存模型？(y/n) [y]: ").strip().lower()
                if ans != 'n':
                    project.model_manager.save(
                        project.settings.get('export.model_file', 'model.json')
                    )
            print("\n    再见！👋\n")
            break
        
        elif choice == '1':
            workflow_interactive_modeling(project)
        
        elif choice == '2':
            workflow_gravity_forward(project)
            input("\n  按Enter返回主菜单...")
        
        elif choice == '3':
            workflow_mesh_tools(project)
        
        elif choice == '4':
            workflow_import_export(project)
        
        elif choice == '5':
            workflow_benchmark(project)
        
        elif choice == '6':
            workflow_settings(project)
        
        elif choice == '7':
            workflow_help()
        
        else:
            print("    无效选择，请输入 0-7")
            input("    按Enter继续...")