#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║   Borehole Geophysics Platform — 钻孔地球物理平台        ║
║                                                          ║
║   运行方法: python main.py                               ║
║                                                          ║
║   功能:                                                  ║
║     • 交互式地质建模（鼠标画矿体/地层线）                  ║
║     • 自适应网格剖分（八叉树/三棱柱/四面体）               ║
║     • 重力正演计算（自研解析+Numba加速+开源库）            ║
║     • 3D可视化 + VTK导出                                 ║
║     • 井地电法/VSP地震网格适配                            ║
║                                                          ║
║   作者: 你的名字                                         ║
║   版本: 1.0                                              ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os

# 确保项目根目录在搜索路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """程序主入口"""
    
    # 检查依赖
    from app.menu import check_dependencies, main_loop
    
    if not check_dependencies():
        print("\n  请安装缺失的依赖库后重试")
        sys.exit(1)
    
    # 创建项目
    from app.project import Project
    project = Project(name="default")
    
    # 自动加载上次的模型
    model_file = project.settings.get('export.model_file', 'model.json')
    if os.path.exists(model_file):
        project.model_manager.load(model_file)
    
    # 启动主菜单
    try:
        main_loop(project)
    except KeyboardInterrupt:
        print("\n\n    程序被中断，再见！")
    except Exception as e:
        print(f"\n    ❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n    请截图错误信息，反馈给开发者")


if __name__ == '__main__':
    main()