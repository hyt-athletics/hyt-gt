"""
第八步：交互式建模界面

运行方法：
    python run_step8.py

操作流程：
    1. 程序启动，弹出绘图窗口
    2. 按 P 进入多边形模式
    3. 左键点击放置矿体顶点
    4. 右键或 Enter 完成绘制
    5. 在弹出对话框中输入名称和密度
    6. 继续画更多矿体
    7. 按 V 查看3D效果
    8. 按 G 生成网格
    9. Ctrl+S 保存模型
"""

import sys
import os

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置中文字体（必须在导入matplotlib相关模块之前）
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from interactive.body_manager import ModelManager
from interactive.drawing_tool import InteractiveDrawingTool


def main():
    print("╔════════════════════════════════════════════╗")
    print("║    钻孔地球物理 — 交互式地质建模工具         ║")
    print("╚════════════════════════════════════════════╝")
    
    # 创建模型管理器
    manager = ModelManager()
    
    # 设置模型域
    manager.domain = {
        'x_range': (0, 1000),
        'y_range': (0, 1000),
        'z_range': (0, 1000),
    }
    
    # 设置钻孔
    manager.well_info = {
        'x': 500,
        'y': 500,
        'z_top': 10,
        'z_bottom': 800,
        'n_stations': 40,
    }
    
    # 检查是否有已保存的模型
    if os.path.exists('model.json'):
        answer = input("发现已保存的模型 model.json，是否加载？(y/n) [y]: ").strip().lower()
        if answer != 'n':
            manager.load('model.json')
    
    # 启动交互工具
    tool = InteractiveDrawingTool(model_manager=manager)
    tool.run()
    
    # 窗口关闭后
    print("\n窗口已关闭。")
    
    if len(manager.bodies) > 0:
        manager.print_summary()
        
        save = input("\n是否保存模型？(y/n) [y]: ").strip().lower()
        if save != 'n':
            manager.save('model.json')
    
    print("程序结束。")


if __name__ == '__main__':
    main()