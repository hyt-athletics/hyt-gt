"""
第十步：交互建模 + 重力正演 一体化

完整工作流:
  1. 用鼠标画矿体
  2. 自动生成网格
  3. 计算钻孔重力
  4. 实时查看重力曲线
  5. 修改矿体 → 重新计算 → 对比

运行: python run_step10.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置中文字体（必须在导入matplotlib相关模块之前）
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from interactive.body_manager import ModelManager, GeologicalBody
from interactive.drawing_tool import InteractiveDrawingTool
from borehole.well import VerticalWell
from mesh.octree import OctreeMesh
from forward.borehole_gravity import BoreholeGravityCalculator
from visualization.plot_gravity import plot_gravity_with_model


class IntegratedModeler:
    """
    交互建模 + 正演计算 一体化工具
    
    在交互绘图工具的基础上，增加了"计算重力"的功能
    """
    
    def __init__(self):
        self.manager = ModelManager()
        
        self.manager.domain = {
            'x_range': (0, 1000),
            'y_range': (0, 1000),
            'z_range': (0, 1000),
        }
        self.manager.well_info = {
            'x': 500, 'y': 500,
            'z_top': 10, 'z_bottom': 800,
            'n_stations': 40,
        }
        
        self.well = VerticalWell(**self.manager.well_info)
        self.last_result = None
        self.result_history = []
    
    def run(self):
        """启动交互工具"""
        print("""
╔════════════════════════════════════════════════════════╗
║     交互建模 + 重力正演 一体化工具                       ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  操作流程:                                              ║
║    1. 用 P/E/R 画矿体                                  ║
║    2. 按 C 计算重力正演                                  ║
║    3. 修改矿体后，再按 C 重新计算                         ║
║    4. 按 F 对比前后两次计算结果                           ║
║                                                        ║
║  新增快捷键:                                            ║
║    C → 计算重力正演                                     ║
║    F → 对比前后两次结果                                  ║
║    V → 3D查看                                          ║
║    G → 生成网格                                         ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
        """)
        
        # 加载已保存的模型
        if os.path.exists('model.json'):
            answer = input("发现已保存的模型，是否加载？(y/n) [y]: ").strip().lower()
            if answer != 'n':
                self.manager.load('model.json')
        
        # 创建自定义的绘图工具（加入正演功能）
        tool = IntegratedDrawingTool(
            model_manager=self.manager,
            integrated_modeler=self
        )
        tool.run()
        
        # 窗口关闭后
        if len(self.manager.bodies) > 0:
            self.manager.print_summary()
            save = input("\n保存模型？(y/n) [y]: ").strip().lower()
            if save != 'n':
                self.manager.save('model.json')
    
    def compute_gravity(self):
        """执行重力正演"""
        if len(self.manager.bodies) == 0:
            print("  ⚠️ 没有地质体！请先画矿体。")
            return None
        
        print("\n" + "=" * 50)
        print("  开始重力正演计算")
        print("=" * 50)
        
        # 生成网格
        domain = self.manager.domain
        mesh = OctreeMesh(
            x_range=domain['x_range'],
            y_range=domain['y_range'],
            z_range=domain['z_range'],
            max_level=5
        )
        
        # 钻孔加密
        mesh.refine_along_borehole(self.well, levels_and_radii=[
            (5, 50), (4, 150), (3, 300),
        ])
        
        # 异常体边界加密
        for body in self.manager.bodies:
            mesh.refine_at_boundary(body.contains_point_3d, target_level=4)
        
        mesh.balance()
        
        # 赋物性
        for body in self.manager.bodies:
            mesh.assign_property(body.contains_point_3d, 'density', body.density)
        
        mesh.get_info()
        
        # 正演计算
        calc = BoreholeGravityCalculator(engine='auto')
        result = calc.compute(self.well, mesh.to_arrays())
        result.get_summary()
        
        # 保存历史
        if self.last_result is not None:
            self.result_history.append(self.last_result)
        self.last_result = result
        
        return result
    
    def show_result(self):
        """显示正演结果"""
        if self.last_result is None:
            print("  还没有计算结果。按 C 先计算。")
            return
        
        plot_gravity_with_model(
            self.last_result,
            bodies=self.manager.bodies
        )
    
    def compare_results(self):
        """对比最近两次计算结果"""
        if self.last_result is None:
            print("  还没有计算结果。")
            return
        if len(self.result_history) == 0:
            print("  只有一次计算结果，需要修改模型后再算一次才能对比。")
            self.show_result()
            return
        
        import matplotlib.pyplot as plt
        
        prev = self.result_history[-1]
        curr = self.last_result
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
        
        # 左图：两次结果叠加
        ax1.plot(prev.gz, prev.depths, 'b--', linewidth=2, alpha=0.6, label='修改前')
        ax1.plot(curr.gz, curr.depths, 'r-', linewidth=2, label='修改后')
        ax1.fill_betweenx(curr.depths, prev.gz, curr.gz,
                          alpha=0.15, color='green', label='差异')
        ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
        ax1.set_xlabel('gz (mGal)', fontsize=12)
        ax1.set_ylabel('Depth (m)', fontsize=12)
        ax1.set_title('前后对比', fontsize=13)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右图：差值
        diff = curr.gz - prev.gz
        ax2.plot(diff, curr.depths, 'g-', linewidth=2)
        ax2.fill_betweenx(curr.depths, 0, diff,
                          where=(diff > 0), color='#e74c3c', alpha=0.2)
        ax2.fill_betweenx(curr.depths, 0, diff,
                          where=(diff < 0), color='#3498db', alpha=0.2)
        ax2.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
        ax2.set_xlabel('Δgz (mGal)', fontsize=12)
        ax2.set_title('差值 (修改后 - 修改前)', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('模型修���前后的重力响应对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class IntegratedDrawingTool(InteractiveDrawingTool):
    """
    扩展的交互绘图工具，增加了 C/F 快捷键
    """
    
    def __init__(self, model_manager, integrated_modeler):
        super().__init__(model_manager)
        self.modeler = integrated_modeler
    
    def _setup_figure(self):
        """重写：增加额外的快捷键说明"""
        super()._setup_figure()
        
        # 更新底部快捷键提示
        self._shortcut_text.set_text(
            '[P]多边形 [E]椭圆 [R]矩形 [S]选择 '
            '[C]计算重力 [F]前后对比 '
            '[V]3D [H]帮助 [Ctrl+S]保存'
        )
    
    def _on_key(self, event):
        """重写：增加 C 和 F 键"""
        if event.key is None:
            return
        
        key = event.key.lower()
        
        if key == 'c':
            self._compute_gravity()
        elif key == 'f':
            self._compare_gravity()
        else:
            super()._on_key(event)
    
    def _compute_gravity(self):
        """计算重力正演"""
        self._update_status('正在计算重力正演...')
        
        result = self.modeler.compute_gravity()
        
        if result is not None:
            self._update_status(
                f'✅ 计算完成! gz: {result.min_anomaly:.3f} ~ '
                f'{result.max_anomaly:.3f} mGal | 按F对比')
            self.modeler.show_result()
        else:
            self._update_status('⚠️ 计算失败，请先画矿体')
    
    def _compare_gravity(self):
        """对比前后两次"""
        self.modeler.compare_results()


def main():
    modeler = IntegratedModeler()
    modeler.run()


if __name__ == '__main__':
    main()