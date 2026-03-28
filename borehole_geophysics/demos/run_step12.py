"""
第十二步：升级版交互建模

包含所有新功能：
  - 撤销/重做
  - 拖拽移动
  - 顶点编辑
  - 地层线
  - 双击编辑物性
  - 复制粘贴
  - 网格吸附
  - 正演计算
  - VTK导出

运行: python run_step12.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive.body_manager import ModelManager
from interactive.drawing_tool_v2 import InteractiveDrawingToolV2
from borehole.well import VerticalWell


class FullModeler:
    """完整一体化建模器"""
    
    def __init__(self):
        self.manager = ModelManager()
        self.manager.layers = []
        
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
    
    def compute(self):
        """正演计算"""
        from mesh.octree import OctreeMesh
        from forward.borehole_gravity import BoreholeGravityCalculator
        from interactive.layer_tool import layers_to_bodies
        from visualization.plot_gravity import plot_gravity_with_model
        
        all_bodies = list(self.manager.bodies)
        if hasattr(self.manager, 'layers') and self.manager.layers:
            layer_bodies = layers_to_bodies(
                self.manager.layers,
                self.manager.domain['x_range'],
                self.manager.domain['z_range']
            )
            all_bodies.extend(layer_bodies)
        
        if not all_bodies:
            print("  ⚠️ 没有地质体")
            return
        
        domain = self.manager.domain
        mesh = OctreeMesh(
            x_range=domain['x_range'],
            y_range=domain['y_range'],
            z_range=domain['z_range'],
            max_level=5
        )
        
        mesh.refine_along_borehole(self.well, levels_and_radii=[
            (5, 50), (4, 150), (3, 300),
        ])
        for body in all_bodies:
            mesh.refine_at_boundary(body.contains_point_3d, target_level=4)
        mesh.balance()
        for body in all_bodies:
            mesh.assign_property(body.contains_point_3d, 'density', body.density)
        
        calc = BoreholeGravityCalculator(engine='auto')
        self.last_result = calc.compute(self.well, mesh.to_arrays())
        self.last_result.get_summary()
        
        plot_gravity_with_model(self.last_result, bodies=self.manager.bodies)
    
    def export_vtk(self):
        """VTK导出"""
        from mesh.octree import OctreeMesh
        from export.vtk_export import VTKExporter
        from interactive.layer_tool import layers_to_bodies
        
        all_bodies = list(self.manager.bodies)
        if hasattr(self.manager, 'layers') and self.manager.layers:
            all_bodies.extend(layers_to_bodies(
                self.manager.layers,
                self.manager.domain['x_range'],
                self.manager.domain['z_range']))
        
        if not all_bodies:
            print("  ⚠️ 没有地质体")
            return
        
        domain = self.manager.domain
        mesh = OctreeMesh(
            x_range=domain['x_range'],
            y_range=domain['y_range'],
            z_range=domain['z_range'],
            max_level=5
        )
        mesh.refine_along_borehole(self.well, levels_and_radii=[
            (5, 50), (4, 150), (3, 300)])
        for body in all_bodies:
            mesh.refine_at_boundary(body.contains_point_3d, target_level=4)
        mesh.balance()
        for body in all_bodies:
            mesh.assign_property(body.contains_point_3d, 'density', body.density)
        
        exporter = VTKExporter(output_dir='vtk_output')
        exporter.export_all(
            octree_mesh=mesh, well=self.well, bodies=all_bodies,
            gravity_result=self.last_result)
    
    def run(self):
        if os.path.exists('model.json'):
            ans = input("发现model.json，加载？(y/n) [y]: ").strip().lower()
            if ans != 'n':
                self.manager.load('model.json')
        
        tool = InteractiveDrawingToolV2(
            model_manager=self.manager,
            on_compute=self.compute,
            on_export=self.export_vtk
        )
        tool.run()
        
        if self.manager.bodies:
            self.manager.print_summary()
            if input("\n保存？(y/n) [y]: ").strip().lower() != 'n':
                self.manager.save('model.json')


def main():
    modeler = FullModeler()
    modeler.run()


if __name__ == '__main__':
    main()