"""
步骤15：3C井中磁测演示

演示内容：
  1. 构建单个磁性体
  2. 生成八叉树网格
  3. 调用 harmonica 进行三分量井中磁测
  4. 显示模型与四条曲线
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.project import Project
from interactive.body_manager import GeologicalBody
from visualization.plot_magnetic import plot_magnetic_with_model


def main():
    project = Project('magnetic_demo')
    project.settings.set('forward.method', 'magnetic_3c')
    project.settings.set('forward.engine', 'harmonica')

    body = GeologicalBody(
        '磁性体1',
        [(420, 260), (580, 260), (580, 520), (420, 520)],
        density=0.3,
        susceptibility=0.03,
        remanent_mag=0.4,
        remanent_inclination=30,
        remanent_declination=15,
    )
    project.model_manager.add_body(body)

    project.generate_mesh_for_method('magnetic_3c')
    result = project.run_forward('magnetic_3c')
    plot_magnetic_with_model(result, bodies=project.model_manager.bodies)


if __name__ == '__main__':
    main()
