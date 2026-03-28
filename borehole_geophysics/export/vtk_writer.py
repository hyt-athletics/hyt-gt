"""
VTK文件写入器（纯Python手写，不依赖任何VTK库）

支持的格式：
  .vtu  → 非结构化网格（最通用）
  .vtp  → 多边形表面
  .vtm  → 多块数据集

写出的文件可以直接用 ParaView 打开

为什么不用VTK库？
  1. VTK库很大（>500MB），安装经常出问题
  2. 我们只需要"写文件"功能，手写XML就够了
  3. 你能完全理解每一行在做什么
"""

import numpy as np
import os
import struct
import base64


# ====================================================================
#  VTK单元类型编号
# ====================================================================

VTK_VERTEX = 1
VTK_LINE = 3
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13        # 三棱柱


# ====================================================================
#  工具函数
# ====================================================================

def _float_array_to_string(arr, precision=6):
    """把numpy数组转成空格分隔的字符串"""
    flat = np.asarray(arr).ravel()
    return ' '.join(f'{v:.{precision}f}' for v in flat)


def _int_array_to_string(arr):
    """把整数数组转成空格分隔的字符串"""
    flat = np.asarray(arr, dtype=int).ravel()
    return ' '.join(str(v) for v in flat)


def _indent(text, level):
    """给文本加缩进"""
    prefix = '  ' * level
    return '\n'.join(prefix + line for line in text.split('\n'))


# ====================================================================
#  VTU 写入器（非结构化网格）
# ====================================================================

def write_vtu(filename, points, cells, cell_types,
              cell_data=None, point_data=None):
    """
    写入 .vtu 文件（非结构化网格）
    
    参数:
        filename: 输出文件名，如 'mesh.vtu'
        points: 节点坐标 (N_points, 3) numpy数组
        cells: 单元连接列表 [(n, v0, v1, ...), ...]
               n = 这个单元有几个节点
               v0, v1, ... = 节点索引
               或者是 numpy数组格式
        cell_types: 单元类型数组 (N_cells,)
                    如 [10, 10, 10, ...] 表示全是四面体
        cell_data: 字典 {'density': array, 'velocity': array, ...}
                   每个值是 (N_cells,) 的数组
        point_data: 字典 {'temperature': array, ...}
                    每个值是 (N_points,) 的数组
    
    用法:
        write_vtu('mesh.vtu',
                  points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]),
                  cells=[(4, 0, 1, 2, 3)],
                  cell_types=[10])
    """
    points = np.asarray(points, dtype=float)
    cell_types = np.asarray(cell_types, dtype=int)
    n_points = len(points)
    
    # 解析单元连接
    if isinstance(cells, np.ndarray) and cells.ndim == 1:
        # 已经是扁平格式 [n, v0, v1, ..., n, v0, v1, ...]
        flat_cells = cells
    else:
        # 列表格式，转成扁平
        flat_cells = []
        for cell in cells:
            if isinstance(cell, (list, tuple)):
                flat_cells.extend(cell)
            else:
                flat_cells.append(cell)
        flat_cells = np.array(flat_cells, dtype=int)
    
    # 计算 connectivity 和 offsets
    connectivity = []
    offsets = []
    offset = 0
    i = 0
    n_cells = 0
    while i < len(flat_cells):
        n_verts = flat_cells[i]
        verts = flat_cells[i+1:i+1+n_verts]
        connectivity.extend(verts)
        offset += n_verts
        offsets.append(offset)
        i += 1 + n_verts
        n_cells += 1
    
    connectivity = np.array(connectivity, dtype=int)
    offsets = np.array(offsets, dtype=int)
    
    # 确保点是3D的
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(n_points)])
    
    # 构建XML
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">')
    lines.append('  <UnstructuredGrid>')
    lines.append(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">')
    
    # 点坐标
    lines.append('      <Points>')
    lines.append('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">')
    lines.append('          ' + _float_array_to_string(points))
    lines.append('        </DataArray>')
    lines.append('      </Points>')
    
    # 单元
    lines.append('      <Cells>')
    lines.append('        <DataArray type="Int32" Name="connectivity" format="ascii">')
    lines.append('          ' + _int_array_to_string(connectivity))
    lines.append('        </DataArray>')
    lines.append('        <DataArray type="Int32" Name="offsets" format="ascii">')
    lines.append('          ' + _int_array_to_string(offsets))
    lines.append('        </DataArray>')
    lines.append('        <DataArray type="UInt8" Name="types" format="ascii">')
    lines.append('          ' + _int_array_to_string(cell_types))
    lines.append('        </DataArray>')
    lines.append('      </Cells>')
    
    # 单元数据
    if cell_data:
        lines.append('      <CellData>')
        for name, values in cell_data.items():
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                n_comp = 1
            else:
                n_comp = values.shape[1]
            lines.append(f'        <DataArray type="Float64" Name="{name}" '
                        f'NumberOfComponents="{n_comp}" format="ascii">')
            lines.append('          ' + _float_array_to_string(values))
            lines.append('        </DataArray>')
        lines.append('      </CellData>')
    
    # 点数据
    if point_data:
        lines.append('      <PointData>')
        for name, values in point_data.items():
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                n_comp = 1
            else:
                n_comp = values.shape[1]
            lines.append(f'        <DataArray type="Float64" Name="{name}" '
                        f'NumberOfComponents="{n_comp}" format="ascii">')
            lines.append('          ' + _float_array_to_string(values))
            lines.append('        </DataArray>')
        lines.append('      </PointData>')
    
    lines.append('    </Piece>')
    lines.append('  </UnstructuredGrid>')
    lines.append('</VTKFile>')
    
    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    size_kb = os.path.getsize(filename) / 1024
    print(f"  ✅ {filename} ({size_kb:.0f} KB, {n_points} 点, {n_cells} 单元)")


# ====================================================================
#  VTP 写入器（多边形表面）
# ====================================================================

def write_vtp(filename, points, polygons, point_data=None, cell_data=None):
    """
    写入 .vtp 文件（多边形表面网格）
    
    参数:
        filename: 输出文件名
        points: 节点坐标 (N, 3)
        polygons: 多边形列表 [(v0, v1, v2), ...] 或 [(v0, v1, v2, v3), ...]
        point_data: 点数据字典
        cell_data: 面数据字典
    """
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    n_polys = len(polygons)
    
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(n_points)])
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">')
    lines.append('  <PolyData>')
    lines.append(f'    <Piece NumberOfPoints="{n_points}" NumberOfPolys="{n_polys}">')
    
    # 点
    lines.append('      <Points>')
    lines.append('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">')
    lines.append('          ' + _float_array_to_string(points))
    lines.append('        </DataArray>')
    lines.append('      </Points>')
    
    # 多边形
    connectivity = []
    offsets = []
    offset = 0
    for poly in polygons:
        connectivity.extend(poly)
        offset += len(poly)
        offsets.append(offset)
    
    lines.append('      <Polys>')
    lines.append('        <DataArray type="Int32" Name="connectivity" format="ascii">')
    lines.append('          ' + _int_array_to_string(connectivity))
    lines.append('        </DataArray>')
    lines.append('        <DataArray type="Int32" Name="offsets" format="ascii">')
    lines.append('          ' + _int_array_to_string(offsets))
    lines.append('        </DataArray>')
    lines.append('      </Polys>')
    
    if cell_data:
        lines.append('      <CellData>')
        for name, values in cell_data.items():
            lines.append(f'        <DataArray type="Float64" Name="{name}" format="ascii">')
            lines.append('          ' + _float_array_to_string(values))
            lines.append('        </DataArray>')
        lines.append('      </CellData>')
    
    if point_data:
        lines.append('      <PointData>')
        for name, values in point_data.items():
            lines.append(f'        <DataArray type="Float64" Name="{name}" format="ascii">')
            lines.append('          ' + _float_array_to_string(values))
            lines.append('        </DataArray>')
        lines.append('      </PointData>')
    
    lines.append('    </Piece>')
    lines.append('  </PolyData>')
    lines.append('</VTKFile>')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    size_kb = os.path.getsize(filename) / 1024
    print(f"  ✅ {filename} ({size_kb:.0f} KB, {n_points} 点, {n_polys} 面)")


# ====================================================================
#  VTM 写入器（多块数据集）
# ====================================================================

def write_vtm(filename, blocks):
    """
    写入 .vtm 文件（多块数据集——把多个VTK文件打包在一起）
    
    参数:
        filename: 输出文件名，如 'model.vtm'
        blocks: 列表 [(名字, 相对路径), ...]
                如 [('mesh', 'mesh.vtu'), ('borehole', 'borehole.vtp')]
    
    在ParaView中打开vtm文件，会自动加载所有子文件
    可以单独控制每个块的可见性
    """
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<VTKFile type="vtkMultiBlockDataSet" version="1.0">')
    lines.append('  <vtkMultiBlockDataSet>')
    
    for i, (name, path) in enumerate(blocks):
        lines.append(f'    <DataSet index="{i}" name="{name}" file="{path}"/>')
    
    lines.append('  </vtkMultiBlockDataSet>')
    lines.append('</VTKFile>')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✅ {filename} (包含 {len(blocks)} 个块)")


# ====================================================================
#  CSV 写入器（用于正演结果）
# ====================================================================

def write_csv_with_header(filename, columns, header_lines=None):
    """
    写入带注释头的CSV文件
    
    参数:
        filename: 文件名
        columns: 字典 {'depth': array, 'gz': array, ...}
        header_lines: 注释行列表
    """
    names = list(columns.keys())
    arrays = [np.asarray(columns[n]) for n in names]
    
    with open(filename, 'w', encoding='utf-8') as f:
        if header_lines:
            for line in header_lines:
                f.write(f'# {line}\n')
        
        f.write(','.join(names) + '\n')
        
        n_rows = len(arrays[0])
        for i in range(n_rows):
            row = ','.join(f'{arr[i]:.6f}' for arr in arrays)
            f.write(row + '\n')
    
    size_kb = os.path.getsize(filename) / 1024
    print(f"  ✅ {filename} ({size_kb:.0f} KB, {n_rows} 行)")