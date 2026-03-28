"""
八叉树自适应网格

核心思想：
    - 整个模型从一个大方块开始
    - 需要精细的地方（钻孔旁边、异常体边界）不断切成8小块
    - 不需要精细的地方保持大块
    
这样既精确又省计算量
"""

import numpy as np


class OctreeNode:
    """
    八叉树的一个节点 = 一个方块
    
    如果这个方块没有被切（is_leaf=True）→ 它就是一个计算格子
    如果被切了（is_leaf=False）→ 它有8个子方块
    """
    
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, level=0):
        """
        参数:
            x_min, x_max: 方块在X方向的范围
            y_min, y_max: 方块在Y方向的范围
            z_min, z_max: 方块在Z方向的范围
            level: 这个方块是第几次切割产生的（0=根节点，最大的那个）
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.level = level
        
        # 是否是叶节点（没被切 = 是最终的计算格子）
        self.is_leaf = True
        
        # 8个子节点（只有被切了才有）
        self.children = None
        
        # 物性值（只有叶节点需要）
        self.density = 0.0
        self.susceptibility = 0.0
        self.resistivity = 100.0
        self.velocity = 3000.0
    
    @property
    def center(self):
        """方块中心点坐标"""
        return (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
            0.5 * (self.z_min + self.z_max)
        )
    
    @property
    def size(self):
        """方块三个方向的尺寸"""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        )
    
    @property
    def volume(self):
        """方块体积"""
        dx, dy, dz = self.size
        return dx * dy * dz
    
    @property
    def min_edge(self):
        """最短边长度"""
        return min(self.size)
    
    def subdivide(self):
        """
        把这个方块切成8个小方块
        
        切法：沿x、y、z各切一刀，从中心切开
        """
        if not self.is_leaf:
            return  # 已经切过了，不重复切
        
        # 中心点
        xc = 0.5 * (self.x_min + self.x_max)
        yc = 0.5 * (self.y_min + self.y_max)
        zc = 0.5 * (self.z_min + self.z_max)
        
        next_level = self.level + 1
        
        # 创建8个子方块
        self.children = [
            # 下层4个 (z_min ~ zc)
            OctreeNode(self.x_min, xc, self.y_min, yc, self.z_min, zc, next_level),
            OctreeNode(xc, self.x_max, self.y_min, yc, self.z_min, zc, next_level),
            OctreeNode(self.x_min, xc, yc, self.y_max, self.z_min, zc, next_level),
            OctreeNode(xc, self.x_max, yc, self.y_max, self.z_min, zc, next_level),
            # 上层4个 (zc ~ z_max)
            OctreeNode(self.x_min, xc, self.y_min, yc, zc, self.z_max, next_level),
            OctreeNode(xc, self.x_max, self.y_min, yc, zc, self.z_max, next_level),
            OctreeNode(self.x_min, xc, yc, self.y_max, zc, self.z_max, next_level),
            OctreeNode(xc, self.x_max, yc, self.y_max, zc, self.z_max, next_level),
        ]
        
        # 子方块继承父方块的物性
        for child in self.children:
            child.density = self.density
            child.susceptibility = self.susceptibility
            child.resistivity = self.resistivity
            child.velocity = self.velocity
        
        self.is_leaf = False
    
    def get_leaves(self):
        """
        获取所有叶节点（最终的计算格子）
        
        递归地往下找，直到找到没被切的方块
        """
        if self.is_leaf:
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves
    
    def get_corners(self):
        """获取方块的8个角点坐标"""
        return [
            (self.x_min, self.y_min, self.z_min),
            (self.x_max, self.y_min, self.z_min),
            (self.x_min, self.y_max, self.z_min),
            (self.x_max, self.y_max, self.z_min),
            (self.x_min, self.y_min, self.z_max),
            (self.x_max, self.y_min, self.z_max),
            (self.x_min, self.y_max, self.z_max),
            (self.x_max, self.y_max, self.z_max),
        ]


class OctreeMesh:
    """
    八叉树自适应网格
    
    使用流程：
        1. 创建网格（指定范围和最大细分层级）
        2. 调用各种 refine_xxx 方法加密
        3. 调用 balance() 确保网格质量
        4. 赋物性值
        5. 导出供正演使用
    """
    
    def __init__(self, x_range, y_range, z_range, max_level=6):
        """
        参数:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            z_range: (z_min, z_max)
            max_level: 最大细分层级
        """
        self.root = OctreeNode(
            x_range[0], x_range[1],
            y_range[0], y_range[1],
            z_range[0], z_range[1],
            level=0
        )
        self.max_level = max_level
    
    def refine_around_point(self, point, radius, target_level):
        """
        在一个点周围的球形区域内加密网格
        """
        self._refine_point_recursive(self.root, point, radius, target_level)
    
    def _refine_point_recursive(self, node, point, radius, target_level):
        """递归实现点加密"""
        if node.level >= target_level:
            return
        if node.level >= self.max_level:
            return
        if not self._box_intersects_sphere(node, point, radius):
            return
        
        if node.is_leaf:
            node.subdivide()
        
        for child in node.children:
            self._refine_point_recursive(child, point, radius, target_level)
    
    def refine_along_borehole(self, well, levels_and_radii):
        """
        沿钻孔路径加密：近处密、远处疏
        """
        path_points = well.get_path_points(n_points=50)
        
        for target_level, radius in levels_and_radii:
            print(f"  加密层级 {target_level}，半径 {radius}m...")
            for point in path_points:
                self.refine_around_point(tuple(point), radius, target_level)
    
    def refine_at_boundary(self, geometry_func, target_level):
        """
        在几何体的边界处加密
        """
        self._refine_boundary_recursive(self.root, geometry_func, target_level)
    
    def _refine_boundary_recursive(self, node, geometry_func, target_level):
        """递归实现边界加密"""
        if node.level >= target_level or node.level >= self.max_level:
            return
        
        corners = node.get_corners()
        inside_flags = [geometry_func(*c) for c in corners]
        n_inside = sum(inside_flags)
        
        if n_inside == 0 or n_inside == 8:
            return
        
        if node.is_leaf:
            node.subdivide()
        
        for child in node.children:
            self._refine_boundary_recursive(child, geometry_func, target_level)
    
    def balance(self):
        """
        2:1平衡约束
        """
        print("正在平衡网格...")
        iteration = 0
        while True:
            iteration += 1
            changed = self._balance_pass()
            if not changed:
                break
            if iteration > 100:
                print("  警告：平衡迭代超过100次")
                break
        print(f"  平衡完成，迭代了 {iteration} 次")
    
    def _balance_pass(self):
        """一轮平衡检查"""
        leaves = self.root.get_leaves()
        changed = False
        
        for leaf in leaves:
            if leaf.level >= self.max_level:
                continue
            
            neighbors = self._find_face_neighbors(leaf)
            
            for neighbor in neighbors:
                if neighbor.level > leaf.level + 1:
                    if leaf.is_leaf:
                        leaf.subdivide()
                        changed = True
                    break
        
        return changed
    
    def _find_face_neighbors(self, node):
        """
        找到一个方块的6个面邻居中的叶节点
        """
        neighbors = []
        dx, dy, dz = node.size
        cx, cy, cz = node.center
        
        offsets = [
            (dx, 0, 0), (-dx, 0, 0),
            (0, dy, 0), (0, -dy, 0),
            (0, 0, dz), (0, 0, -dz),
        ]
        
        for ox, oy, oz in offsets:
            nx_pos = cx + ox
            ny_pos = cy + oy
            nz_pos = cz + oz
            
            neighbor = self._find_leaf_at_point(self.root, nx_pos, ny_pos, nz_pos)
            if neighbor is not None and neighbor is not node:
                neighbors.append(neighbor)
        
        return neighbors
    
    def _find_leaf_at_point(self, node, x, y, z):
        """找到包含指定点的叶节点"""
        if not (node.x_min <= x <= node.x_max and
                node.y_min <= y <= node.y_max and
                node.z_min <= z <= node.z_max):
            return None
        
        if node.is_leaf:
            return node
        
        for child in node.children:
            result = self._find_leaf_at_point(child, x, y, z)
            if result is not None:
                return result
        
        return None
    
    def assign_property(self, geometry_func, prop_name, value):
        """给几何体内的所有叶节点赋物性值"""
        count = 0
        for leaf in self.root.get_leaves():
            cx, cy, cz = leaf.center
            if geometry_func(cx, cy, cz):
                setattr(leaf, prop_name, value)
                count += 1
        print(f"  已给 {count} 个格子赋 {prop_name}={value}")
    
    @property
    def n_cells(self):
        """总格子数"""
        return len(self.root.get_leaves())
    
    def get_info(self):
        """打印网格信息"""
        leaves = self.root.get_leaves()
        levels = [leaf.level for leaf in leaves]
        sizes = [leaf.min_edge for leaf in leaves]
        
        info = f"""
        ╔══════════════════════════════════════╗
        ║        八叉树网格信息                  ║
        ╠══════════════════════════════════════╣
        ║ 总格子数: {len(leaves)}
        ║ 最大细分层级: {max(levels)} / {self.max_level}
        ║ 层级分布:"""
        
        for lv in range(max(levels) + 1):
            count = levels.count(lv)
            if count > 0:
                size = self.root.size[0] / (2 ** lv)
                info += f"\n        ║   层级{lv}: {count}个格子 ({size:.1f}m)"
        
        info += f"""
        ║ 最小格子: {min(sizes):.1f} m
        ║ 最大格子: {max(sizes):.1f} m
        ╚══════════════════════════════════════╝
        """
        print(info)
    
    def to_arrays(self):
        """
        导出为numpy数组格式（供正演计算使用）
        """
        leaves = self.root.get_leaves()
        
        centers = np.array([leaf.center for leaf in leaves])
        sizes = np.array([leaf.size for leaf in leaves])
        density = np.array([leaf.density for leaf in leaves])
        susceptibility = np.array([leaf.susceptibility for leaf in leaves])
        resistivity = np.array([leaf.resistivity for leaf in leaves])
        velocity = np.array([leaf.velocity for leaf in leaves])
        
        return {
            'centers': centers,
            'sizes': sizes,
            'volumes': sizes[:, 0] * sizes[:, 1] * sizes[:, 2],
            'density': density,
            'susceptibility': susceptibility,
            'resistivity': resistivity,
            'velocity': velocity,
            'n_cells': len(leaves)
        }
    
    @staticmethod
    def _box_intersects_sphere(node, center, radius):
        """判断方块是否与球体相交"""
        px, py, pz = center
        cx = max(node.x_min, min(px, node.x_max))
        cy = max(node.y_min, min(py, node.y_max))
        cz = max(node.z_min, min(pz, node.z_max))
        dist_sq = (cx - px)**2 + (cy - py)**2 + (cz - pz)**2
        return dist_sq <= radius * radius
