"""
命令模式（Command Pattern）—— 撤销/重做的基础

每一个"用户操作"都封装成一个命令对象
每个命令知道如何 execute（执行）和 undo（撤销）

历史栈：
  ┌─────────────────────────────────────┐
  │  undo_stack（已执行的命令）            │
  │  ┌─────┬─────┬─────┬─────┐          │
  │  │添加 │添加  │移动 │编辑 │ ← 最新    │
  │  │矿体1│矿体2 │矿体1│密度 │           │
  │  └─────┴─────┴─────┴─────┘          │
  │                                      │
  │  Ctrl+Z → 把最新命令移到 redo_stack  │
  │  Ctrl+Y → 把最新的redo移回来         │
  │                                      │
  │  redo_stack（已撤销的命令）            │
  │  ┌─────┐                             │
  │  │编辑 │                              │
  │  │密度 │                              │
  │  └─────┘                             │
  └─────────────────────────────────────┘
"""

import copy


class CommandHistory:
    """
    命令历史管理器
    
    使用方法：
        history = CommandHistory()
        history.execute(some_command)    # 执行并记录
        history.undo()                  # 撤销
        history.redo()                  # 重做
    """
    
    def __init__(self, max_size=100):
        self.undo_stack = []    # 可撤销的命令
        self.redo_stack = []    # 可重做的命令
        self.max_size = max_size
    
    def execute(self, command):
        """执行一个命令并记录"""
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()   # 新操作清空重做栈
        
        # 限制栈大小
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
    
    def undo(self):
        """撤销最近一个命令"""
        if not self.undo_stack:
            return False, "没有可撤销的操作"
        
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        return True, f"已撤销: {command.description}"
    
    def redo(self):
        """重做最近撤销的命令"""
        if not self.redo_stack:
            return False, "没有可重做的操作"
        
        command = self.redo_stack.pop()
        command.execute()
        self.undo_stack.append(command)
        return True, f"已重做: {command.description}"
    
    @property
    def can_undo(self):
        return len(self.undo_stack) > 0
    
    @property
    def can_redo(self):
        return len(self.redo_stack) > 0
    
    @property
    def undo_description(self):
        if self.undo_stack:
            return self.undo_stack[-1].description
        return ""
    
    @property
    def redo_description(self):
        if self.redo_stack:
            return self.redo_stack[-1].description
        return ""


# ===================================================================
#  具体命令类
# ===================================================================

class AddBodyCommand:
    """添加地质体"""
    
    def __init__(self, manager, body):
        self.manager = manager
        self.body = body
        self.description = f"添加 '{body.name}'"
    
    def execute(self):
        self.manager.add_body(self.body)
    
    def undo(self):
        # 找到这个体并删除
        if self.body in self.manager.bodies:
            self.manager.bodies.remove(self.body)


class DeleteBodyCommand:
    """删除地质体"""
    
    def __init__(self, manager, index):
        self.manager = manager
        self.index = index
        self.body = manager.bodies[index]
        self.description = f"删除 '{self.body.name}'"
    
    def execute(self):
        if self.body in self.manager.bodies:
            self.manager.bodies.remove(self.body)
    
    def undo(self):
        self.manager.bodies.insert(self.index, self.body)


class MoveBodyCommand:
    """移动地质体"""
    
    def __init__(self, manager, index, dx, dz):
        self.manager = manager
        self.index = index
        self.dx = dx
        self.dz = dz
        body = manager.bodies[index]
        self.description = f"移动 '{body.name}' ({dx:.0f}, {dz:.0f})"
    
    def execute(self):
        body = self.manager.bodies[self.index]
        body.vertices_xz = [
            (x + self.dx, z + self.dz)
            for x, z in body.vertices_xz
        ]
    
    def undo(self):
        body = self.manager.bodies[self.index]
        body.vertices_xz = [
            (x - self.dx, z - self.dz)
            for x, z in body.vertices_xz
        ]


class MoveVertexCommand:
    """移动单个顶点"""
    
    def __init__(self, manager, body_index, vertex_index, new_x, new_z):
        self.manager = manager
        self.body_index = body_index
        self.vertex_index = vertex_index
        self.new_x = new_x
        self.new_z = new_z
        body = manager.bodies[body_index]
        self.old_x, self.old_z = body.vertices_xz[vertex_index]
        self.description = f"编辑 '{body.name}' 顶点{vertex_index}"
    
    def execute(self):
        body = self.manager.bodies[self.body_index]
        body.vertices_xz[self.vertex_index] = (self.new_x, self.new_z)
    
    def undo(self):
        body = self.manager.bodies[self.body_index]
        body.vertices_xz[self.vertex_index] = (self.old_x, self.old_z)


class EditPropertyCommand:
    """编辑物性"""
    
    def __init__(self, manager, index, prop_name, new_value):
        self.manager = manager
        self.index = index
        self.prop_name = prop_name
        self.new_value = new_value
        body = manager.bodies[index]
        self.old_value = getattr(body, prop_name)
        self.description = f"修改 '{body.name}' {prop_name}={new_value}"
    
    def execute(self):
        body = self.manager.bodies[self.index]
        setattr(body, self.prop_name, self.new_value)
        if hasattr(body, '_sync_derived_properties'):
            body._sync_derived_properties(explicit_conductivity=self.prop_name == 'conductivity')
    
    def undo(self):
        body = self.manager.bodies[self.index]
        setattr(body, self.prop_name, self.old_value)
        if hasattr(body, '_sync_derived_properties'):
            body._sync_derived_properties(explicit_conductivity=self.prop_name == 'conductivity')


class AddLayerCommand:
    """添加地层线"""
    
    def __init__(self, manager, layer):
        self.manager = manager
        self.layer = layer
        self.description = f"添加地层线 '{layer.name}'"
    
    def execute(self):
        if not hasattr(self.manager, 'layers'):
            self.manager.layers = []
        self.manager.layers.append(self.layer)
    
    def undo(self):
        if hasattr(self.manager, 'layers') and self.layer in self.manager.layers:
            self.manager.layers.remove(self.layer)


class PasteBodyCommand:
    """粘贴（复制的地质体带偏移）"""
    
    def __init__(self, manager, body, offset_x=50, offset_z=50):
        self.manager = manager
        self.new_body = copy.deepcopy(body)
        self.new_body.name = body.name + "_copy"
        self.new_body.vertices_xz = [
            (x + offset_x, z + offset_z)
            for x, z in body.vertices_xz
        ]
        self.description = f"粘贴 '{body.name}'"
    
    def execute(self):
        self.manager.add_body(self.new_body)
    
    def undo(self):
        if self.new_body in self.manager.bodies:
            self.manager.bodies.remove(self.new_body)
