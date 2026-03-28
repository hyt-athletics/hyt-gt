"""
全局设置

所有可配置的参数集中在这里
"""

import json
import os


DEFAULT_SETTINGS = {
    # 模型域
    'domain': {
        'x_range': [0, 1000],
        'y_range': [0, 1000],
        'z_range': [0, 1000],
    },
    
    # 钻孔
    'well': {
        'x': 500,
        'y': 500,
        'z_top': 10,
        'z_bottom': 800,
        'n_stations': 40,
    },
    
    # 网格
    'mesh': {
        'type': 'octree',                # octree / 25d / tet
        'max_level': 5,                  # 八叉树最大层级
        'borehole_refine_levels': [      # 钻孔加密配置
            [5, 50],                     # [层级, 半径]
            [4, 150],
            [3, 300],
        ],
        'boundary_refine_level': 4,      # 异常体边界加密层级
    },
    
    # 正演
    'forward': {
        'engine': 'auto',                # auto / numba / numpy / harmonica / builtin
        'method': 'gravity',             # gravity / magnetic / resistivity / seismic
    },
    
    # 显示
    'display': {
        'grid_snap': False,
        'grid_spacing': 50,
        'show_coordinates': True,
    },
    
    # 导出
    'export': {
        'vtk_output_dir': 'vtk_output',
        'model_file': 'model.json',
    },
}


class Settings:
    """
    全局设置管理器
    
    使用：
        settings = Settings()
        settings.get('well.x')            → 500
        settings.set('well.x', 600)
        settings.save()
    """
    
    def __init__(self, filename='settings.json'):
        self.filename = filename
        self.data = {}
        self._load_defaults()
        
        if os.path.exists(filename):
            self._load_from_file()
    
    def _load_defaults(self):
        """加载默认值"""
        import copy
        self.data = copy.deepcopy(DEFAULT_SETTINGS)
    
    def _load_from_file(self):
        """从文件加载（覆盖默认值）"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            self._merge(self.data, saved)
        except Exception as e:
            print(f"  ⚠️ 加载设置失败: {e}")
    
    def _merge(self, base, override):
        """递归合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path, default=None):
        """
        获取设置值
        
        支持点号分隔的路径：
            settings.get('well.x') → 500
            settings.get('mesh.max_level') → 5
        """
        keys = key_path.split('.')
        val = self.data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
    
    def set(self, key_path, value):
        """设置值"""
        keys = key_path.split('.')
        d = self.data
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    
    def save(self):
        """保存到文件"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"  ✅ 设置已保存: {self.filename}")
    
    def print_all(self):
        """打印所有设置"""
        print("\n  ┌─ 当前设置 ──────────────────────────────┐")
        self._print_dict(self.data, indent=2)
        print("  └──────────────────────────────────────────┘")
    
    def _print_dict(self, d, indent=0):
        prefix = "  │" + " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                self._print_dict(v, indent + 2)
            else:
                print(f"{prefix}{k}: {v}")