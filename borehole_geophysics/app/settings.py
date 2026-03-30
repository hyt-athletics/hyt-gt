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
        'strategy': 'auto',              # auto / manual
        'max_level': 5,                  # 八叉树最大层级
        'target_cell_size': None,        # 目标最小格子尺寸（m），None=自动估计
        'borehole_refine_levels': [      # 钻孔加密配置
            [5, 50],                     # [层级, 半径]
            [4, 150],
            [3, 300],
        ],
        'boundary_refine_level': 4,      # 异常体边界加密层级
        'body_core_refine_level': None,  # 异常体内部加密层级，None=自动
        'body_padding_refine_level': None, # 异常体外围包络加密层级，None=自动
        'body_padding_distance': None,   # 异常体外围加密距离，None=自动
        'active_model': {
            'mode': 'auto',              # auto / manual
            'bounds': {
                'x_range': [250, 750],
                'y_range': [250, 750],
                'z_range': [0, 900],
            },
        },
    },
    
    # 正演
    'forward': {
        'engine': 'auto',                # auto / numba / numpy / harmonica / builtin
        'method': 'gravity',             # gravity / magnetic_3c / ...
    },
    
    # 磁场参数
    'magnetic': {
        'b0_strength': 52000.0,          # nT
        'b0_inclination': 55.0,          # degree
        'b0_declination': -6.0,          # degree
    },

    # 反演
    'inversion': {
        'gravity': {
            'preset': 'balanced',        # balanced / compact_body / wide_search
            'max_iter': 5,
            'relative_error': 0.03,
            'noise_floor': None,
            'lower_bound': -1.5,         # g/cc
            'upper_bound': 1.5,          # g/cc
            'regularization': {
                'mode': 'smooth',        # smooth / compact
                'reference_model': 'property',  # property / zero
                'reference_model_in_smooth': False,
                'alpha_s': 1.0,
                'alpha_x': 1.0,
                'alpha_y': 1.0,
                'alpha_z': 1.0,
                'norms': [0.0, 1.0, 1.0, 1.0],
            },
            'update_region': {
                'mode': 'all_active',    # all_active / property_only / manual
                'bounds': {
                    'x_range': [300, 700],
                    'y_range': [300, 700],
                    'z_range': [50, 850],
                },
            },
        },
        'magnetic_3c': {
            'preset': 'balanced',        # balanced / compact_body / wide_search
            'max_iter': 8,
            'lower_bound': 0.0,          # SI susceptibility
            'upper_bound': 0.08,         # SI susceptibility
            'regularization': {
                'mode': 'smooth',        # smooth / compact
                'reference_model': 'property',  # property / zero
                'reference_model_in_smooth': False,
                'alpha_s': 1.0,
                'alpha_x': 1.0,
                'alpha_y': 1.0,
                'alpha_z': 1.0,
                'norms': [0.0, 1.0, 1.0, 1.0],
            },
            'update_region': {
                'mode': 'all_active',    # all_active / property_only / manual
                'bounds': {
                    'x_range': [300, 700],
                    'y_range': [300, 700],
                    'z_range': [50, 850],
                },
            },
        },
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
