# 算法模块开发指南

> 本指南面向算法研究员，假设你熟悉 Python 编程但不了解本系统的内部结构。
> 按照以下步骤操作，你可以将自己的算法封装为平台可调用的模块。

---

## 你需要做什么

把你的算法封装成 **一个文件夹**，里面放 **两个文件**：

```
algorithms/
└── 你的算法名/
    ├── manifest.json    ← 告诉系统：算法叫什么、需要什么输入、产出什么输出
    └── algorithm.py     ← 你的算法代码
```

放好后系统会自动识别。**不需要修改系统中任何其他文件。**

---

## 第一步：创建文件夹

在 `algorithms/` 目录下新建一个文件夹。

**命名规则：** `方法_类别_名称`，全部小写，用下划线连接。

| 部分 | 可选值                                    | 说明                               |
| ---- | ----------------------------------------- | ---------------------------------- |
| 方法 | `logging` / `tem` / `ert` / `ip` / `em`   | 你的算法属于哪种物探方法           |
| 类别 | `preproc` / `forward` / `inv` / `process` | 预处理 / 正演 / 反演 / 通用处理    |
| 名称 | 任意                                      | 算法特征，如 `savgol` / `occam_2d` |

**举例：**

- `logging_preproc_myfilter` — 测井曲线预处理
- `tem_forward_mymodel` — TEM 正演模拟
- `ert_inv_mymethod` — ERT 反演算法

---

## 第二步：写配置文件 manifest.json

manifest.json 告诉系统你的算法需要什么输入、产出什么输出。

### 直接复制下面的模板，修改标注了「改这里」的部分：

**模板 A：一维曲线处理（最常见）**

```json
{
  "name": "logging_preproc_myfilter",        ← 改这里：和文件夹名一致
  "display_name": "我的滤波算法",             ← 改这里：在界面上显示的名字
  "version": "1.0.0",
  "method": "logging",                        ← 改这里：方法（logging/tem/ert/ip/em）
  "category": "preprocess",                   ← 改这里：类别
  "dimension": "1d",
  "entry": {
    "type": "python",
    "module": "algorithm",
    "class": "LoggingPreprocMyfilter"         ← 改这里：你的类名（见第三步）
  },
  "inputs": [
    {
      "name": "curve",                        ← 代码里用 inputs["curve"] 获取
      "label": "输入曲线",
      "type": "ndarray_1d",
      "file": "GR.csv"                        ← 从 raw/ 文件夹读取这个文件
    },
    {
      "name": "window_size",                  ← 代码里用 inputs.get("window_size", 11) 获取
      "label": "窗口大小",
      "type": "int",
      "default": 11,
      "required": false
    }
  ],
  "outputs": [
    {
      "name": "result",                       ← 代码里返回 {"result": ...} 对应这个名字
      "label": "处理结果",
      "type": "ndarray_1d"
    },
    {
      "name": "quality",
      "label": "质量指标",
      "type": "float"
    }
  ]
}
```

**模板 B：正演模拟**

```json
{
  "name": "tem_forward_mymodel",
  "display_name": "我的TEM正演",
  "version": "1.0.0",
  "method": "tem",
  "category": "forward",
  "dimension": "1d",
  "entry": {
    "type": "python",
    "module": "algorithm",
    "class": "TemForwardMymodel"
  },
  "inputs": [
    {
      "name": "resistivity",
      "label": "各层电阻率 (Ω·m)",
      "type": "ndarray_1d",
      "file": "resistivity.csv"
    },
    {
      "name": "depths",
      "label": "层界面深度 (m)",
      "type": "ndarray_1d",
      "file": "depths.csv"
    },
    {
      "name": "frequency",
      "label": "频率 (Hz)",
      "type": "float",
      "default": 100.0,
      "required": false
    }
  ],
  "outputs": [
    {
      "name": "response",
      "label": "正演响应",
      "type": "ndarray_1d"
    },
    {
      "name": "misfit",
      "label": "拟合误差",
      "type": "float"
    }
  ]
}
```

**模板 C：反演算法**

```json
{
  "name": "ert_inv_mymethod",
  "display_name": "我的ERT反演",
  "version": "1.0.0",
  "method": "ert",
  "category": "inversion",
  "dimension": "2d",
  "entry": {
    "type": "python",
    "module": "algorithm",
    "class": "ErtInvMymethod"
  },
  "inputs": [
    {
      "name": "observed_data",
      "label": "观测数据",
      "type": "ndarray_2d",
      "file": "observed.csv"
    },
    {
      "name": "max_iter",
      "label": "最大迭代次数",
      "type": "int",
      "default": 20,
      "required": false
    },
    {
      "name": "reg_param",
      "label": "正则化参数",
      "type": "float",
      "default": 0.01,
      "required": false
    }
  ],
  "outputs": [
    {
      "name": "model",
      "label": "反演模型",
      "type": "ndarray_2d"
    },
    {
      "name": "misfit",
      "label": "拟合误差",
      "type": "float"
    }
  ]
}
```

### 输入输出类型对照表

| 你在 manifest 里写的 | Python 里是什么              | 文件格式 | 什么时候用         |
| -------------------- | ---------------------------- | -------- | ------------------ |
| `ndarray_1d`         | 一维数组 `np.array([1,2,3])` | .csv     | 曲线、时间序列     |
| `ndarray_2d`         | 二维数组（矩阵）             | .csv     | 剖面数据、电极坐标 |
| `ndarray_3d`         | 三维数组                     | .npz     | 三维模型           |
| `int`                | 整数                         | -        | 迭代次数、窗口大小 |
| `float`              | 小数                         | -        | 频率、正则化参数   |
| `str`                | 文字                         | -        | 文件路径、标签     |

---

## 第三步：写算法代码 algorithm.py

### 最简模板（直接复制，改标注的地方）：

```python
import numpy as np
from algorithms.base import AlgorithmResult, BaseAlgorithm


class LoggingPreprocMyfilter(BaseAlgorithm):        # ← 改这里：类名要和 manifest 里的 class 一致
    def run(self, **inputs):

        # ===== 第1步：获取输入 =====
        curve = inputs["curve"]                       # 从 manifest 的 inputs 里取（名字要对应）
        window_size = int(inputs.get("window_size", 11))  # 可选参数，给默认值

        # ===== 第2步：你的算法 =====
        #
        # 把你的算法代码写在这里
        # curve 就是一个普通的 numpy 数组，可以直接做运算
        #
        result = curve * 2  # ← 替换成你的实际算法
        quality = 0.95      # ← 替换成你的实际指标

        # ===== 第3步：返回结果 =====
        return AlgorithmResult(
            outputs={
                "result": result,        # ← 名字要和 manifest 的 outputs 对应
                "quality": quality,
            },
        )
```

### 进阶功能（可选）

**报告进度**（适合运行时间长的算法）：

```python
for i in range(100):
    self.report_progress((i + 1) / 100, f"正在计算第 {i+1} 步...")
    # 你的计算代码
```

**返回警告信息**（不影响结果，但提醒用户注意）：

```python
warnings = []
if some_condition:
    warnings.append("数据中存在异常值，已自动处理。")

return AlgorithmResult(
    outputs={"result": result},
    warnings=warnings,
)
```

**输入参数验证**（检查输入是否合理）：

```python
if len(curve) < window_size:
    raise ValueError(
        f"曲线长度 ({len(curve)}) 小于窗口大小 ({window_size})，"
        f"请减小窗口大小或提供更长的数据。"
    )
```

---

## 第四步：验证和测试

### 方法一：双击验证（Windows）

双击 `tools/verify.bat`，输入你的算法文件夹路径，或将文件夹拖到 `verify.bat` 上。

### 方法二：命令行验证

打开命令提示符（CMD），进入项目目录，运行：

```
python tools\validate_algo.py algorithms\你的算法名
```

看到全部 `[OK]` 就说明配置正确。如果有 `[ERROR]`，按照提示修正。

### 方法三：在软件中测试

启动软件后：

1. 点击工具栏「新建钻孔」创建测试钻孔
2. 点击「导入数据」导入你的测试数据
3. 在右侧算法列表中找到你的算法
4. 调整参数，点击「运行算法」

---

## 神经网络 / 深度学习算法封装

如果你的反演算法使用了 PyTorch、TensorFlow 或 scikit-learn 等机器学习框架，封装方式与普通算法完全一样——只是多了一个「加载模型权重」的步骤。

### 你需要提交什么

```
algorithms/你的算法名/
├── manifest.json          ← 和普通算法一样
├── algorithm.py           ← 算法代码（加载模型 + 推理）
└── weights/               ← 新增：模型权重文件夹
    ├── model.pt           ← PyTorch 模型权重（或 .onnx / .joblib）
    └── scaler.joblib      ← 数据标准化器（如果有）
```

### 模板：PyTorch 模型

```python
import numpy as np
import torch
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


class MyUNetInversion(BaseAlgorithm):
    def run(self, **inputs):
        observed_data = inputs["observed_data"]   # numpy 数组

        # ===== 加载模型 =====
        weight_dir = Path(__file__).parent / "weights"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(weight_dir / "model.pt", map_location=device)
        model.eval()

        # ===== 数据预处理 =====
        x = torch.from_numpy(observed_data).float().to(device)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # 添加 batch 维度

        # ===== 推理 =====
        self.report_progress(0.3, "Running inference...")
        with torch.no_grad():
            output = model(x)
        result = output.cpu().numpy()

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={"resistivity_model": result, "confidence": 0.95},
        )
```

### 模板：ONNX Runtime（推荐用于部署）

ONNX 的优势：不需要安装 PyTorch 或 TensorFlow，只需 `pip install onnxruntime`（约 50MB）。
PyTorch / TensorFlow / sklearn 训练的模型都可以导出为 ONNX 格式。

```python
import numpy as np
import onnxruntime as ort
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


class OnnxInversion(BaseAlgorithm):
    def run(self, **inputs):
        observed_data = np.atleast_2d(inputs["observed_data"]).astype(np.float32)

        # 加载 ONNX 模型
        model_path = Path(__file__).parent / "weights" / "model.onnx"
        session = ort.InferenceSession(str(model_path))

        # 推理（一行代码）
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: observed_data})[0]

        return AlgorithmResult(
            outputs={"resistivity_model": result},
        )
```

### 模板：sklearn MLPRegressor

```python
import numpy as np
import joblib
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


class SklearnMlpInversion(BaseAlgorithm):
    def run(self, **inputs):
        observed_data = np.atleast_2d(inputs["observed_data"])

        # 加载模型和标准化器
        weight_dir = Path(__file__).parent / "weights"
        model = joblib.load(weight_dir / "mlp_model.joblib")
        scaler_X = joblib.load(weight_dir / "scaler_X.joblib")
        scaler_y = joblib.load(weight_dir / "scaler_y.joblib")

        # 推理
        X_scaled = scaler_X.transform(observed_data)
        y_scaled = model.predict(X_scaled)
        result = scaler_y.inverse_transform(y_scaled)

        return AlgorithmResult(
            outputs={"resistivity_model": result},
        )
```

### manifest.json 示例（神经网络反演）

```json
{
  "name": "tem_inv_unet_2d",
  "display_name": "U-Net TEM 二维智能反演",
  "version": "1.0.0",
  "method": "tem",
  "category": "inversion",
  "dimension": "2d",
  "entry": {
    "type": "python",
    "module": "algorithm",
    "class": "MyUNetInversion"
  },
  "inputs": [
    {
      "name": "observed_data",
      "label": "TEM 观测数据",
      "type": "ndarray_2d",
      "file": "observed.csv"
    }
  ],
  "outputs": [
    {
      "name": "resistivity_model",
      "label": "反演电阻率模型",
      "type": "ndarray_2d"
    },
    {
      "name": "confidence",
      "label": "预测置信度",
      "type": "float"
    }
  ]
}
```

### 如何导出 ONNX 模型

**从 PyTorch 导出：**

```python
import torch
model = ...  # 你训练好的模型
dummy_input = torch.randn(1, 30)  # 示例输入（batch=1, features=30）
torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])
```

**从 sklearn 导出：**

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onnx_model = convert_sklearn(pipeline, initial_types=[("input", FloatTensorType([None, 30]))])
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### 需要安装的额外依赖

| 框架         | 安装命令                  | 说明                     |
| ------------ | ------------------------- | ------------------------ |
| PyTorch      | `pip install torch`       | GPU 版本需指定 CUDA 版本 |
| ONNX Runtime | `pip install onnxruntime` | 推荐：轻量、跨框架       |
| sklearn      | 已预装                    | 无需额外安装             |
| TensorFlow   | `pip install tensorflow`  | 可选                     |

### 提交注意事项

- [ ] `weights/` 文件夹中包含所有模型文件
- [ ] 如果使用 GPU，确保算法中有 CPU 回退（`map_location="cpu"`）
- [ ] 模型权重文件可能很大（几十 MB），确认文件完整
- [ ] 推荐使用 ONNX 部署——不依赖特定框架，体积更小

---

## 实战案例：三种典型神经网络算法封装

以下是三个已验证通过的真实地球物理算法封装案例，你可以直接参考对应场景。

### 案例 1：CNN U-Net 井间声波 CT 智能去噪

**场景**：输入含噪声的声波 CT 层析图像（二维矩阵），输出去噪后的速度模型。

**文件结构**：

```
algorithms/tem_preproc_unet_denoise/
├── manifest.json
├── algorithm.py
└── weights/
    ├── unet.pt              ← PyTorch 模型权重（U-Net state_dict）
    └── norm_params.npz      ← 训练时的归一化参数（最小值、最大值）
```

**manifest.json**：

```json
{
  "name": "tem_preproc_unet_denoise",
  "display_name": "U-Net 井间声波CT智能去噪",
  "version": "1.0.0",
  "method": "tem",
  "category": "preprocess",
  "dimension": "2d",
  "entry": {
    "type": "python",
    "module": "algorithm",
    "class": "UNetCTDenoise"
  },
  "inputs": [
    {
      "name": "noisy_image",
      "label": "含噪声波CT图像",
      "type": "ndarray_2d",
      "file": "ct_noisy.csv"
    }
  ],
  "outputs": [
    { "name": "denoised_image", "label": "去噪后图像", "type": "ndarray_2d" },
    { "name": "snr_improvement", "label": "信噪比改善 (dB)", "type": "float" }
  ]
}
```

**algorithm.py**（关键部分）：

```python
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


# ===== 把你的网络定义复制到这里 =====
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 16)
        self.enc2 = DoubleConv(16, 32)
        self.bottleneck = DoubleConv(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.out_conv = nn.Conv2d(16, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ===== 封装类（只需改这里的加载和推理逻辑）=====
class UNetCTDenoise(BaseAlgorithm):
    def run(self, **inputs):
        noisy = np.asarray(inputs["noisy_image"], dtype=np.float32)
        weight_dir = Path(__file__).parent / "weights"

        # 1. 加载归一化参数
        norm = np.load(weight_dir / "norm_params.npz")
        x_min, x_max = float(norm["x_min"]), float(norm["x_max"])

        # 2. 加载模型权重
        model = UNet(1, 1)
        model.load_state_dict(torch.load(
            weight_dir / "unet.pt", map_location="cpu", weights_only=True
        ))
        model.eval()

        self.report_progress(0.3, "Model loaded")

        # 3. 归一化 → 推理 → 反归一化
        x = (noisy - x_min) / (x_max - x_min + 1e-8)
        x_tensor = torch.from_numpy(x[None, None]).float()  # 加 batch 和 channel 维度
        with torch.no_grad():
            y_tensor = model(x_tensor)
        denoised_norm = y_tensor.squeeze().numpy()
        denoised = denoised_norm * (x_max - x_min) + x_min

        # 4. 计算信噪比
        noise_power = np.mean((noisy - denoised) ** 2)
        signal_power = np.mean(denoised ** 2)
        snr_db = float(10 * np.log10(signal_power / max(noise_power, 1e-12)))

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={"denoised_image": denoised, "snr_improvement": snr_db},
        )
```

**训练时如何保存权重**（在你自己的训练脚本中）：

```python
# 训练完成后保存
torch.save(model.state_dict(), "weights/unet.pt")
np.savez("weights/norm_params.npz", x_min=X_train.min(), x_max=X_train.max())
```

---

### 案例 2：GWO-SVM 测井岩性识别与评价

**场景**：输入多条测井曲线（GR、SP、COND、DEN），输出每个深度点的岩性分类标签和概率。

**文件结构**：

```
algorithms/log_lithology_gwo_svm/
├── manifest.json
├── algorithm.py
└── weights/
    ├── svm_model.joblib     ← 训练好的 SVM 模型
    ├── scaler.joblib        ← 输入数据标准化器
    └── labels.json          ← 岩性名称列表 ["sandstone", "shale", ...]
```

**manifest.json**：

```json
{
  "name": "log_lithology_gwo_svm",
  "display_name": "GWO-SVM 测井岩性识别",
  "version": "1.0.0",
  "method": "logging",
  "category": "process",
  "dimension": "1d",
  "entry": {
    "type": "python",
    "module": "algorithm",
    "class": "GwoSvmLithology"
  },
  "inputs": [
    {
      "name": "log_data",
      "label": "多曲线测井数据 (n_samples, 4): GR/SP/COND/DEN",
      "type": "ndarray_2d",
      "file": "log_multi.csv"
    }
  ],
  "outputs": [
    {
      "name": "lithology_labels",
      "label": "岩性标签 (整数编码)",
      "type": "ndarray_1d"
    },
    { "name": "probabilities", "label": "分类概率", "type": "ndarray_2d" },
    { "name": "accuracy", "label": "平均置信度", "type": "float" }
  ]
}
```

**algorithm.py**：

```python
import json
import numpy as np
import joblib
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


class GwoSvmLithology(BaseAlgorithm):
    def run(self, **inputs):
        log_data = np.atleast_2d(inputs["log_data"])
        weight_dir = Path(__file__).parent / "weights"

        # 加载模型
        model = joblib.load(weight_dir / "svm_model.joblib")
        scaler = joblib.load(weight_dir / "scaler.joblib")
        labels = json.load(open(weight_dir / "labels.json"))

        self.report_progress(0.3, "Classifying...")

        # 标准化 → 预测
        X_scaled = scaler.transform(log_data)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # 置信度 = 每个样本最高概率的平均值
        avg_confidence = float(np.mean(np.max(probabilities, axis=1)))

        self.report_progress(1.0, f"Classified {len(predictions)} samples")
        return AlgorithmResult(
            outputs={
                "lithology_labels": predictions.astype(float),
                "probabilities": probabilities,
                "accuracy": avg_confidence,
            },
        )
```

**训练时如何保存**（在你自己的训练脚本中）：

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib, json

# 1. GWO 调参（灰狼优化搜索最佳 C 和 gamma）
best_C, best_gamma = gwo_optimize(X_train, y_train)

# 2. 用最佳参数训练
scaler = StandardScaler().fit(X_train)
svm = SVC(C=best_C, gamma=best_gamma, kernel="rbf", probability=True)
svm.fit(scaler.transform(X_train), y_train)

# 3. 保存
joblib.dump(svm, "weights/svm_model.joblib")
joblib.dump(scaler, "weights/scaler.joblib")
json.dump(["sandstone", "shale", "limestone", "coal"], open("weights/labels.json", "w"))
```

---

### 案例 3：Attention U-Net 井地电阻率智能反演成像

**场景**：输入视电阻率观测数据（一维或多组），输出二维电阻率断面图像。

**文件结构**：

```
algorithms/ert_inv_attn_unet/
├── manifest.json
├── algorithm.py
└── weights/
    ├── attn_unet.pt         ← Attention U-Net 权重
    ├── scaler_x.joblib      ← 输入标准化器
    ├── scaler_y.joblib      ← 输出标准化器
    └── config.json          ← 网络结构参数 {"n_obs": 40, "n_model": 80}
```

**manifest.json**：

```json
{
  "name": "ert_inv_attn_unet",
  "display_name": "Attention U-Net 井地电阻率智能反演",
  "version": "1.0.0",
  "method": "ert",
  "category": "inversion",
  "dimension": "2d",
  "entry": {
    "type": "python",
    "module": "algorithm",
    "class": "AttnUNetErtInversion"
  },
  "inputs": [
    {
      "name": "observed_data",
      "label": "视电阻率观测数据 (n_samples, n_obs)",
      "type": "ndarray_2d",
      "file": "ert_observed.csv"
    }
  ],
  "outputs": [
    {
      "name": "resistivity_section",
      "label": "电阻率断面 log10(Ohm.m)",
      "type": "ndarray_2d"
    },
    { "name": "mean_confidence", "label": "平均置信度", "type": "float" }
  ]
}
```

**algorithm.py**（关键部分）：

```python
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


# ===== 把你的 Attention U-Net 网络定义复制到这里 =====
class AttentionUNet1D(nn.Module):
    def __init__(self, n_obs, n_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256),
        )
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, n_model),
        )
    def forward(self, x):
        features = self.encoder(x)
        attn = self.attention(features)
        attended = features * torch.cat([attn, attn], dim=1)
        return self.decoder(attended)


# ===== 封装类 =====
class AttnUNetErtInversion(BaseAlgorithm):
    def run(self, **inputs):
        observed = np.atleast_2d(inputs["observed_data"]).astype(np.float32)
        weight_dir = Path(__file__).parent / "weights"

        # 1. 加载配置和模型
        config = json.load(open(weight_dir / "config.json"))
        scaler_x = joblib.load(weight_dir / "scaler_x.joblib")
        scaler_y = joblib.load(weight_dir / "scaler_y.joblib")

        model = AttentionUNet1D(config["n_obs"], config["n_model"])
        model.load_state_dict(torch.load(
            weight_dir / "attn_unet.pt", map_location="cpu", weights_only=True
        ))
        model.eval()

        self.report_progress(0.3, "Model loaded")

        # 2. 标准化 → 推理 → 反标准化
        X_scaled = scaler_x.transform(observed)
        with torch.no_grad():
            y_scaled = model(torch.from_numpy(X_scaled).float()).numpy()
        result = scaler_y.inverse_transform(y_scaled)

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={
                "resistivity_section": result,
                "mean_confidence": float(1.0 - np.mean(np.std(y_scaled, axis=0))),
            },
        )
```

**训练时如何保存**：

```python
# 训练完成后
torch.save(model.state_dict(), "weights/attn_unet.pt")
joblib.dump(scaler_x, "weights/scaler_x.joblib")
joblib.dump(scaler_y, "weights/scaler_y.joblib")
json.dump({"n_obs": 40, "n_model": 80}, open("weights/config.json", "w"))
```

---

### 封装要点总结

无论是哪种神经网络算法，封装步骤都是相同的：

| 步骤            | 你做什么                                              | 放在哪里          |
| --------------- | ----------------------------------------------------- | ----------------- |
| 1. 定义网络     | 把 `class MyNet(nn.Module)` 复制到 algorithm.py 中    | algorithm.py 顶部 |
| 2. 保存权重     | 训练完后 `torch.save(model.state_dict(), ...)`        | weights/ 文件夹   |
| 3. 保存标准化器 | `joblib.dump(scaler, ...)`                            | weights/ 文件夹   |
| 4. 写封装类     | 继承 BaseAlgorithm，在 run() 中加载模型并推理         | algorithm.py 中   |
| 5. 写 manifest  | 声明输入输出类型和文件名                              | manifest.json     |
| 6. 验证         | `python tools\validate_algo.py algorithms\你的算法名` | 命令行            |

**关键注意事项**：

- 网络定义（`class MyNet`）必须写在 algorithm.py 里，不能从外部 import
- `torch.load()` 务必加 `map_location="cpu"`，确保没有 GPU 也能运行
- 如果有标准化器（scaler），训练和推理时必须用**同一个** scaler
- `weights/` 文件夹和 algorithm.py 一起提交

---

## 常见问题

### 我的算法在软件里看不到

**原因：** manifest.json 中的 `inputs` 或 `outputs` 是空的 `[]`。

**解决：** 至少添加一个输入和一个输出。参考上面的模板。

### 运行时提示「文件未找到」

**原因：** 算法需要的输入文件不在钻孔的 `raw/` 目录中。

**解决：** 先通过软件的「导入数据」功能导入数据，或手动把文件复制到对应的 `raw/` 文件夹。

### 运行时提示「模块未找到」(ImportError)

**原因：** 你的算法用了额外的 Python 库（如 scipy），但系统没有安装。

**解决：** 在命令行运行 `pip install 库名`，例如 `pip install scipy`。

### 结果维度不匹配 (TypeError)

**原因：** manifest 里写的是 `ndarray_1d`，但算法返回了二维数组。

**解决：** 确保返回的数组维度和 manifest 声明一致。一维用 `.ravel()` 展平。

### 验证工具报错「Output keys not in AlgorithmResult」

**原因：** `AlgorithmResult(outputs={...})` 里的 key 名字和 manifest `outputs` 里的 `name` 不一致。

**解决：** 逐一检查 outputs 字典的 key，确保和 manifest 里写的完全相同。

---

## 提交清单

提交你的算法前，确认以下几点：

- [ ] 文件夹名和 manifest 里的 `name` 完全一致
- [ ] `entry.class` 和 algorithm.py 里的类名完全一致
- [ ] 所有输入都在 manifest 的 `inputs` 里声明了
- [ ] 所有输出都在 manifest 的 `outputs` 里声明了，且 key 名字一一对应
- [ ] 运行 `python tools\validate_algo.py` 全部 `[OK]`
- [ ] 在软件中实际运行过一次，结果正确

只需要提交 **两个文件**：`manifest.json` 和 `algorithm.py`。不要提交 `__pycache__` 文件夹。
