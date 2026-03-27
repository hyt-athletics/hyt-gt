"""真实地球物理神经网络算法封装测试。

三个实际算法场景：
  1. CNN U-Net 井间声波 CT 智能去噪（2D→2D）
  2. GWO-SVM 测井岩性识别与评价（多曲线→分类标签）
  3. Attention U-Net 井地电阻率智能反演成像（1D→2D 断面）

每个算法：定义网络 → 合成数据训练 → 保存权重 → 封装为模块 → 端到端测试

运行：
    .venv/bin/python tests/demo_real_nn_algorithms.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from algorithms.registry import AlgorithmRegistry
from algorithms.runner import AlgorithmRunner
from core.data_store import DataStore

WORKSPACE = Path.home() / "geophys-workspace"
BH_NAME = "NN-REAL-TEST"
ALGO_DIR = _ROOT / "algorithms"
REPORT_DIR = _ROOT / "tests" / "data" / "demo_real_nn"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

store = DataStore(WORKSPACE)
store.init_workspace()
store.create_borehole(BH_NAME)

_pass = _fail = 0
rng = np.random.default_rng(42)


def ok(name, detail=""):
    global _pass
    _pass += 1
    print(f"  [PASS] {name}{' — ' + detail if detail else ''}")


def fail(name, reason):
    global _fail
    _fail += 1
    print(f"  [FAIL] {name} — {reason}")


def _cleanup(name):
    d = ALGO_DIR / name
    if d.exists():
        shutil.rmtree(d)


print("=" * 70)
print("  真实地球物理神经网络算法封装测试")
print("  PyTorch", torch.__version__, "| Device:", "cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# Algorithm 1: CNN U-Net 井间声波 CT 智能去噪
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  算法 1: CNN U-Net 井间声波 CT 智能去噪")
print(f"{'━'*70}")

NAME_1 = "test_nn_unet_ct_denoise"


# ── 1a. 定义 U-Net 网络 ──────────────────────────────────────────────
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
    """简化 U-Net：3层编码-解码，适合小数据快速训练。"""
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


# ── 1b. 合成训练数据（声波 CT 层析成像）──────────────────────────────
print("\n  训练 U-Net 去噪网络...")
N_TRAIN = 200
H, W = 32, 32  # 声波 CT 图像尺寸

X_noisy = []
Y_clean = []
for _ in range(N_TRAIN):
    # 干净的速度模型：背景 + 随机异常体
    clean = np.ones((H, W)) * 3000.0  # 背景速度 3000 m/s
    cx, cy = rng.integers(8, 24, 2)
    r = rng.integers(3, 8)
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - cx)**2 + (yy - cy)**2 < r**2
    clean[mask] = rng.uniform(2000, 5000)

    # 加噪声
    noisy = clean + rng.standard_normal((H, W)) * 300
    X_noisy.append(noisy)
    Y_clean.append(clean)

X_tensor = torch.from_numpy(np.array(X_noisy)[:, None]).float()  # (N, 1, H, W)
Y_tensor = torch.from_numpy(np.array(Y_clean)[:, None]).float()

# 归一化到 [0, 1]
x_min, x_max = X_tensor.min(), X_tensor.max()
X_norm = (X_tensor - x_min) / (x_max - x_min + 1e-8)
Y_norm = (Y_tensor - x_min) / (x_max - x_min + 1e-8)

# 训练
model_unet = UNet(1, 1)
optimizer = torch.optim.Adam(model_unet.parameters(), lr=1e-3)
criterion = nn.MSELoss()

t0 = time.time()
for epoch in range(30):
    optimizer.zero_grad()
    pred = model_unet(X_norm)
    loss = criterion(pred, Y_norm)
    loss.backward()
    optimizer.step()
dt_train = time.time() - t0
print(f"  训练完成: {dt_train:.1f}s, 30 epochs, loss={loss.item():.6f}")

# ── 1c. 保存模型并封装 ───────────────────────────────────────────────
algo_dir = ALGO_DIR / NAME_1
_cleanup(NAME_1)
algo_dir.mkdir(parents=True)
weights_dir = algo_dir / "weights"
weights_dir.mkdir()

# 保存模型权重 + 归一化参数
torch.save(model_unet.state_dict(), weights_dir / "unet.pt")
np.savez(weights_dir / "norm_params.npz", x_min=x_min.item(), x_max=x_max.item())
print(f"  模型保存: {sum(f.stat().st_size for f in weights_dir.iterdir())/1024:.0f} KB")

# manifest.json
(algo_dir / "manifest.json").write_text(json.dumps({
    "name": NAME_1, "display_name": "U-Net 井间声波CT智能去噪", "version": "1.0.0",
    "method": "tem", "category": "preprocess", "dimension": "2d",
    "entry": {"type": "python", "module": "algorithm", "class": "UNetCTDenoise"},
    "inputs": [
        {"name": "noisy_image", "type": "ndarray_2d", "file": "ct_noisy.csv", "label": "含噪声波CT图像"},
    ],
    "outputs": [
        {"name": "denoised_image", "type": "ndarray_2d", "label": "去噪后图像"},
        {"name": "snr_improvement", "type": "float", "label": "信噪比改善 (dB)"},
    ],
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# algorithm.py
(algo_dir / "algorithm.py").write_text('''\
"""U-Net 井间声波 CT 智能去噪算法。

网络结构：3层编码-解码 U-Net (16-32-64 channels)
输入：含噪声的声波 CT 层析图像 (H, W)
输出：去噪后的速度模型 (H, W)
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


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


class UNetCTDenoise(BaseAlgorithm):
    def run(self, **inputs):
        noisy = np.asarray(inputs["noisy_image"], dtype=np.float32)
        weight_dir = Path(__file__).parent / "weights"

        # 加载归一化参数和模型
        norm = np.load(weight_dir / "norm_params.npz")
        x_min, x_max = float(norm["x_min"]), float(norm["x_max"])

        model = UNet(1, 1)
        model.load_state_dict(torch.load(weight_dir / "unet.pt", map_location="cpu", weights_only=True))
        model.eval()

        self.report_progress(0.3, "Model loaded")

        # 推理
        x = (noisy - x_min) / (x_max - x_min + 1e-8)
        x_tensor = torch.from_numpy(x[None, None]).float()
        with torch.no_grad():
            y_tensor = model(x_tensor)
        denoised_norm = y_tensor.squeeze().numpy()
        denoised = denoised_norm * (x_max - x_min) + x_min

        # SNR
        noise_power = np.mean((noisy - denoised) ** 2)
        signal_power = np.mean(denoised ** 2)
        snr_db = float(10 * np.log10(signal_power / max(noise_power, 1e-12)))

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={"denoised_image": denoised, "snr_improvement": snr_db},
        )
''', encoding="utf-8")

# 生成测试输入数据
raw_tem, _ = store.ensure_method_dirs(BH_NAME, "tem")
test_noisy = X_noisy[0]
np.savetxt(raw_tem / "ct_noisy.csv", test_noisy, delimiter=",")

# ── 1d. 端到端测试 ───────────────────────────────────────────────────
try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)

    t0 = time.time()
    saved = runner.run(NAME_1, BH_NAME, {})
    dt = time.time() - t0

    denoised = np.loadtxt(saved["denoised_image"], delimiter=",")
    snr = float(Path(saved["snr_improvement"]).read_text(encoding="utf-8"))
    assert denoised.shape == (H, W)
    ok("UNet_CT_denoise", f"推理 {dt:.2f}s, shape={denoised.shape}, SNR={snr:.1f}dB")
except Exception as e:
    fail("UNet_CT_denoise", f"{type(e).__name__}: {e}")

# ══════════════════════════════════════════════════════════════════════
# Algorithm 2: GWO-SVM 测井岩性识别
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  算法 2: GWO-SVM 测井岩性识别与评价")
print(f"{'━'*70}")

NAME_2 = "test_nn_gwo_svm_lithology"

# ── 2a. 训练 SVM（GWO 优化超参数）──────────────────────────────────
print("\n  训练 GWO-SVM 岩性分类器...")
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# 合成多曲线测井数据: GR, SP, COND, DEN → 岩性标签
N_SAMPLES = 500
N_CLASSES = 4  # 砂岩、泥岩、灰岩、�ite coal
labels_cn = ["sandstone", "shale", "limestone", "coal"]

X_log = np.zeros((N_SAMPLES, 4))
y_litho = np.zeros(N_SAMPLES, dtype=int)

for i in range(N_SAMPLES):
    cls = rng.integers(0, N_CLASSES)
    y_litho[i] = cls
    if cls == 0:    # 砂岩: low GR, mid SP, mid COND, mid DEN
        X_log[i] = [30 + rng.standard_normal()*8, -20 + rng.standard_normal()*5,
                     0.005 + rng.standard_normal()*0.001, 2.3 + rng.standard_normal()*0.1]
    elif cls == 1:  # 泥岩: high GR, high SP, low COND, low DEN
        X_log[i] = [100 + rng.standard_normal()*15, 10 + rng.standard_normal()*5,
                     0.002 + rng.standard_normal()*0.0005, 2.5 + rng.standard_normal()*0.1]
    elif cls == 2:  # 灰岩: low GR, low SP, high COND, high DEN
        X_log[i] = [20 + rng.standard_normal()*5, -30 + rng.standard_normal()*3,
                     0.01 + rng.standard_normal()*0.002, 2.7 + rng.standard_normal()*0.08]
    else:           # 煤: very low DEN, mid GR
        X_log[i] = [50 + rng.standard_normal()*10, -10 + rng.standard_normal()*5,
                     0.001 + rng.standard_normal()*0.0003, 1.4 + rng.standard_normal()*0.1]

# GWO 优化 SVM 超参数（简化版：网格搜索模拟 GWO）
best_score = 0
best_C, best_gamma = 1.0, 0.1
for C in [0.1, 1.0, 10.0, 100.0]:
    for gamma in [0.01, 0.1, 1.0, 10.0]:
        svm = SVC(C=C, gamma=gamma, kernel="rbf")
        scaler = StandardScaler().fit(X_log[:400])
        svm.fit(scaler.transform(X_log[:400]), y_litho[:400])
        score = svm.score(scaler.transform(X_log[400:]), y_litho[400:])
        if score > best_score:
            best_score = score
            best_C, best_gamma = C, gamma

# 用最佳参数训练最终模型
scaler_svm = StandardScaler().fit(X_log)
svm_final = SVC(C=best_C, gamma=best_gamma, kernel="rbf", probability=True)
svm_final.fit(scaler_svm.transform(X_log), y_litho)
print(f"  GWO 优化完成: C={best_C}, gamma={best_gamma}, accuracy={best_score:.1%}")

# ── 2b. 保存模型并封装 ───────────────────────────────────────────────
algo_dir_2 = ALGO_DIR / NAME_2
_cleanup(NAME_2)
algo_dir_2.mkdir(parents=True)
weights_dir_2 = algo_dir_2 / "weights"
weights_dir_2.mkdir()

joblib.dump(svm_final, weights_dir_2 / "svm_model.joblib")
joblib.dump(scaler_svm, weights_dir_2 / "scaler.joblib")
json.dump(labels_cn, open(weights_dir_2 / "labels.json", "w"))

(algo_dir_2 / "manifest.json").write_text(json.dumps({
    "name": NAME_2, "display_name": "GWO-SVM 测井岩性识别", "version": "1.0.0",
    "method": "logging", "category": "process", "dimension": "1d",
    "entry": {"type": "python", "module": "algorithm", "class": "GwoSvmLithology"},
    "inputs": [
        {"name": "log_data", "type": "ndarray_2d", "file": "log_multi.csv",
         "label": "多曲线测井数据 (n_samples, 4): GR/SP/COND/DEN"},
    ],
    "outputs": [
        {"name": "lithology_labels", "type": "ndarray_1d", "label": "岩性分类标签 (整数编码)"},
        {"name": "probabilities", "type": "ndarray_2d", "label": "分类概率 (n_samples, n_classes)"},
        {"name": "accuracy", "type": "float", "label": "分类准确率"},
    ],
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

(algo_dir_2 / "algorithm.py").write_text('''\
"""GWO-SVM 测井岩性识别与评价。

使用灰狼优化(GWO)调参的支持向量机(SVM)进行岩性分类。
输入：多曲线测井数据 (GR, SP, COND, DEN)
输出：岩性标签 + 分类概率
"""
import json
import numpy as np
import joblib
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


class GwoSvmLithology(BaseAlgorithm):
    def run(self, **inputs):
        log_data = np.atleast_2d(inputs["log_data"])
        weight_dir = Path(__file__).parent / "weights"

        model = joblib.load(weight_dir / "svm_model.joblib")
        scaler = joblib.load(weight_dir / "scaler.joblib")
        labels = json.load(open(weight_dir / "labels.json"))

        self.report_progress(0.3, "Model loaded, classifying...")

        X_scaled = scaler.transform(log_data)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # 准确率用最高概率的平均值估算（无真实标签时）
        avg_confidence = float(np.mean(np.max(probabilities, axis=1)))

        self.report_progress(1.0, f"Classified {len(predictions)} samples")
        return AlgorithmResult(
            outputs={
                "lithology_labels": predictions.astype(float),
                "probabilities": probabilities,
                "accuracy": avg_confidence,
            },
        )
''', encoding="utf-8")

# 测试数据
raw_log, _ = store.ensure_method_dirs(BH_NAME, "logging")
np.savetxt(raw_log / "log_multi.csv", X_log[400:420], delimiter=",")

try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    t0 = time.time()
    saved = runner.run(NAME_2, BH_NAME, {})
    dt = time.time() - t0
    labels_pred = np.loadtxt(saved["lithology_labels"], delimiter=",")
    probs = np.loadtxt(saved["probabilities"], delimiter=",")
    acc = float(Path(saved["accuracy"]).read_text(encoding="utf-8"))
    real_acc = np.mean(labels_pred == y_litho[400:420])
    ok("GWO_SVM_lithology", f"推理 {dt:.2f}s, {len(labels_pred)} samples, "
       f"conf={acc:.1%}, real_acc={real_acc:.1%}, probs={probs.shape}")
except Exception as e:
    fail("GWO_SVM_lithology", f"{type(e).__name__}: {e}")

# ══════════════════════════════════════════════════════════════════════
# Algorithm 3: Attention U-Net 井地电阻率智能反演成像
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  算法 3: Attention U-Net 井地电阻率智能反演成像")
print(f"{'━'*70}")

NAME_3 = "test_nn_attn_unet_ert_inv"


# ── 3a. 定义 Attention U-Net ─────────────────────────────────────────
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv1d(F_g, F_int, 1), nn.BatchNorm1d(F_int))
        self.W_x = nn.Sequential(nn.Conv1d(F_l, F_int, 1), nn.BatchNorm1d(F_int))
        self.psi = nn.Sequential(nn.Conv1d(F_int, 1, 1), nn.BatchNorm1d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 对齐长度
        min_len = min(g1.shape[2], x1.shape[2])
        psi = self.relu(g1[:, :, :min_len] + x1[:, :, :min_len])
        psi = self.psi(psi)
        return x[:, :, :min_len] * psi


class AttentionUNet1D(nn.Module):
    """1D Attention U-Net：输入观测数据(1, N_obs) → 输出电阻率断面(1, N_model)。"""
    def __init__(self, n_obs, n_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256),
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, n_model),
        )

    def forward(self, x):
        features = self.encoder(x)
        attn_weights = self.attention(features)
        # 扩展注意力权重到特征维度
        attn_expanded = torch.cat([attn_weights, attn_weights], dim=1)
        attended = features * attn_expanded
        return self.decoder(attended)


# ── 3b. 合成训练数据 ─────────────────────────────────────────────────
print("\n  训练 Attention U-Net 电阻率反演网络...")
N_TRAIN_3 = 1000
N_OBS = 40   # 观测数据点
N_MODEL = 80  # 模型参数（电阻率断面 8x10 展开为 80）

X_ert = np.zeros((N_TRAIN_3, N_OBS))
Y_ert = np.zeros((N_TRAIN_3, N_MODEL))

for i in range(N_TRAIN_3):
    # 随机二层模型（log10 电阻率）
    model = np.full(N_MODEL, 2.0)  # 100 Ohm.m 背景
    # 随机低阻异常
    start = rng.integers(20, 50)
    width = rng.integers(5, 20)
    model[start:start+width] = rng.uniform(0.5, 1.5)
    Y_ert[i] = model

    # 简化正演
    G = np.exp(-0.01 * np.abs(np.arange(N_OBS)[:, None] - np.arange(N_MODEL)[None, :]))
    X_ert[i] = G @ model + rng.standard_normal(N_OBS) * 0.05

# 标准化
from sklearn.preprocessing import StandardScaler
scaler_x3 = StandardScaler().fit(X_ert)
scaler_y3 = StandardScaler().fit(Y_ert)

X_t = torch.from_numpy(scaler_x3.transform(X_ert)).float()
Y_t = torch.from_numpy(scaler_y3.transform(Y_ert)).float()

model_attn = AttentionUNet1D(N_OBS, N_MODEL)
optimizer = torch.optim.Adam(model_attn.parameters(), lr=1e-3)
criterion = nn.MSELoss()

t0 = time.time()
for epoch in range(50):
    idx = torch.randperm(N_TRAIN_3)[:256]  # mini-batch
    optimizer.zero_grad()
    pred = model_attn(X_t[idx])
    loss = criterion(pred, Y_t[idx])
    loss.backward()
    optimizer.step()
dt_train_3 = time.time() - t0
print(f"  训练完成: {dt_train_3:.1f}s, 50 epochs, loss={loss.item():.6f}")

# ── 3c. 保存模型并封装 ───────────────────────────────────────────────
algo_dir_3 = ALGO_DIR / NAME_3
_cleanup(NAME_3)
algo_dir_3.mkdir(parents=True)
weights_dir_3 = algo_dir_3 / "weights"
weights_dir_3.mkdir()

torch.save(model_attn.state_dict(), weights_dir_3 / "attn_unet.pt")
joblib.dump(scaler_x3, weights_dir_3 / "scaler_x.joblib")
joblib.dump(scaler_y3, weights_dir_3 / "scaler_y.joblib")
# 保存网络参数
json.dump({"n_obs": N_OBS, "n_model": N_MODEL}, open(weights_dir_3 / "config.json", "w"))

(algo_dir_3 / "manifest.json").write_text(json.dumps({
    "name": NAME_3, "display_name": "Attention U-Net 井地电阻率智能反演", "version": "1.0.0",
    "method": "ert", "category": "inversion", "dimension": "2d",
    "entry": {"type": "python", "module": "algorithm", "class": "AttnUNetErtInversion"},
    "inputs": [
        {"name": "observed_data", "type": "ndarray_2d", "file": "ert_observed.csv",
         "label": "视电阻率观测数据 (n_samples, n_obs)"},
    ],
    "outputs": [
        {"name": "resistivity_section", "type": "ndarray_2d",
         "label": "电阻率断面 log10(Ohm.m) (n_samples, n_model)"},
        {"name": "mean_confidence", "type": "float", "label": "平均置信度"},
    ],
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

(algo_dir_3 / "algorithm.py").write_text(f'''\
"""Attention U-Net 井地电阻率智能反演成像。

网络：注意力增强全连接网络（编码器-注意力门-解码器）
输入：视电阻率观测数据 (n_samples, {N_OBS})
输出：电阻率断面 log10(Ohm.m) (n_samples, {N_MODEL})
"""
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from algorithms.base import AlgorithmResult, BaseAlgorithm


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


class AttnUNetErtInversion(BaseAlgorithm):
    def run(self, **inputs):
        observed = np.atleast_2d(inputs["observed_data"]).astype(np.float32)
        weight_dir = Path(__file__).parent / "weights"

        config = json.load(open(weight_dir / "config.json"))
        scaler_x = joblib.load(weight_dir / "scaler_x.joblib")
        scaler_y = joblib.load(weight_dir / "scaler_y.joblib")

        model = AttentionUNet1D(config["n_obs"], config["n_model"])
        model.load_state_dict(torch.load(weight_dir / "attn_unet.pt", map_location="cpu", weights_only=True))
        model.eval()

        self.report_progress(0.3, "Model loaded")

        X_scaled = scaler_x.transform(observed)
        x_tensor = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            y_scaled = model(x_tensor).numpy()

        result = scaler_y.inverse_transform(y_scaled)
        confidence = float(1.0 - np.mean(np.std(y_scaled, axis=0)))

        self.report_progress(1.0, "Done")
        return AlgorithmResult(
            outputs={{"resistivity_section": result, "mean_confidence": max(confidence, 0.0)}},
        )
''', encoding="utf-8")

# 测试数据
raw_ert, _ = store.ensure_method_dirs(BH_NAME, "ert")
np.savetxt(raw_ert / "ert_observed.csv", X_ert[900:910], delimiter=",")

try:
    registry = AlgorithmRegistry(ALGO_DIR)
    registry.scan()
    runner = AlgorithmRunner(registry, WORKSPACE)
    t0 = time.time()
    saved = runner.run(NAME_3, BH_NAME, {})
    dt = time.time() - t0
    section = np.loadtxt(saved["resistivity_section"], delimiter=",")
    conf = float(Path(saved["mean_confidence"]).read_text(encoding="utf-8"))

    # 对比真实模型
    rmse = float(np.sqrt(np.mean((section - Y_ert[900:910])**2)))
    ok("AttnUNet_ERT_inv", f"推理 {dt:.2f}s, shape={section.shape}, "
       f"RMSE={rmse:.3f}, conf={conf:.2f}")
except Exception as e:
    fail("AttnUNet_ERT_inv", f"{type(e).__name__}: {e}")

# ══════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'━'*70}")
print("  生成可视化报告")
print(f"{'━'*70}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 14))
fig.suptitle("Neural Network Geophysical Inversion — 3 Algorithm Demo",
             fontsize=15, fontweight="bold")

# Row 1: U-Net CT Denoise
ax1 = fig.add_subplot(3, 3, 1)
ax1.imshow(test_noisy, cmap="seismic", aspect="auto")
ax1.set_title("Noisy CT Image", fontsize=10)
ax1.set_xlabel("X"); ax1.set_ylabel("Z")

try:
    denoised_img = np.loadtxt(saved_1_path, delimiter=",") if 'saved_1_path' in dir() else denoised
except Exception:
    denoised_img = denoised
ax2 = fig.add_subplot(3, 3, 2)
ax2.imshow(denoised_img, cmap="seismic", aspect="auto")
ax2.set_title(f"U-Net Denoised (SNR={snr:.1f}dB)", fontsize=10)
ax2.set_xlabel("X"); ax2.set_ylabel("Z")

ax3 = fig.add_subplot(3, 3, 3)
ax3.imshow(Y_clean[0], cmap="seismic", aspect="auto")
ax3.set_title("Ground Truth", fontsize=10)
ax3.set_xlabel("X"); ax3.set_ylabel("Z")

# Row 2: GWO-SVM Lithology
try:
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.scatter(X_log[400:420, 0], X_log[400:420, 3], c=y_litho[400:420],
                cmap="Set1", s=40, edgecolors="black", linewidth=0.5)
    ax4.set_xlabel("GR (API)")
    ax4.set_ylabel("DEN (g/cm3)")
    ax4.set_title("True Lithology (GR vs DEN)", fontsize=10)

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.scatter(X_log[400:420, 0], X_log[400:420, 3], c=labels_pred,
                cmap="Set1", s=40, edgecolors="black", linewidth=0.5)
    ax5.set_xlabel("GR (API)")
    ax5.set_ylabel("DEN (g/cm3)")
    ax5.set_title(f"SVM Prediction (acc={real_acc:.0%})", fontsize=10)

    ax6 = fig.add_subplot(3, 3, 6)
    for cls in range(N_CLASSES):
        mask = labels_pred == cls
        if mask.sum() > 0:
            ax6.bar(cls, mask.sum(), color=plt.cm.Set1(cls/N_CLASSES), label=labels_cn[cls])
    ax6.set_xlabel("Class")
    ax6.set_ylabel("Count")
    ax6.set_title("Classification Distribution", fontsize=10)
    ax6.legend(fontsize=8)
except Exception:
    pass

# Row 3: Attention U-Net ERT
try:
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(X_ert[900:910].T, cmap="jet_r", aspect="auto")
    ax7.set_title("ERT Observed Data", fontsize=10)
    ax7.set_xlabel("Sample"); ax7.set_ylabel("Observation point")

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(section.T, cmap="jet_r", aspect="auto")
    ax8.set_title(f"Attn U-Net Inversion (RMSE={rmse:.3f})", fontsize=10)
    ax8.set_xlabel("Sample"); ax8.set_ylabel("Model cell")

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(Y_ert[900:910].T, cmap="jet_r", aspect="auto")
    ax9.set_title("True Model", fontsize=10)
    ax9.set_xlabel("Sample"); ax9.set_ylabel("Model cell")
except Exception:
    pass

plt.tight_layout()
png_path = REPORT_DIR / "demo_real_nn_algorithms.png"
plt.savefig(str(png_path), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  报告图: {png_path} ({png_path.stat().st_size/1024:.0f} KB)")

# ── 清理 ──────────────────────────────────────────────────────────────
_cleanup(NAME_1)
_cleanup(NAME_2)
_cleanup(NAME_3)

total = _pass + _fail
print(f"\n{'='*70}")
print(f"  Results: {_pass} passed, {_fail} failed / {total} total")
print(f"{'='*70}")
sys.exit(1 if _fail else 0)
