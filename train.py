import os
import time
import argparse
import hashlib
from datetime import datetime
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from common.models import GestureCNNTemporal, load_gesture_mlp

# 文件用途：训练入口，生成轻量 CNN+GRU 手势模型权重
# 最后修改：2025-12-04
# 主要功能：
# - 读取 data/*.npy 并做标签编码
# - 构建同标签短序列训练集
# - 训练轻量 CNN+GRU（含早停与评估）
# - 保存最佳模型与元数据
# 重要组件：GestureCNNTemporal、AdamW、EarlyStopping
# ---------- 配置 ----------
DATA_DIR = "data"
BASELINE_MODEL_PATH = "rps_mlp.pth"
VERSION = "v1"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = "logs"
CKPT_DIR = "checkpoints"
MODEL_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=os.path.join(DATA_DIR, "dataset.npy"))
parser.add_argument("--labels", type=str, default=os.path.join(DATA_DIR, "labels.npy"))
parser.add_argument("--output", type=str, default=os.path.join(MODEL_DIR, f"gesture_recognition_cnn_temporal_{VERSION}_{RUN_ID}.pth"))
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--seq-len", type=int, default=5)
parser.add_argument("--hidden-size", type=int, default=64)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--augment", action="store_true")
args = parser.parse_args()

OUT_MODEL_PATH = args.output
LOG_CSV = os.path.join(LOG_DIR, f"train_{VERSION}_{RUN_ID}.csv")
CKPT_PATH = os.path.join(CKPT_DIR, f"{VERSION}_latest.pt")
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay
SEQ_LEN = args.seq_len
HIDDEN = args.hidden_size
PATIENCE = args.patience
AUGMENT = bool(args.augment)
# --------------------------

# 1. 读取数据
dataset_path = args.data
labels_path = args.labels
X = np.load(dataset_path)
y = np.load(labels_path)
print("✅ 数据加载完成:", X.shape, y.shape)

# 2. 标签编码
le = LabelEncoder()
if len(y) > 0:
    y_enc = le.fit_transform(y)
    classes_list = list(le.classes_)
    num_classes = len(classes_list)
else:
    y_enc = np.array([], dtype=np.int64)
    classes_list = ["rock", "paper", "scissors"]
    num_classes = len(classes_list)
print("类别:", classes_list)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

DATASET_HASH = sha256_file(dataset_path)
LABELS_HASH = sha256_file(labels_path)

def build_sequence_dataset(
    feats: np.ndarray, labels: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    seq_X = []
    seq_y = []
    n = len(labels)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        segment_len = j - i
        if segment_len >= seq_len:
            for s in range(i, j - seq_len + 1):
                seq_X.append(feats[s : s + seq_len])
                seq_y.append(labels[s])
        else:
            pad = np.repeat(feats[j - 1 : j], seq_len - segment_len, axis=0)
            seq = np.concatenate([feats[i:j], pad], axis=0)
            seq_X.append(seq)
            seq_y.append(labels[i])
        i = j
    return np.array(seq_X), np.array(seq_y)

def clean_features(feats: np.ndarray) -> np.ndarray:
    if feats.size == 0:
        return feats.astype(np.float32)
    mask_nan = ~np.isnan(feats).any(axis=1)
    feats = feats[mask_nan]
    if feats.size == 0:
        return feats.astype(np.float32)
    feats = feats.astype(np.float32)
    xyz = feats.reshape(-1, 21, 3)
    xyz[:, :, 0] = np.clip(xyz[:, :, 0], 0.0, 1.0)
    xyz[:, :, 1] = np.clip(xyz[:, :, 1], 0.0, 1.0)
    xyz[:, :, 2] = np.clip(xyz[:, :, 2], -1.0, 1.0)
    return xyz.reshape(-1, 63)

# 3. 清洗与构建时序全集并划分 70/20/10
X = clean_features(X)
if len(X) > 0 and len(y_enc) > 0:
    SX, SY = build_sequence_dataset(X, y_enc, SEQ_LEN)
    X_train_seq, X_rem_seq, y_train_seq, y_rem_seq = train_test_split(
        SX, SY, test_size=0.3, random_state=42, stratify=SY
    )
    X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
        X_rem_seq, y_rem_seq, test_size=(1/3), random_state=42, stratify=y_rem_seq
    )
else:
    X_train_seq = np.empty((0, SEQ_LEN, 63), dtype=np.float32)
    y_train_seq = np.empty((0,), dtype=np.int64)
    X_val_seq = np.empty((0, SEQ_LEN, 63), dtype=np.float32)
    y_val_seq = np.empty((0,), dtype=np.int64)
    X_test_seq = np.empty((0, SEQ_LEN, 63), dtype=np.float32)
    y_test_seq = np.empty((0,), dtype=np.int64)

def augment_seq(sample: np.ndarray) -> np.ndarray:
    T = sample.shape[0]
    xyz = sample.reshape(T, 21, 3)
    ang = np.deg2rad(np.random.uniform(-10, 10))
    rot = np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]], dtype=np.float32)
    xy = xyz[:, :, :2]
    cen = xy.mean(axis=1, keepdims=True)
    xy = ((xy - cen) @ rot.T) + cen
    shift = np.random.uniform(-0.02, 0.02, size=(T,1,2)).astype(np.float32)
    scale = np.random.uniform(0.95, 1.05)
    xy = np.clip((xy + shift) * scale, 0.0, 1.0)
    xyz[:, :, :2] = xy
    xyz[:, :, 2] += np.random.normal(0.0, 0.01, size=(T,21)).astype(np.float32)
    xyz[:, :, 2] = np.clip(xyz[:, :, 2], -1.0, 1.0)
    return xyz.reshape(T, 63).astype(np.float32)

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment_seq(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

train_ds = SeqDataset(X_train_seq, y_train_seq, augment=AUGMENT)
val_ds = SeqDataset(X_val_seq, y_val_seq, augment=False)
test_ds = SeqDataset(X_test_seq, y_test_seq, augment=False)

X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.long)
X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
y_val_t = torch.tensor(y_val_seq, dtype=torch.long)
X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_t = torch.tensor(y_test_seq, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=(len(train_ds) > 0))
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

# 4. 定义 MLP 模型
class MLP(nn.Module):
    """三层感知机，用于手势分类"""
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = GestureCNNTemporal(num_classes=num_classes, hidden_size=HIDDEN)
model = model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss()

torch.manual_seed(42)
np.random.seed(42)

best_loss = float("inf")
pat = 0
best_state = None
best_epoch = 0
start_time = datetime.now().isoformat()
start_epoch = 1

# 断点续训：若存在最新检查点则加载继续训练
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    if ckpt.get("version") == VERSION:
        model.load_state_dict(ckpt["model_state"])  # type: ignore
        optimizer.load_state_dict(ckpt["optimizer_state"])  # type: ignore
        best_loss = float(ckpt.get("best_loss", best_loss))
        pat = int(ckpt.get("patience_count", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"▶️ 续训从 epoch {start_epoch} 开始 | best_loss={best_loss:.4f}"
        )

# 写入日志 CSV 头
with open(LOG_CSV, "w", encoding="utf-8") as f:
    f.write("epoch,train_loss,val_loss,patience,best_loss\n")

if len(X_train_t) == 0 or len(y_train_t) == 0:
    print("⚠️ 数据集为空，跳过训练并保存未训练模型以保留结构与参数配置。")
    avg_loss = float("nan")
    val_loss = float("nan")
else:
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | train={avg_loss:.4f} | val={val_loss:.4f}"
        )
        with open(LOG_CSV, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{val_loss:.6f},{pat},{best_loss:.6f}\n"
            )

        scheduler.step(val_loss)

        # 保存最新检查点以支持断点续训
        torch.save(
            {
                "version": VERSION,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss,
                "patience_count": pat,
                "classes": le.classes_,
            },
            CKPT_PATH,
        )

        # 每 10 轮保存一次额外检查点
        if epoch % 10 == 0:
            ckpt_epoch_path = os.path.join(CKPT_DIR, f"{VERSION}_epoch_{epoch}.pt")
            torch.save(
                {
                    "version": VERSION,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "classes": le.classes_,
                },
                ckpt_epoch_path,
            )

        # 早停逻辑
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                print("⏹️ 早停触发")
                break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
preds, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(yb.numpy())

REPORT_TXT = os.path.join(LOG_DIR, f"report_{VERSION}_{RUN_ID}.txt")
CM_CSV = os.path.join(LOG_DIR, f"cm_{VERSION}_{RUN_ID}.csv")
if len(targets) > 0:
    rep = classification_report(targets, preds, target_names=classes_list)
    print("✅ 分类结果：")
    print(rep)
    acc = (np.array(preds) == np.array(targets)).mean()
    cm = confusion_matrix(targets, preds)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(rep)
        f.write("\n\nConfusion Matrix:\n")
        for row in cm:
            f.write(",".join(map(str, row)) + "\n")
    np.savetxt(CM_CSV, cm, fmt="%d", delimiter=",")
else:
    print("⚠️ 无测试样本，跳过分类报告。")
    acc = float("nan")
print(f"Accuracy: {acc:.4f}")

def measure_latency_model(model: nn.Module, X_seq: torch.Tensor) -> Tuple[float, float, float]:
    times = []
    n = X_seq.size(0)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    for i in range(min(500, n)):
        x = X_seq[i : i + 1].to(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
    return float(np.mean(times)), float(np.min(times)), float(np.max(times))

mean_ms, min_ms, max_ms = measure_latency_model(model, X_test_t if len(X_test_t) > 0 else X_val_t if len(X_val_t) > 0 else X_train_t)
print(
    f"Latency (ms) | mean={mean_ms:.2f} min={min_ms:.2f} max={max_ms:.2f}"
)

# 与基线 MLP 比较（若存在）
baseline_ms = None
if os.path.exists(BASELINE_MODEL_PATH):
    base_model, base_classes = load_gesture_mlp(BASELINE_MODEL_PATH, DEVICE)
    def measure_baseline(m, X_seq):
        times = []
        for i in range(min(500, X_seq.size(0))):
            # 取序列最后一帧，符合旧 MLP 输入
            x = X_seq[i, -1].unsqueeze(0).to(DEVICE)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = m(x)
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
        return float(np.mean(times)) if len(times) > 0 else float("nan")
    baseline_ms = measure_baseline(base_model, X_test_t)
    print(f"Baseline MLP mean latency: {baseline_ms:.2f} ms")
    if baseline_ms > 0:
        reduction = (baseline_ms - mean_ms) / baseline_ms
        print(f"Latency reduction: {reduction*100:.1f}%")

end_time = datetime.now().isoformat()

torch.save(
    {
        "model_state": model.state_dict(),
        "classes": classes_list,
        "meta": {
            "arch": "GestureCNNTemporal",
            "version": VERSION,
            "run_id": RUN_ID,
            "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN,
            "use_maxpool": True,
            "best_epoch": best_epoch,
            "train_config": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "patience": PATIENCE,
                "device": DEVICE,
                "seed": 42,
            },
            "dataset": {
                "path_dataset": dataset_path,
                "path_labels": labels_path,
                "hash_dataset": DATASET_HASH,
                "hash_labels": LABELS_HASH,
                "num_classes": int(num_classes),
            },
            "metrics": {
                "accuracy": float(acc),
                "latency_ms": {
                    "mean": float(mean_ms),
                    "min": float(min_ms),
                    "max": float(max_ms),
                },
                "baseline_mean_ms": float(baseline_ms) if baseline_ms else None,
                "report_txt": REPORT_TXT,
                "cm_csv": CM_CSV,
            },
            "time": {
                "start": start_time,
                "end": end_time,
            },
            "preprocess": {
                "clip_xy": [0.0, 1.0],
                "clip_z": [-1.0, 1.0],
                "remove_nan": True,
            },
            "augmentation": {
                "enabled": AUGMENT,
                "rotation_deg": 10,
                "shift": 0.02,
                "scale": [0.95, 1.05],
                "z_noise_std": 0.01,
            },
        },
    },
    OUT_MODEL_PATH,
)

size_mb = os.path.getsize(OUT_MODEL_PATH) / (1024 * 1024)
with open(OUT_MODEL_PATH, "rb") as f:
    model_sha = hashlib.sha256(f.read()).hexdigest()
print(
    f"💾 模型已保存: {OUT_MODEL_PATH} | size={size_mb:.2f} MB | sha256={model_sha}"
)

print(f"📝 日志: {LOG_CSV}")
print(f"💠 检查点: {CKPT_PATH}")
