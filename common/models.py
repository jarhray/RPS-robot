# 文件用途：手势识别模型与加载工具
# 最后修改：2025-12-04
# 主要功能：
# - 定义三层 MLP（GestureMLP）
# - 统一加载模型与类别集合（load_gesture_mlp）
# 使用说明：入口脚本统一从此处加载模型与类别。

import torch
import torch.nn as nn
from typing import List, Tuple


class GestureMLP(nn.Module):
    """三层 MLP：输入 63 维手部关键点坐标，输出 3+ 类手势

    解释：
    - Linear：线性层（加权求和），用于特征变换。
    - ReLU：激活函数，负数置 0，增强非线性表达能力。
    - input_size=63：21 个关键点 × 每点 (x,y,z)。
    - num_classes：由训练时的类别集合决定（通常是 rock/paper/scissors）。
    """

    def __init__(self, input_size=63, hidden_size=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_gesture_mlp(model_path: str, device: str, input_size: int = 63, hidden_size: int = 128):
    """加载训练好的 MLP 模型并返回 (model, classes)

    参数：
    - model_path：训练后保存的权重文件路径（rps_mlp.pth）。
    - device：'cuda' 或 'cpu'，根据你的环境自动选择。
    - input_size/hidden_size：需与训练时保持一致（默认 63/128）。
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])  # 训练时的类别集合
    model = GestureMLP(input_size=input_size, hidden_size=hidden_size, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.depth = nn.Conv1d(
            in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch
        )
        self.point = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return self.act(x)


class GestureCNNTemporal(nn.Module):
    """轻量级 CNN + GRU 时序模型

    输入：
    - 单帧：形状 [B, 63]
    - 序列：形状 [B, T, 63]

    结构：
    - 1D 深度可分离卷积堆叠（3→16→24→32→32）
    - 可选最多 2 个最大池化层
    - 全局平均池化得到帧级特征 (32)
    - GRU(hidden<=64) 进行时序处理，输出分类
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 64,
        use_maxpool: bool = True,
    ):
        super().__init__()
        self.use_maxpool = use_maxpool
        self.conv1 = DepthwiseSeparableConv1D(3, 16)
        self.conv2 = DepthwiseSeparableConv1D(16, 24)
        self.conv3 = DepthwiseSeparableConv1D(24, 32)
        self.conv4 = DepthwiseSeparableConv1D(32, 32)
        self.pool = nn.MaxPool1d(2)
        self.gru = nn.GRU(32, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def _cnn_frame(self, x63: torch.Tensor) -> torch.Tensor:
        b = x63.size(0)
        x = x63.view(b, 21, 3).transpose(1, 2)  # [B, 3, 21]
        x = self.conv1(x)
        if self.use_maxpool:
            x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.use_maxpool:
            x = self.pool(x)
        x = self.conv4(x)
        x = x.mean(dim=2)  # GAP → [B, 32]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            feat = self._cnn_frame(x)
            feat = feat.unsqueeze(1)
        else:
            b, t, f = x.size()
            x_flat = x.view(b * t, f)
            feat = self._cnn_frame(x_flat)
            feat = feat.view(b, t, -1)
        out, _ = self.gru(feat)
        last = out[:, -1, :]
        return self.head(last)


def load_gesture_cnn_temporal(
    model_path: str, device: str
) -> Tuple[nn.Module, List[str], dict]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])  # type: ignore
    meta = ckpt.get("meta", {})
    hidden = int(meta.get("hidden_size", 64))
    use_maxpool = bool(meta.get("use_maxpool", True))
    model = GestureCNNTemporal(
        num_classes=len(classes), hidden_size=hidden, use_maxpool=use_maxpool
    )
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.to(device)
    model.eval()
    return model, classes, meta

