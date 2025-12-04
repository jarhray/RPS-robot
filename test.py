import cv2
import mediapipe as mp
import torch
import numpy as np
from collections import deque

# --------------------------
# 配置区
# --------------------------
MODEL_PATH = "rps_mlp.pth"  # 你训练的模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH_WINDOW = 3  # 平滑窗口大小

# --------------------------
# 模型定义（和 train_mlp.py 保持一致）
# --------------------------
class RPS_MLP(torch.nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# 加载模型
# --------------------------
print("✅ 加载模型中...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = RPS_MLP(input_size=63, hidden_size=128, num_classes=len(ckpt["classes"]))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()
classes = list(ckpt["classes"])
print(f"✅ 模型已加载，类别：{classes}")

# --------------------------
# 初始化 MediaPipe
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# --------------------------
# 平滑预测缓存
# --------------------------
pred_buffer = deque(maxlen=SMOOTH_WINDOW)

# --------------------------
# 摄像头循环
# --------------------------
cap = cv2.VideoCapture(0)
print("🎮 启动实时识别，按 q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 提取关键点坐标
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # 转为Tensor并预测
            x = torch.tensor(landmarks, dtype=torch.float32).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                preds = model(x)
                probs = torch.nn.functional.softmax(preds, dim=1)
                label_idx = torch.argmax(probs, dim=1).item()
                conf = probs[0, label_idx].item()
                label = classes[label_idx]

            # 加入平滑窗口
            pred_buffer.append(label)
            if len(pred_buffer) == SMOOTH_WINDOW:
                label = max(set(pred_buffer), key=pred_buffer.count)

            # 显示结果
            
            if label == 'rock':
                cv2.putText(frame, f"paper ({conf*100:.1f}%)",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            elif label == 'paper':
                cv2.putText(frame, f"scissors ({conf*100:.1f}%)",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            elif label == 'scissors':
                cv2.putText(frame, f"rock ({conf*100:.1f}%)",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    else:
        pred_buffer.clear()

    cv2.imshow("RPS Real-time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
