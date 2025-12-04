import cv2
import mediapipe as mp
import torch
import numpy as np
from collections import deque
import time # 用于模拟串口延迟，实际应用中可能需要更精确的控制

# --------------------------
# 配置区
# --------------------------
MODEL_PATH = "rps_mlp.pth"  # 你训练的模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH_WINDOW = 1  # ⚠️ 关键：减小平滑窗口到 1 或 2，以追求最低延迟
# --------------------------

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
# 必胜逻辑函数
# --------------------------
def get_winning_move(user_move):
    """根据用户的动作，返回机器人必胜的动作"""
    if user_move == 'rock':
        return 'paper'
    elif user_move == 'paper':
        return 'scissors'
    elif user_move == 'scissors':
        return 'rock'
    return "waiting" # 未检测到有效手势时，保持等待状态

# --------------------------
# 硬件指令发送函数（重点替换部分）
# --------------------------
# ⚠️ 注意：你需要将此函数中的 'pass' 和 print 替换为实际的串口或硬件通信代码
# 比如使用 `import serial` 并配置你的串口对象
def send_command(move):
    """
    发送指令给舵机控制板。
    move: 机器人需要出的手势 ('rock', 'paper', 'scissors', 'waiting')
    """
    if move == "waiting":
        # 保持手势不变或归位
        # print(f"-> 保持或归位指令")
        pass
    else:
        # 实际操作: serial_port.write(f"{move}\n".encode())
        print(f"🤖 **发送指令: 出 {move}**")
        # 模拟一个发送和执行的微小延迟
        # time.sleep(0.01) 
    return move

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
# 初始化 MediaPipe 和状态变量
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# 调低 min_detection_confidence 以更快地识别正在形成的手
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.4) 

pred_buffer = deque(maxlen=SMOOTH_WINDOW)
# 状态变量初始化
current_robot_move = "waiting" # 机器人当前已经执行的动作指令

# --------------------------
# 摄像头循环
# --------------------------
cap = cv2.VideoCapture(0)
print("🎮 启动必胜识别模式，按 q 退出")
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    user_move = "waiting" # 用户当前预测到的手势
    
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
                predicted_label = classes[label_idx]

            # 应用平滑（SMOOTH_WINDOW=1 时相当于无平滑）
            pred_buffer.append(predicted_label)
            if len(pred_buffer) == SMOOTH_WINDOW:
                # 简单多数投票来平滑（即使窗口为 1 也能工作）
                user_move = max(set(pred_buffer), key=pred_buffer.count)
                
            # 显示结果
            cv2.putText(frame, f"You: {user_move} ({conf*100:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
    else:
        pred_buffer.clear()
        user_move = "waiting"

    # ----------------------------------------------------
    # 核心作弊/指令发送逻辑：状态变更触发
    # ----------------------------------------------------
    
    # 1. 计算机器人必胜手势
    required_robot_move = get_winning_move(user_move)

    # 2. 判断是否需要发送新指令
    if required_robot_move != current_robot_move:
        # 用户的预测动作发生了变化 (例如从 waiting -> rock, 或 rock -> paper)
        # 立即发送新的必胜指令
        current_robot_move = send_command(required_robot_move)
    
    # 显示机器人动作
    cv2.putText(frame, f"Robot: {current_robot_move.upper()}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # FPS 计算 (可选，用于确认系统延迟)
    frame_count += 1
    if frame_count % 30 == 0:
        end_time = time.time()
        fps = 30 / (end_time - start_time)
        start_time = end_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Janken Robot - Cheat Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()