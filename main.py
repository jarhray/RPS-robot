import cv2
import mediapipe as mp
import torch
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
import time
import threading

# ==========================================
# 1. 极速配置
# ==========================================
XML_PATH = "adam_u.xml"
MODEL_PATH = "rps_mlp.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ⚡ 提升舵机速度到 20.0 (接近人眼极限的快)
SERVO_SPEED = 20.0 
# ⚡ 降低图像分辨率以提高处理速度
CAM_WIDTH, CAM_HEIGHT = 640, 480

# 关节与动作配置 (保持不变)
BEND_VAL = 1.0  
STRAIGHT_VAL = 0.0
FINGER_JOINTS = {
    "thumb":  ["R_thumb_MCP_joint1", "R_thumb_MCP_joint2", "R_thumb_PIP_joint", "R_thumb_DIP_joint"],
    "index":  ["R_index_MCP_joint", "R_index_DIP_joint"],
    "middle": ["R_middle_MCP_joint", "R_middle_DIP_joint"],
    "ring":   ["R_ring_MCP_joint", "R_ring_DIP_joint"],
    "pinky":  ["R_pinky_MCP_joint", "R_pinky_DIP_joint"]
}
ARM_POSE = {"shoulderPitch_Right": -0.5, "elbow_Right": -1.0, "wristPitch_Right": 0.0}
GESTURES = {
    "rock":     {"thumb": 1, "index": 1, "middle": 1, "ring": 1, "pinky": 1},
    "paper":    {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0},
    "scissors": {"thumb": 1, "index": 0, "middle": 0, "ring": 1, "pinky": 1},
    "waiting":  {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0} 
}

# ==========================================
# 2. 多线程摄像头类 (核心优化)
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        # 尝试设置 FPS 为 60 (取决于摄像头是否支持)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        
        self.status, self.frame = self.capture.read()
        self.stop_event = False
        self.lock = threading.Lock()
        
        # 启动后台线程
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stop_event:
            status, frame = self.capture.read()
            if status:
                with self.lock:
                    self.frame = frame
                    self.status = status
            else:
                self.stop_event = True

    def get_frame(self):
        with self.lock:
            return self.status, self.frame.copy() # 返回副本以防竞争

    def release(self):
        self.stop_event = True
        self.thread.join()
        self.capture.release()

# ==========================================
# 3. 逻辑与模型
# ==========================================
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
    def forward(self, x): return self.net(x)

def get_winning_move(user_move):
    mapping = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    return mapping.get(user_move, "waiting")

# ==========================================
# 4. 仿真控制器
# ==========================================
class RealisticServoController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.joint_ids = {}
        self.target_qpos = {}
        
        for finger, joint_names in FINGER_JOINTS.items():
            ids = []
            for name in joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid != -1:
                    ids.append(jid)
                    addr = model.jnt_qposadr[jid]
                    self.target_qpos[addr] = data.qpos[addr]
            self.joint_ids[finger] = ids
            
        self.arm_ids = {}
        for name in ARM_POSE.keys():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1: self.arm_ids[name] = jid

    def set_initial_pose(self):
        for name, angle in ARM_POSE.items():
            if name in self.arm_ids:
                self.data.qpos[self.model.jnt_qposadr[self.arm_ids[name]]] = angle
        if self.model.nq >= 7: self.data.qpos[2] = 1.0 
        mujoco.mj_forward(self.model, self.data)

    def apply_gesture(self, gesture_name):
        target_config = GESTURES.get(gesture_name, GESTURES["waiting"])
        for finger, is_bent in target_config.items():
            target_angle = BEND_VAL if is_bent else STRAIGHT_VAL
            if finger in self.joint_ids:
                for jid in self.joint_ids[finger]:
                    self.target_qpos[self.model.jnt_qposadr[jid]] = target_angle

    def update_servos(self, dt):
        for addr, target in self.target_qpos.items():
            current = self.data.qpos[addr]
            step_limit = SERVO_SPEED * dt
            diff = target - current
            if abs(diff) > 1e-4:
                self.data.qpos[addr] = current + np.clip(diff, -step_limit, step_limit)

# ==========================================
# 5. 主程序
# ==========================================
def run_simulation():
    # Load Model
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    net = RPS_MLP(num_classes=len(ckpt["classes"]))
    net.load_state_dict(ckpt["model_state"])
    net.to(DEVICE)
    net.eval()
    classes = list(ckpt["classes"])

    # Init MediaPipe (CPU Mode, but optimized)
    mp_hands = mp.solutions.hands
    # ⚠️ 降低置信度阈值：让它更早地检测到手，哪怕有点模糊
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.3, 
        min_tracking_confidence=0.3
    )

    # Init MuJoCo
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.gravity = (0, 0, 0)
    controller = RealisticServoController(m, d)
    controller.set_initial_pose()

    # Init Threaded Camera
    print("🎥 启动多线程摄像头采集...")
    cam = ThreadedCamera(0)
    time.sleep(1.0) # 等待摄像头预热

    current_robot_move = "waiting"
    dt = m.opt.timestep 
    
    # FPS Calculation
    fps_start = time.time()
    frames = 0
    inference_time = 0

    print("🚀 极速模式已启动 - 请测试")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and cam.status:
            loop_start = time.time()
            
            # 1. 获取最新帧 (非阻塞)
            ret, frame = cam.get_frame()
            if not ret: break
            
            # 2. 图像处理
            frame_flip = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
            
            t0 = time.time()
            result = hands.process(rgb)
            inference_time = (time.time() - t0) * 1000 # ms
            
            user_move = "waiting"
            if result.multi_hand_landmarks:
                landmarks = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # MLP Inference
                x_tensor = torch.tensor(landmarks, dtype=torch.float32).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    preds = net(x_tensor)
                    # ⚠️ 激进策略：只要概率分布有一点倾向，马上行动，不要等 softmax 完全确定
                    user_move = classes[torch.argmax(preds, dim=1).item()]

            # 3. 立即触发动作
            winning_move = get_winning_move(user_move)
            if winning_move != current_robot_move:
                current_robot_move = winning_move
                controller.apply_gesture(current_robot_move)

            # 4. 物理步进 (追赶时间)
            # 尽量保持物理循环不阻塞视觉循环
            sim_time_budget = 0.005 # 给物理引擎分配 5ms
            sim_start = time.time()
            while time.time() - sim_start < sim_time_budget:
                controller.update_servos(dt)
                d.qvel[:] = 0
                d.qpos[0:3] = [0,0,1]
                mujoco.mj_step(m, d)
            
            viewer.sync()
            
            # 5. 信息显示
            frames += 1
            if time.time() - fps_start >= 1.0:
                fps = frames / (time.time() - fps_start)
                print(f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms")
                frames = 0
                fps_start = time.time()

            cv2.putText(frame_flip, f"User: {user_move}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_flip, f"Robot: {current_robot_move}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_flip, f"Inference: {inference_time:.1f}ms", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
            cv2.imshow("Fast Sim", frame_flip)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_simulation()