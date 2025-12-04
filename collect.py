import cv2
import numpy as np
import os
import sys
import argparse
import logging

MP_AVAILABLE = True
try:
    import mediapipe as mp
except ImportError:
    MP_AVAILABLE = False
    mp = None

# 文件用途：数据采集入口，录制手势关键点数据
# 最后修改：2025-12-04
# 主要功能：
# - 交互采集不同手势的 21 点 (x,y,z) 坐标
# - 打包为 63 维向量保存到 .npy 文件
# 重要函数：record_gesture(label)
# 使用说明：采集完成后运行 train.py 训练生成 rps_mlp.pth。
if MP_AVAILABLE:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_hands = None
    mp_drawing = None

# 三类动作（可扩展）
GESTURES = ["rock", "paper", "scissors"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
# 解释：如果不存在 data 文件夹就创建一个，用来保存采集结果。

all_data = []
all_labels = []

def run_self_test() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Python {sys.version.split()[0]}")
    logging.info(f"NumPy {np.__version__}")
    logging.info(f"OpenCV {cv2.__version__}")
    logging.info(f"MediaPipe {'OK' if MP_AVAILABLE else 'MISSING'}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.warning("Camera not available")
        return 0
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logging.warning("Camera read failed")
        return 0
    logging.info("Camera OK")
    if MP_AVAILABLE:
        hands_test = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ = hands_test.process(rgb)
        try:
            hands_test.close()
        except Exception:
            pass
        logging.info("Mediapipe process OK")
    else:
        logging.info("Mediapipe process skipped")
    logging.info("Self-test completed")
    return 0

def record_gesture(label):
    """录制指定标签的若干帧关键点并缓存到内存列表"""
    print(f"\n准备录制手势: {label}")
    print("按 's' 开始录制，'q' 退出该手势")

    cap = cv2.VideoCapture(0)
    # 解释：打开默认摄像头（编号 0）。
    recording = False
    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 解释：把检测到的手部关键点和连接线画到画面上，便于可视化。

        cv2.putText(frame, f"Gesture: {label} ({collected})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
            print("开始录制...")
        elif key == ord('q'):
            break
        # 解释：按 's' 开始采集，按 'q' 退出该手势的采集循环。

        if recording and result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            data = []
            for lm in landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            all_data.append(data)
            all_labels.append(label)
            collected += 1
            # 解释：把一帧的 63 个数字存入 all_data，并记录对应的手势标签。

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        code = run_self_test()
        sys.exit(code)

    if not MP_AVAILABLE:
        raise ImportError("未安装 mediapipe，请先执行: python -m pip install mediapipe")

    def init_hands():
        try:
            return mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        except FileNotFoundError as e:
            def get_ascii_resource_dir() -> str:
                env_dir = os.environ.get("MP_RESOURCE_DIR") or os.environ.get("MEDIAPIPE_RESOURCE_DIR")
                if env_dir:
                    # allow passing either site-packages or site-packages\\mediapipe
                    return env_dir
                return r"C:\\Users\\32028\\RPSrobotEnv\\myenv312\\Lib\\site-packages"

            ascii_dir = get_ascii_resource_dir()
            # normalize to site-packages root
            if os.path.basename(ascii_dir).lower() == "mediapipe":
                ascii_dir = os.path.dirname(ascii_dir)
            binarypb = os.path.join(
                ascii_dir,
                "mediapipe",
                "modules",
                "hand_landmark",
                "hand_landmark_tracking_cpu.binarypb",
            )
            if not (os.path.isdir(ascii_dir) and os.path.isfile(binarypb)):
                msg = (
                    "Mediapipe 资源文件加载失败。请使用 ASCII 路径的虚拟环境"
                    "（例如 C:\\Users\\...\\myenv312），并重新安装依赖。"
                )
                raise RuntimeError(msg) from e

            from mediapipe.python._framework_bindings import resource_util
            _orig_set = resource_util.set_resource_dir
            def _override_set(_path: str):
                return _orig_set(ascii_dir)
            resource_util.set_resource_dir = _override_set

            from mediapipe.framework import calculator_pb2
            with open(binarypb, "rb") as f:
                graph_bytes = f.read()
            graph = calculator_pb2.CalculatorGraphConfig()
            graph.ParseFromString(graph_bytes)

            from mediapipe.python.solution_base import SolutionBase
            return SolutionBase(
                graph_config=graph,
                side_inputs={
                    "model_complexity": 1,
                    "num_hands": 1,
                    "use_prev_landmarks": True,
                },
                calculator_params={},
                outputs=[
                    "multi_hand_landmarks",
                    "multi_hand_world_landmarks",
                    "multi_handedness",
                ],
            )

    global hands
    hands = init_hands()

    for gesture in GESTURES:
        record_gesture(gesture)

    np.save(os.path.join(DATA_DIR, "dataset.npy"), np.array(all_data))
    np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(all_labels))
    print(f"✅ 数据保存完成，共 {len(all_data)} 条样本")

if __name__ == "__main__":
    main()
