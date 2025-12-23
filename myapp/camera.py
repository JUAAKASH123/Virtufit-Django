import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Virtufit.settings")
django.setup()

# Suppress TensorFlow / MediaPipe logs
# 0 = all logs, 1 = info, 2 = warnings, 3 = errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from django.conf import settings
import cv2
import numpy as np
import math
import time
import mediapipe as mp
from collections import deque
import os
import threading

# You had a cloth.py that loads images. We'll include a simple loader below.
# If you already have cloth.py, you can import that instead of this loader.
ASSET_PATH = os.path.join(settings.BASE_DIR,'myapp', 'static', 'assets','he')

# ---------------- Configuration ----------------
DEBUG = False
NEAR_WHITE_THRESH = 200

POS_SMOOTH = 6
SCALE_SMOOTH = 6

WIDTH_SCALE = 2.0
HEIGHT_SCALE = 1.6
VERTICAL_OFFSET_FACTOR = 0.15

MOTION_LOCK_PX = 2

BUTTON_COOLDOWN = 0.6
PINCH_HOLD_TIME = 1.0
PINCH_THRESHOLD = 55

# ------------------------------------------------

_pos_x_q = deque(maxlen=POS_SMOOTH)
_pos_y_q = deque(maxlen=POS_SMOOTH)
_scale_q = deque(maxlen=SCALE_SMOOTH)

hand_was_inside = {"left": False, "right": False}
hand_current_inside = {"left": False, "right": False}
pinch_start = {"left": None, "right": None}
last_button_time = 0.0

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global controller & thread
class CameraController:
    def __init__(self, device=0):
        self.device = device
        self.cap = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.images = []
        self.clothes = []
        self.cur_id = 0
        self.num = 0
        self.prev_top_x = None
        self.prev_top_y = None
        self.last_button_time = 0.0
        self.loadImages()
        

    def loadImages(self, path=ASSET_PATH):
        images = []
        if not os.path.exists(path):
            print("Assets folder not found:", path)
            self.images = images
            self.num = len(images)
            return

        files = sorted(os.listdir(path))
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                # rotate here like your loader
                img = cv2.rotate(img, cv2.ROTATE_180)
                images.append(img)
        self.images = images
        self.num = len(images)
        if self.num == 0:
            print("No clothes loaded in assets. Place PNG/JPGs in vdressing/assets/")

    # utility functions (dist, overlay, centroid) from your script:
    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def  set_cloth(self,index:int):
        if self.num==0:
            return
        index=max(0,min(index,self.num-1))
        self.cur_id=index
        _pos_x_q.clear()
        _pos_y_q.clear()
        _scale_q.clear()

    def get_cloth_centroid(self, cloth_img):
        h, w = cloth_img.shape[:2]
        if cloth_img.shape[2] == 4:
            a = cloth_img[:, :, 3]
            mask = (a > 10).astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, NEAR_WHITE_THRESH, 255, cv2.THRESH_BINARY_INV)
        if mask.sum() == 0:
            return w // 2
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return w // 2
        return int(M["m10"] / M["m00"])

    def overlay_png(self, frame, img, x, y):
        if img is None:
            return frame
        img = img.copy()
        H, W = frame.shape[:2]
        h, w = img.shape[:2]
        if img.shape[2] == 3:
            lower = np.array([NEAR_WHITE_THRESH] * 3)
            mask = cv2.inRange(img, lower, np.array([255, 255, 255]))
            alpha = 255 - mask
            b, g, r = cv2.split(img)
            img = cv2.merge((b, g, r, alpha))
            h, w = img.shape[:2]
        if x < 0:
            img = img[:, -x:]
            w = img.shape[1]
            x = 0
        if y < 0:
            img = img[-y:, :]
            h = img.shape[0]
            y = 0
        if x >= W or y >= H or h <= 0 or w <= 0:
            return frame
        if x + w > W:
            img = img[:, :W - x]
            w = img.shape[1]
        if y + h > H:
            img = img[:H - y, :]
            h = img.shape[0]
        b, g, r, a = cv2.split(img)
        overlay = cv2.merge((b, g, r)).astype(np.float32)
        alpha = (a.astype(np.float32) / 255.0)[..., None]
        roi = frame[y:y + h, x:x + w].astype(np.float32)
        blended = alpha * overlay + (1 - alpha) * roi
        frame[y:y + h, x:x + w] = blended.astype(np.uint8)
        return frame

    def is_point_in_box(self, pt, box):
        x, y = pt
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def is_thumbs_up(self, hand_lm, W, H):
        try:
            TH_TIP = 4
            TH_IP = 3
            F2_TIP, F2_MCP = 8, 5
            F3_TIP, F3_MCP = 12, 9
            F4_TIP, F4_MCP = 16, 13
            F5_TIP, F5_MCP = 20, 17
            def P(id):
                return (int(hand_lm.landmark[id].x * W), int(hand_lm.landmark[id].y * H))
            th_tip = P(TH_TIP)
            th_ip = P(TH_IP)
            f2_tip, f2_mcp = P(F2_TIP), P(F2_MCP)
            f3_tip, f3_mcp = P(F3_TIP), P(F3_MCP)
            f4_tip, f4_mcp = P(F4_TIP), P(F4_MCP)
            f5_tip, f5_mcp = P(F5_TIP), P(F5_MCP)
            thumb_up = th_tip[1] < th_ip[1]
            others_down = (
                f2_tip[1] > f2_mcp[1] and
                f3_tip[1] > f3_mcp[1] and
                f4_tip[1] > f4_mcp[1] and
                f5_tip[1] > f5_mcp[1]
            )
            return thumb_up and others_down
        except Exception:
            return False

    # Thread that captures+processes frames
    def _run(self):
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            print("Cannot open camera.")
            self.running = False
            return
        pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            # Button boxes
            button_width = 140
            button_height = 50
            left_btn = (20, 20, 20 + button_width, 20 + button_height)
            right_btn = (W - button_width - 20, 20, W - 20, 20 + button_height)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pres = pose.process(rgb)

            shoulder_center = (W // 2, int(H * 0.25))
            shoulder_dist = max(40, W * 0.2)
            hip_mid = (W // 2, int(H * 0.5))

            if pres.pose_landmarks:
                pose_lm = pres.pose_landmarks.landmark
                LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                LHIP = mp_pose.PoseLandmark.LEFT_HIP.value
                RHIP = mp_pose.PoseLandmark.RIGHT_HIP.value
                ls = (int(pose_lm[LSH].x * W), int(pose_lm[LSH].y * H))
                rs = (int(pose_lm[RSH].x * W), int(pose_lm[RSH].y * H))
                lh = (int(pose_lm[LHIP].x * W), int(pose_lm[LHIP].y * H))
                rh = (int(pose_lm[RHIP].x * W), int(pose_lm[RHIP].y * H))
                shoulder_center = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
                shoulder_dist = max(40, self.dist(ls, rs))
                hip_mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)

            # scale & smooth
            target_w = int(shoulder_dist * WIDTH_SCALE)
            target_h = int(self.dist(shoulder_center, hip_mid) * HEIGHT_SCALE)
            _scale_q.append(target_w)
            smooth_w = int(sum(_scale_q) / len(_scale_q)) if len(_scale_q) else target_w

            orig = self.images[self.cur_id] if self.num else None
            if orig is not None:
                resized = cv2.resize(orig, (smooth_w, target_h))
                resized = cv2.flip(resized, 0)  # as original
                centroid_x = self.get_cloth_centroid(resized)
                top_x = shoulder_center[0] - centroid_x
                top_y = int(shoulder_center[1] - target_h * VERTICAL_OFFSET_FACTOR)
                _pos_x_q.append(top_x)
                _pos_y_q.append(top_y)
                sm_x = int(sum(_pos_x_q) / len(_pos_x_q))
                sm_y = int(sum(_pos_y_q) / len(_pos_y_q))

                if self.prev_top_x is None:
                    fx = sm_x
                else:
                    fx = self.prev_top_x if abs(self.prev_top_x - sm_x) < MOTION_LOCK_PX else sm_x

                if self.prev_top_y is None:
                    fy = sm_y
                else:
                    fy = self.prev_top_y if abs(self.prev_top_y - sm_y) < MOTION_LOCK_PX else sm_y

                self.prev_top_x = fx
                self.prev_top_y = fy

                frame = self.overlay_png(frame, resized, fx, fy)

            # hands
            global hand_current_inside, hand_was_inside, pinch_start
            hand_current_inside = {"left": False, "right": False}
            hres = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if hres.multi_hand_landmarks and hres.multi_handedness:
                now = time.time()
                for idx, handed in enumerate(hres.multi_handedness):
                    label = "left" if handed.classification[0].label.lower().startswith('l') else "right"
                    hand_lm = hres.multi_hand_landmarks[idx]
                    
                    ix = int(hand_lm.landmark[8].x * W)
                    iy = int(hand_lm.landmark[8].y * H)
                    tx = int(hand_lm.landmark[4].x * W)
                    ty = int(hand_lm.landmark[4].y * H)
                    if self.is_point_in_box((ix, iy), left_btn):
                        hand_current_inside["left"] = True
                    if self.is_point_in_box((ix, iy), right_btn):
                        hand_current_inside["right"] = True
                    pinch = math.hypot(ix - tx, iy - ty) < PINCH_THRESHOLD

                    # controls
                    if label == "left":
                        if hand_current_inside["left"] and not hand_was_inside["left"] and (now - self.last_button_time) > BUTTON_COOLDOWN:
                            self.cur_id = (self.cur_id - 1) % self.num if self.num else 0
                            self.last_button_time = now
                            _pos_x_q.clear(); _pos_y_q.clear(); _scale_q.clear()
                        if pinch and hand_current_inside["left"] and (now - self.last_button_time) > BUTTON_COOLDOWN:
                            if pinch_start["left"] is None:
                                pinch_start["left"] = now
                            elapsed = now - pinch_start["left"]
                            if elapsed >= PINCH_HOLD_TIME:
                                self.cur_id = (self.cur_id - 1) % self.num if self.num else 0
                                self.last_button_time = now
                                pinch_start["left"] = None
                                _pos_x_q.clear(); _pos_y_q.clear(); _scale_q.clear()
                        else:
                            if not pinch:
                                pinch_start["left"] = None
                    else:
                        if hand_current_inside["right"] and not hand_was_inside["right"] and (now - self.last_button_time) > BUTTON_COOLDOWN:
                            self.cur_id = (self.cur_id + 1) % self.num if self.num else 0
                            self.last_button_time = now
                            _pos_x_q.clear(); _pos_y_q.clear(); _scale_q.clear()
                        if pinch and hand_current_inside["right"] and (now - self.last_button_time) > BUTTON_COOLDOWN:
                            if pinch_start["right"] is None:
                                pinch_start["right"] = now
                            elapsed = now - pinch_start["right"]
                            if elapsed >= PINCH_HOLD_TIME:
                                self.cur_id = (self.cur_id + 1) % self.num if self.num else 0
                                self.last_button_time = now
                                pinch_start["right"] = None
                                _pos_x_q.clear(); _pos_y_q.clear(); _scale_q.clear()
                        else:
                            if not pinch:
                                pinch_start["right"] = None

                    mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            else:
                pinch_start["left"] = None
                pinch_start["right"] = None

            for s in ("left", "right"):
                hand_was_inside[s] = hand_current_inside[s]


            # encode as JPEG for streaming
            ret, jpeg = cv2.imencode(".jpg", frame)
            if ret:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()
            # small sleep to avoid 100% CPU
            time.sleep(0.01)

        hands.close()
        pose.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def start(self):
        if self.running and self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.thread = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def next_cloth(self):
        if self.num == 0:
            return
        self.cur_id = (self.cur_id + 1) % self.num
        _pos_x_q.clear(); _pos_y_q.clear(); _scale_q.clear()

    def prev_cloth(self):
        if self.num == 0:
            return
        self.cur_id = (self.cur_id - 1) % self.num
        _pos_x_q.clear(); _pos_y_q.clear(); _scale_q.clear()

    

# singleton controller
controller = CameraController()
