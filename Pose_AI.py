import cv2
import numpy as np
import threading
import time
import os
import sys
from tkinter import Tk, Button
from ultralytics import YOLO

# Глобальные переменные
cap = None
running = False
model = None

# Для доступа к встроенным ресурсам (в .exe)
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Загрузка модели
def load_model():
    global model
    model_path = resource_path("yolov8s-pose.pt")
    if not os.path.exists(model_path):
        print("Скачиваем модель YOLOv8s...")
        YOLO("yolov8s-pose.pt")  # Автоматическая загрузка
    model = YOLO(model_path)

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COLORS = {
    "head": (0, 255, 255),
    "body": (0, 255, 0),
    "arm": (255, 0, 0),
    "leg": (0, 0, 255),
    "eye_line": (255, 255, 0),
    "angle": (255, 255, 255),
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return round(np.degrees(angle), 1)

def draw_skeleton(frame, keypoints):
    keypoints = keypoints.astype(int)

    for i, (x, y) in enumerate(keypoints):
        if i <= 4:
            cv2.circle(frame, (x, y), 4, COLORS["head"], -1)
        elif 5 <= i <= 10:
            cv2.circle(frame, (x, y), 4, COLORS["arm"], -1)
        elif 11 <= i <= 12:
            cv2.circle(frame, (x, y), 4, COLORS["body"], -1)
        else:
            cv2.circle(frame, (x, y), 4, COLORS["leg"], -1)

    for start, end in POSE_CONNECTIONS:
        x1, y1 = keypoints[start]
        x2, y2 = keypoints[end]
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

    # Вектор взгляда
    try:
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        direction = (eye_center[0] - nose[0], eye_center[1] - nose[1])
        gaze_end = (nose[0] + direction[0]*2, nose[1] + direction[1]*2)
        cv2.arrowedLine(frame, tuple(nose), gaze_end, COLORS["eye_line"], 2)
    except:
        pass

    # Углы суставов
    angle_points = {
        "L-локоть": (5, 7, 9),
        "R-локоть": (6, 8, 10),
        "L-колено": (11, 13, 15),
        "R-колено": (12, 14, 16),
    }

    for label, (a, b, c) in angle_points.items():
        try:
            angle = calculate_angle(keypoints[a], keypoints[b], keypoints[c])
            x, y = keypoints[b]
            cv2.putText(frame, f"{label}: {angle}°", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["angle"], 2)
        except:
            continue

    # Выделение лица
    try:
        face_indices = [0, 1, 2, 3, 4]
        face_points = keypoints[face_indices]
        x_min = np.min(face_points[:, 0])
        y_min = np.min(face_points[:, 1])
        x_max = np.max(face_points[:, 0])
        y_max = np.max(face_points[:, 1])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)
        cv2.putText(frame, "Face", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    except:
        pass


def detect_loop():
    global cap, running

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    prev_time = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (640, 360))
        results = model.predict(resized, verbose=False, device="cpu")

        if results and results[0].keypoints is not None:
            for kp in results[0].keypoints:
                keypoints = kp.xy[0].numpy()
                draw_skeleton(resized, keypoints)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(resized, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Pose CPU", resized)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    running = False

def start_detection():
    global running
    if not running:
        running = True
        threading.Thread(target=detect_loop, daemon=True).start()

def stop_detection():
    global running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    root.quit()

# GUI
load_model()
root = Tk()
root.title("YOLOv8 Pose GUI")
root.geometry("300x120")

Button(root, text="▶ Запустить", command=start_detection, width=25, height=2, bg="green", fg="white").pack(pady=10)
Button(root, text="⛔ Выход", command=stop_detection, width=25, height=2, bg="red", fg="white").pack()

root.mainloop()
