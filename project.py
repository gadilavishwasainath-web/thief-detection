'''

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 19:36:23 2025

@author: Industry 4.0
Modified by ChatGPT for theft detection alarm system
"""

import os
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading

# ---- defaults ----
MODEL_PATH_PREFERRED = "runs/detect/train/weights/best.pt"
MODEL_FALLBACK = "yolov8n.pt"
IMGSZ_DEFAULT = 640
CONF_DEFAULT = 0.25
OUTPUT_VIDEO_DEFAULT = "yolo_webcam_output.avi"
# ------------------

# --- alarm function ---
def play_alarm():
    """Play alarm sound in a separate thread."""
    threading.Thread(target=playsound, args=("alarm.wav",), daemon=True).start()


def build_parser():
    p = argparse.ArgumentParser(description="YOLOv8 realtime webcam inference with theft alarm")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--model", "-m", type=str, default=None, help="Path to model weights (falls back to trained or demo)")
    p.add_argument("--conf", type=float, default=CONF_DEFAULT, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT, help="Inference image size")
    p.add_argument("--save-video", action="store_true", help="Save annotated output to video file")
    p.add_argument("--out", type=str, default=OUTPUT_VIDEO_DEFAULT, help="Path for saved video")
    p.add_argument("--no-display", action="store_true", help="Don't open a display window (useful on headless systems)")
    return p


def choose_model_path(arg_model):
    if arg_model:
        return arg_model
    if os.path.exists(MODEL_PATH_PREFERRED):
        return MODEL_PATH_PREFERRED
    return MODEL_FALLBACK


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def make_video_writer(out_path, fourcc, fps, frame_size):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass
    return cv2.VideoWriter(out_path, fourcc, fps, frame_size)


def main():
    args = build_parser().parse_args()
    model_path = choose_model_path(args.model)

    if model_path == MODEL_FALLBACK:
        print(f"Preferred model not found. Falling back to demo model '{MODEL_FALLBACK}'.")
    else:
        print(f"Using model: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print("Failed to load model:", e)
        return

    cap = open_camera(args.camera)
    if not cap or not cap.isOpened():
        print(f"Failed to open camera index {args.camera}. Exiting.")
        return

    save_video = args.save_video
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        print("Saving annotated video to:", args.out)

    last_time = time.time()
    fps_smooth = None
    cooldown_time = 0  # cooldown between alarms (seconds)
    last_alarm_time = 0

    print("Starting realtime detection... Press 'q' or 'Esc' to quit.")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed -- stopping.")
                break

            results = model(frame, imgsz=args.imgsz, conf=args.conf)
            if len(results) == 0:
                annotated = frame
            else:
                res0 = results[0]
                annotated = res0.plot()
                if not isinstance(annotated, np.ndarray):
                    annotated = np.array(annotated)

                # --- Detection logic for theft event ---
                boxes = res0.boxes
                if boxes is not None:
                    classes = boxes.cls.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy.cpu().numpy()

                    persons = []
                    bikes = []
                    for i, cls in enumerate(classes):
                        x1, y1, x2, y2 = xyxy[i]
                        if cls == 0:  # person
                            persons.append((x1, y1, x2, y2))
                        elif cls in [1, 3]:  # bicycle or motorcycle
                            bikes.append((x1, y1, x2, y2))

                    alert_triggered = False
                    for (px1, py1, px2, py2) in persons:
                        for (bx1, by1, bx2, by2) in bikes:
                            overlap_x1 = max(px1, bx1)
                            overlap_y1 = max(py1, by1)
                            overlap_x2 = min(px2, bx2)
                            overlap_y2 = min(py2, by2)
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                alert_triggered = True
                                break
                        if alert_triggered:
                            break

                    if alert_triggered and (time.time() - last_alarm_time > cooldown_time):
                        print("🚨 ALERT: Person is touching the bike! 🚨")
                        play_alarm()
                        last_alarm_time = time.time()

            # FPS overlay
            t1 = time.time()
            dt = max(t1 - t0, 1e-6)
            fps = 1.0 / dt
            fps_smooth = fps if fps_smooth is None else (fps_smooth * 0.9 + fps * 0.1)
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = make_video_writer(args.out, cv2.VideoWriter_fourcc(*"XVID"), 20, (w, h))
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("YOLOv8 Theft Detection", annotated)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord('q'):
                    print("Quit key pressed. Exiting.")
                    break

    finally:
        if writer is not None:
            writer.release()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 19:36:23 2025

@author: Industry 4.0
Modified by ChatGPT for theft detection alarm system with photo capture
"""

import os
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading

# ---- defaults ----
MODEL_PATH_PREFERRED = "runs/detect/train/weights/best.pt"
MODEL_FALLBACK = "yolov8n.pt"
IMGSZ_DEFAULT = 640
CONF_DEFAULT = 0.25
OUTPUT_VIDEO_DEFAULT = "yolo_webcam_output.avi"
CAPTURE_DIR = "captures"  # Folder to save theft photos
# ------------------

# --- alarm function ---
def play_alarm():
    """Play alarm sound in a separate thread."""
    threading.Thread(target=playsound, args=("alarm.wav",), daemon=True).start()


def build_parser():
    p = argparse.ArgumentParser(description="YOLOv8 realtime webcam inference with theft alarm and photo capture")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--model", "-m", type=str, default=None, help="Path to model weights (falls back to trained or demo)")
    p.add_argument("--conf", type=float, default=CONF_DEFAULT, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT, help="Inference image size")
    p.add_argument("--save-video", action="store_true", help="Save annotated output to video file")
    p.add_argument("--out", type=str, default=OUTPUT_VIDEO_DEFAULT, help="Path for saved video")
    p.add_argument("--no-display", action="store_true", help="Don't open a display window (useful on headless systems)")
    return p


def choose_model_path(arg_model):
    if arg_model:
        return arg_model
    if os.path.exists(MODEL_PATH_PREFERRED):
        return MODEL_PATH_PREFERRED
    return MODEL_FALLBACK


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def make_video_writer(out_path, fourcc, fps, frame_size):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass
    return cv2.VideoWriter(out_path, fourcc, fps, frame_size)


def save_theft_photo(frame):
    """Save frame to captures/ folder with timestamp."""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CAPTURE_DIR, f"theft_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Theft photo saved: {filename}")


def main():
    args = build_parser().parse_args()
    model_path = choose_model_path(args.model)

    if model_path == MODEL_FALLBACK:
        print(f"Preferred model not found. Falling back to demo model '{MODEL_FALLBACK}'.")
    else:
        print(f"Using model: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print("Failed to load model:", e)
        return

    cap = open_camera(args.camera)
    if not cap or not cap.isOpened():
        print(f"Failed to open camera index {args.camera}. Exiting.")
        return

    save_video = args.save_video
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        print("Saving annotated video to:", args.out)

    fps_smooth = None
    cooldown_time = 3  # seconds between alarms
    last_alarm_time = 0

    print("Starting realtime detection... Press 'q' or 'Esc' to quit.")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed -- stopping.")
                break

            results = model(frame, imgsz=args.imgsz, conf=args.conf)
            if len(results) == 0:
                annotated = frame
            else:
                res0 = results[0]
                annotated = res0.plot()
                if not isinstance(annotated, np.ndarray):
                    annotated = np.array(annotated)

                # --- Detection logic for theft event ---
                boxes = res0.boxes
                if boxes is not None:
                    classes = boxes.cls.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy.cpu().numpy()

                    persons = []
                    bikes = []
                    for i, cls in enumerate(classes):
                        x1, y1, x2, y2 = xyxy[i]
                        if cls == 0:  # person
                            persons.append((x1, y1, x2, y2))
                        elif cls in [1, 3]:  # bicycle or motorcycle
                            bikes.append((x1, y1, x2, y2))

                    alert_triggered = False
                    for (px1, py1, px2, py2) in persons:
                        for (bx1, by1, bx2, by2) in bikes:
                            overlap_x1 = max(px1, bx1)
                            overlap_y1 = max(py1, by1)
                            overlap_x2 = min(px2, bx2)
                            overlap_y2 = min(py2, by2)
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                alert_triggered = True
                                break
                        if alert_triggered:
                            break

                    if alert_triggered and (time.time() - last_alarm_time > cooldown_time):
                        print("🚨 ALERT: Person is touching the bike! 🚨")
                        play_alarm()
                        save_theft_photo(frame)
                        last_alarm_time = time.time()

            # FPS overlay
            t1 = time.time()
            dt = max(t1 - t0, 1e-6)
            fps = 1.0 / dt
            fps_smooth = fps if fps_smooth is None else (fps_smooth * 0.9 + fps * 0.1)
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = make_video_writer(args.out, cv2.VideoWriter_fourcc(*"XVID"), 20, (w, h))
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("YOLOv8 Theft Detection", annotated)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord('q'):
                    print("Quit key pressed. Exiting.")
                    break

    finally:
        if writer is not None:
            writer.release()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    '''


import os
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading

# ---- defaults ----
MODEL_PATH_PREFERRED = "runs/detect/train/weights/best.pt"
MODEL_FALLBACK = "yolov8n.pt"
IMGSZ_DEFAULT = 640
CONF_DEFAULT = 0.25
OUTPUT_VIDEO_DEFAULT = "yolo_webcam_output.avi"

# 📂 Save theft photos on Desktop
CAPTURE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "captures")
# ------------------


# --- alarm function ---
def play_alarm():
    """Play alarm sound in a separate thread."""
    threading.Thread(target=playsound, args=("alarm.wav",), daemon=True).start()


def build_parser():
    p = argparse.ArgumentParser(description="YOLOv8 realtime webcam inference with theft alarm and photo capture")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--model", "-m", type=str, default=None, help="Path to model weights (falls back to trained or demo)")
    p.add_argument("--conf", type=float, default=CONF_DEFAULT, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT, help="Inference image size")
    p.add_argument("--save-video", action="store_true", help="Save annotated output to video file")
    p.add_argument("--out", type=str, default=OUTPUT_VIDEO_DEFAULT, help="Path for saved video")
    p.add_argument("--no-display", action="store_true", help="Don't open a display window (useful on headless systems)")
    return p


def choose_model_path(arg_model):
    if arg_model:
        return arg_model
    if os.path.exists(MODEL_PATH_PREFERRED):
        return MODEL_PATH_PREFERRED
    return MODEL_FALLBACK


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def make_video_writer(out_path, fourcc, fps, frame_size):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass
    return cv2.VideoWriter(out_path, fourcc, fps, frame_size)


def save_theft_photo(frame):
    """Save frame to Desktop/captures/ folder with timestamp."""
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_DIR, f"theft_{timestamp}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"📸 Theft photo saved successfully to Desktop: {filename}")
        else:
            print(f"❌ Failed to save theft photo at: {filename}")
    except Exception as e:
        print(f"⚠️ Error while saving theft photo: {e}")


def main():
    args = build_parser().parse_args()
    model_path = choose_model_path(args.model)

    if model_path == MODEL_FALLBACK:
        print(f"Preferred model not found. Falling back to demo model '{MODEL_FALLBACK}'.")
    else:
        print(f"Using model: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print("Failed to load model:", e)
        return

    cap = open_camera(args.camera)
    if not cap or not cap.isOpened():
        print(f"Failed to open camera index {args.camera}. Exiting.")
        return

    save_video = args.save_video
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        print("Saving annotated video to:", args.out)

    fps_smooth = None
    cooldown_time = 3  # seconds between alarms
    last_alarm_time = 0

    print("Starting realtime detection... Press 'q' or 'Esc' to quit.")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed -- stopping.")
                break

            results = model(frame, imgsz=args.imgsz, conf=args.conf)
            if len(results) == 0:
                annotated = frame
            else:
                res0 = results[0]
                annotated = res0.plot()
                if not isinstance(annotated, np.ndarray):
                    annotated = np.array(annotated)

                # --- Detection logic for theft event ---
                boxes = res0.boxes
                if boxes is not None:
                    classes = boxes.cls.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy.cpu().numpy()

                    persons = []
                    bikes = []
                    for i, cls in enumerate(classes):
                        x1, y1, x2, y2 = xyxy[i]
                        if cls == 0:  # person
                            persons.append((x1, y1, x2, y2))
                        elif cls in [1, 3]:  # bicycle or motorcycle
                            bikes.append((x1, y1, x2, y2))

                    alert_triggered = False
                    for (px1, py1, px2, py2) in persons:
                        for (bx1, by1, bx2, by2) in bikes:
                            overlap_x1 = max(px1, bx1)
                            overlap_y1 = max(py1, by1)
                            overlap_x2 = min(px2, bx2)
                            overlap_y2 = min(py2, by2)
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                alert_triggered = True
                                break
                        if alert_triggered:
                            break

                    if alert_triggered and (time.time() - last_alarm_time > cooldown_time):
                        print("🚨 ALERT: Person is touching the bike! 🚨")
                        play_alarm()
                        save_theft_photo(frame)
                        last_alarm_time = time.time()

            # FPS overlay
            t1 = time.time()
            dt = max(t1 - t0, 1e-6)
            fps = 1.0 / dt
            fps_smooth = fps if fps_smooth is None else (fps_smooth * 0.9 + fps * 0.1)
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = make_video_writer(args.out, cv2.VideoWriter_fourcc(*"XVID"), 20, (w, h))
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("YOLOv8 Theft Detection", annotated)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord('q'):
                    print("Quit key pressed. Exiting.")
                    break

    finally:
        if writer is not None:
            writer.release()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
