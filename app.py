'''
from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import time
from playsound import playsound
import threading

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # lightweight YOLO model

camera_url = "http://192.168.1.8:8080/video"
# replace with your IP camera link
cap = cv2.VideoCapture(camera_url)

def play_alarm():
    playsound("static/alarm.wav")

def detect_objects():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame)
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.6 and cls == 0:  # person detected
                print("⚠️ Possible Theft Detected!")
                cv2.imwrite(f"theft_{time.time()}.jpg", frame)
                threading.Thread(target=play_alarm).start()
        time.sleep(0.5)

threading.Thread(target=detect_objects, daemon=True).start()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
'''

from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import time
import threading
import os
from playsound import playsound

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Change this to your phone camera URL (from IP Webcam app)
# Example: "http://192.168.1.8:8080/video"
camera_url = "http://192.168.1.8:8080/video"

# Fallback to laptop camera if phone not available
def open_camera():
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("⚠️ Couldn't connect to phone camera. Using default webcam instead.")
        cap = cv2.VideoCapture(0)
    return cap

cap = open_camera()

# Alarm sound
def play_alarm():
    try:
        playsound("static/alarm.wav")
    except Exception as e:
        print("⚠️ Alarm sound failed:", e)

# Folder to save images
CAPTURE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Theft detection
def detect_objects():
    last_alarm_time = 0
    cooldown = 5  # seconds between alarms

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera feed lost. Retrying...")
            time.sleep(2)
            continue

        results = model(frame)
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Detect 'person' (COCO class 0)
            if conf > 0.6 and cls == 0:
                if time.time() - last_alarm_time > cooldown:
                    print("🚨 ALERT: Person detected near the bike!")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(CAPTURE_DIR, f"theft_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved theft photo: {filename}")
                    threading.Thread(target=play_alarm, daemon=True).start()
                    last_alarm_time = time.time()

        time.sleep(0.5)

# Run detection in a background thread
threading.Thread(target=detect_objects, daemon=True).start()

# Stream video frames to browser
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Stream interrupted, reconnecting...")
            time.sleep(1)
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("✅ Theft detection server started.")
    print("Visit: http://127.0.0.1:10000 or http://<your_wifi_ip>:10000")
    app.run(host="0.0.0.0", port=10000)
