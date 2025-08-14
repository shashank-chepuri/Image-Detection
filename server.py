from flask import Flask, Response, render_template, jsonify
import cv2
import torch
import requests
import time

# ✅ Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# ✅ Open Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ No working camera found!")

# ✅ Flask App
app = Flask(__name__)

# ✅ Telegram Bot Details (Replace with real values)
TELEGRAM_BOT_TOKEN = "AAEUatJIFK9VDu4F-K0efUMN4BBxb0PV99o"
CHAT_ID = "1393005905"

last_alert_time = 0  # Prevent alert spam
last_detected_animal = "Detecting..."  # Store last detected animal


def send_alert(frame, label):
    """ Sends an alert to Telegram with detected image """
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time < 30:  # Prevent spam
        return  

    last_alert_time = current_time  # Update last alert time

    cv2.imwrite("detected.jpg", frame)  # Save frame
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open("detected.jpg", "rb") as img:
        requests.post(url, files={"photo": img}, data={"chat_id": CHAT_ID, "caption": f"⚠️ Wild Animal Detected: {label}"})


def generate_frames():
    """ Continuously captures frames, detects objects, and streams them """
    global last_detected_animal

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO Object Detection
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Convert to Pandas DataFrame

        wild_animals = [ "dog", "cat", "cow", "horse", "bear", "elephant", "zebra", "lion", "tiger"]
        detected = False
        detected_label = "Detecting..."

        for _, row in detections.iterrows():
            label = row['name']
            confidence = row['confidence']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            if label in wild_animals and confidence > 0.5:
                detected = True
                detected_label = label  # Update last detected animal

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if detected:
            last_detected_animal = detected_label
            send_alert(frame, last_detected_animal)  # Send Telegram Alert

        # Encode frame for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/')
def index():
    """ Render the HTML page """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """ Stream video continuously """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_animal')
def get_animal():
    """ Returns the last detected animal """
    return jsonify({'animal': last_detected_animal})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
