import cv2
import torch
from ultralytics import YOLO
from sort import Sort
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load YOLOv8 lightweight model
model = YOLO('yolov8n.pt')  # Ensure this model file is in the same directory or provide the correct path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Set up video source (DroidCam IP)
video_source = 'http://192.168.25.217:4747/video'  # Replace with your DroidCam IP
cap = cv2.VideoCapture(video_source)

# Parameters for line crossing
line_x_position = 300  # Adjust this value as needed
frame_skip = 2  # Skip frames for faster processing
frame_counter = 0

def generate_frames():
    global frame_counter
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        # Perform YOLO inference
        results = model.predict(frame, conf=0.5, device=device, stream=False)
        detections = []
        for detection in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            if int(class_id) == 0:  # Class 'person'
                detections.append([x1, y1, x2, y2, confidence])

        detections = torch.tensor(detections) if detections else torch.empty((0, 5))
        tracks = tracker.update(detections)

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            center_x = (x1 + x2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if the person crosses the line from left to right
            if center_x > line_x_position:
                cv2.putText(frame, "ALERT: Person crossed the safety line", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw crossing line
        cv2.line(frame, (line_x_position, 0), (line_x_position, frame.shape[0]), (255, 0, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Complete the app.run() statement