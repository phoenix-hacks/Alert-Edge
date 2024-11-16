import cv2
import torch
from ultralytics import YOLO
from sort import Sort  # Import the SORT tracker
import winsound  # To generate sound alerts (Windows only)

# Load YOLOv8 lightweight model
model = YOLO('yolov8n.pt')  # Use nano version for better speed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Set up video source (DroidCam IP)
video_source = 'http://192.168.25.217:4747/video'  # Replace with your DroidCam IP
cap = cv2.VideoCapture(video_source)

# Parameters for line crossing
line_x_position = 300  # X-coordinate of the blue line
frame_skip = 2  # Skip frames for better performance
frame_counter = 0

# Function to trigger a sound alert
def play_alert_sound():
    frequency = 1000  # Set frequency to 1000 Hz
    duration = 1000  # Set duration to 1 second
    winsound.Beep(frequency, duration)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video stream disconnected or unable to fetch frames.")
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue  # Skip frames for faster processing

    # Perform YOLO inference
    results = model.predict(frame, conf=0.5, device=device, stream=False)

    # Extract detections for 'person' class
    detections = []
    for detection in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        if int(class_id) == 0:  # Class 'person'
            detections.append([x1, y1, x2, y2, confidence])

    # Convert detections to numpy array
    detections = torch.tensor(detections) if detections else torch.empty((0, 5))

    # Update tracker
    tracks = tracker.update(detections)

    # Process tracks
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        center_x = (x1 + x2) // 2

        # Draw bounding box without ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Check if the person crosses the line from left to right
        if center_x > line_x_position:
            # Trigger an alert sound for the station controller
            play_alert_sound()
            # Display alert message on the screen
            cv2.putText(frame, "ALERT: Person crossed the safety line", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw crossing line (blue)
    cv2.line(frame, (line_x_position, 0), (line_x_position, frame.shape[0]), (255, 0, 0), 2)

    # Show the video feed
    cv2.imshow('Vertical Line Crossing Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
