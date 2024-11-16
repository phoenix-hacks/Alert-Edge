import cv2
import numpy as np

# Load YOLO model for human detection
net = cv2.dnn.readNet(r'D:\Human_line_detect\Resource\yolov3.weights', 
                      r'D:\Human_line_detect\Resource\yolov3.cfg')

# Use DroidCam IP as the video source
droidcam_ip = 'http://192.168.1.92:4747/video'  # Replace with your actual DroidCam IP
cap = cv2.VideoCapture(droidcam_ip)

# Define the x position for the vertical line crossing detection
line_x_position = 200  # Adjust based on the video

# Variable to count the number of line crossings
crossing_count = 0

# Define YOLO parameters
conf_threshold = 0.5  # Confidence threshold
nms_threshold = 0.4   # Non-maximum suppression threshold

# Load the COCO class labels YOLO was trained on
with open(r'D:\Human_line_detect\Resource\coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Get the class ID for "person" in COCO dataset (usually ID 0 for YOLO models)
person_class_id = classes.index('person')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    (frame_height, frame_width) = frame.shape[:2]

    # Create blob from the frame to pass into YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for bounding boxes and confidences
    boxes = []
    confidences = []

    # Process each output
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections for humans only and above confidence threshold
            if class_id == person_class_id and confidence > conf_threshold:
                # Scale bounding box to frame size
                box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate top-left corner coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Add bounding box and confidence to respective lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

