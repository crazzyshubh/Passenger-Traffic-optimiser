from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # Replace with the appropriate model version

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters for smoothing the crowd count
history_size = 10  # Number of recent counts to keep in history
crowd_count_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv8 to detect objects
    results = model(frame)
    
    # Parsing results
    boxes = results[0].boxes
    current_crowd_count = 0

    for box in boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Check if the detected object is a person
        if class_id == 0:  # Assuming 'person' class is at index 0 in YOLOv8
            current_crowd_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Update the crowd count history
    crowd_count_history.append(current_crowd_count)
    if len(crowd_count_history) > history_size:
        crowd_count_history.pop(0)

    # Calculate the smoothed crowd count (average of history)
    smoothed_crowd_count = int(np.mean(crowd_count_history))

    # Display the smoothed crowd count on the frame
    cv2.putText(frame, f"Crowd Count: {smoothed_crowd_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection and Crowd Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

