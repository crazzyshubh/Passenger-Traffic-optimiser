from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the smallest model, you can choose other versions like 'yolov8s.pt', etc.
# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv8 to detect objects
    results = model(frame)  # Pass the frame directly to the model

    # Parsing results
    boxes = results[0].boxes  # Extract bounding boxes from the results
    crowd_count = 0

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Check if the detected object is a person
        if class_id == 0:  # Assuming 'person' class is at index 0 in YOLOv8
            crowd_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the crowd count on the frame
    cv2.putText(frame, f"Crowd Count: {crowd_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection and Crowd Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
