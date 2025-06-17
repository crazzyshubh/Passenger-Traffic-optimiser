from ultralytics import YOLO

model = YOLO('yolov8n.pt')

result = model(frame)

# Count the number of "person" detections
person_count = sum(1 for r in results if r['class'] == 'person')

#  If you have specific CCTV footage from your metro station, it might help to fine-tune the YOLOv8 model to improve accuracy on your custom dataset:

