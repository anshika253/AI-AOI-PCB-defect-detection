
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="PCB Defect.v1i.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="pcb_defect_model",
    patience=10
)

print("Training Complete!")
print("Model saved in runs/detect/pcb_defect_model/")