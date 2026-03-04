from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# Run prediction on sample image
results = model.predict(
    source="pcb_test.jpg",
    conf=0.25,
    save=True
)

# Show results
for result in results:
    boxes = result.boxes
    print(f"Total defects detected: {len(boxes)}")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"Defect: {label}, Confidence: {conf:.2f}")

print("Done! Check runs/detect/predict folder for result image!")
