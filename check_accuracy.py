from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val(
    data="PCB Defect.v1i.yolov8/data.yaml",
    split="test"
)

print("=" * 50)
print("📊 MODEL ACCURACY RESULTS:")
print("=" * 50)
print(f"mAP50: {metrics.box.map50:.1%}")
print(f"Precision: {metrics.box.mp:.1%}")
print(f"Recall: {metrics.box.mr:.1%}")
print("=" * 50)