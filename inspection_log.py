import cv2
from ultralytics import YOLO
import csv
import datetime
import os

# Load trained model
model = YOLO("best.pt")

# Create CSV log file
log_file = "inspection_log.csv"

# Create file with headers if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Defects Found", "Status", "Defect Details"])
    print("Log file created!")

# Open camera
cap = cv2.VideoCapture(0)
print("Camera started!")
print("Press 's' to save inspection result")
print("Press 'q' to quit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run AI detection
    results = model.predict(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    defect_count = len(results[0].boxes)
    status = "PASS" if defect_count == 0 else "FAIL"

    # Show status
    color = (0, 255, 0) if defect_count == 0 else (0, 0, 255)
    cv2.putText(annotated_frame, f"{status} - Defects: {defect_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(annotated_frame, "Press 'S' to save log", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("AI-AOI Live Camera", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # Press S to save log
    if key == ord('s'):
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        defect_details = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            defect_details.append(f"{label}({conf:.2f})")

        details = ", ".join(defect_details) if defect_details else "None"

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([date, time, defect_count, status, details])

        print(f"Log saved! {date} {time} - {status} - {defect_count} defects")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Inspection complete! Check inspection_log.csv!")
