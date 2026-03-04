import cv2
from ultralytics import YOLO
import numpy as np

# Load trained model
model = YOLO("best.pt")

# Open camera
cap = cv2.VideoCapture(0)

print("AI Camera started!")
print("👉 Show PCB to camera")
print("Press 'q' to quit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if green PCB is present
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)

    annotated_frame = frame.copy()

    if green_pixels > 5000:  # PCB detected!
        results = model.predict(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()
        defect_count = len(results[0].boxes)
        status = "PASS" if defect_count == 0 else f"FAIL - {defect_count} defects"
        color = (0, 255, 0) if defect_count == 0 else (0, 0, 255)
        cv2.putText(annotated_frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(annotated_frame, "Show PCB to camera!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("AI-AOI Live Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()