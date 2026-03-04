import cv2
import numpy as np
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found!")
else:
    print("Camera opened successfully!")
    print("Press 'q' to quit...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame!")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            defect_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Defect", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"Defects: {defect_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live AOI Camera", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       
cap.release()
cv2.destroyAllWindows()
print("Camera closed! Step 5 Complete!")