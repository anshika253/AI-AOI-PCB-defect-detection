import cv2
import numpy as np
image = cv2.imread("sample.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total defects found: {len(contours)}")
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 100:  # ignore tiny noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, f"Defect {i+1}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"Defect {i+1} found at position: x={x}, y={y}, size={area:.0f}px")
cv2.imshow("Defect Detection", image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Step 4 Complete!")  