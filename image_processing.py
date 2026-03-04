import cv2
import numpy as np
image = cv2.imread("sample.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blurred)
cv2.imshow("Edges", edges)
print("Original image loaded!")
print("Grayscale conversion done!")
print("Blurring done!")
print("Edge detection done!")
print("Press any key to close all windows...")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Step 3 Complete!")