import cv2
import numpy as np
image = cv2.imread("sample.jpg")
if image is None:
    print("Error: Image not found! Make sure sample.jpg is in the project folder")
else:
    print("Image loaded successfully!")
    print("Image Shape:", image.shape)  # height, width, channels
    print("Image Size:", image.size)
    print("Image Type:", image.dtype)
    cv2.imshow("My First AOI Image", image)
    print("Image displayed! Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done!")