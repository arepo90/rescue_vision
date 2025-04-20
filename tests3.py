import cv2
import numpy as np

img = cv2.imread('assets/circles4.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = img.copy()
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour = max(contours, key=cv2.contourArea)
(x, y), radius = cv2.minEnclosingCircle(contour)
center = (int(x), int(y))

max_empty = 0
best_angle = 0
for angle in np.linspace(0, 2*np.pi, 360):
    empty_count = 0
    for r in np.linspace(radius*0.8, radius*1.2, 10):
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if img[y, x] < 127:
                empty_count += 1
    if empty_count > max_empty:
        max_empty = empty_count
        best_angle = angle

gap_x = int(center[0] + radius * np.cos(best_angle))
gap_y = int(center[1] + radius * np.sin(best_angle))
cv2.line(result, center, (gap_x, gap_y), (0, 255, 0), 2)

cv2.imshow("final", result)

cv2.waitKey(0)