"""
TESTS
WIP
"""

import time
import cv2
import numpy as np

def detectShapeHough(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = cv2.resize(gray_frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    ext_circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, temp.shape[0]//8, param1=100, param2=50, minRadius=temp.shape[0]//8, maxRadius=temp.shape[0]//4)
    min_dis = float('inf')
    ext_sector = None

    if ext_circles is not None:
        ext_circles = ext_circles[0]

        for i in range(len(ext_circles)):
            ext_circles[i][0] *= 4.0
            ext_circles[i][1] *= 4.0
            ext_circles[i][2] *= 4.0
            center = (round(ext_circles[i][0]), round(ext_circles[i][1]))
            radius = round(ext_circles[i][2])
            dis = center[0]**2 + (frame.shape[0] - center[1])**2

            if dis < min_dis:
                min_dis = dis
                ext_sector = ext_circles[i]

    if min_dis == float('inf'):
        print("no outer circle")
        return None
    
    ext_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ext_mask, (round(ext_sector[0]), round(ext_sector[1])), round(ext_sector[2]), 255, -1)
    temp = np.zeros_like(gray_frame)
    temp = cv2.bitwise_and(gray_frame, gray_frame, mask=ext_mask)
    ext_box = cv2.boundingRect(ext_mask)
    frame_roi = temp[ext_box[1]:ext_box[1]+ext_box[3], ext_box[0]:ext_box[0]+ext_box[2]]

    if frame_roi.size == 0 or frame_roi.shape[0] < 8 or frame_roi.shape[1] < 8:
        print("empty final")
        return None

    ext_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)
    roi_mask = np.ones(frame_roi.shape, dtype=np.uint8) * 255
    
    if ext_circles is not None:
        ext_circles = ext_circles[0]

        for i in range(len(ext_circles)):
            center = (round(ext_circles[i][0]), round(ext_circles[i][1]))
            radius = round(ext_circles[i][2]) + 5
            cv2.circle(roi_mask, center, radius, 0, 8)
            center_on_frame = (center[0] + ext_box[0], center[1] + ext_box[1])
            #cv2.circle(frame, center_on_frame, radius, (255, 0, 0), 2)
    else:
        print("no inner circles")
        return None
    
    mask_roi = np.zeros(frame_roi.shape, dtype=np.uint8)
    cv2.circle(mask_roi, (round(ext_circles[0][0]), round(ext_circles[0][1])), round(ext_circles[0][2])-10, 255, -1)
    final = np.zeros_like(frame_roi)
    final = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_roi)
    _, final_thresh = cv2.threshold(final, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)

        if area <= 10.0:
            continue

        bound_rect = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        aspect_ratio = float(bound_rect[2]) / bound_rect[3]
        solidity = area / cv2.contourArea(hull)

        if 0.5 < aspect_ratio < 1.5 and solidity > 0.5:
            filtered_contours.append(contour)
    
    if filtered_contours:
        center = (final.shape[1] // 2, final.shape[0] // 2)
        min_distance = float('inf')
        best_contour_idx = -1

        for i, contour in enumerate(filtered_contours):
            M = cv2.moments(contour)

            if M['m00'] != 0:
                center_of_mass = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                distance = np.linalg.norm(np.array(center_of_mass) - np.array(center))
                
                if distance < min_distance:
                    min_distance = distance
                    best_contour_idx = i
        
        if best_contour_idx >= 0:
            box = cv2.boundingRect(filtered_contours[best_contour_idx])
            box = (
                box[0] + ext_box[0] - 10,
                box[1] + ext_box[1] - 10,
                box[2] + 20,
                box[3] + 20
            )
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    
    return frame

def detectCircles(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi_gray = gray_frame.copy()
    mask = np.ones_like(roi_gray, dtype=np.uint8) * 255
    circles_found = []

    offset_x, offset_y = 0, 0
    i = 0
    while i < 2:
        outer = None
        inner = None
        for j in range(2):
            blurred = cv2.GaussianBlur(roi_gray, (9, 9), 2)

            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=30, minRadius=blurred.shape[0]//4, maxRadius=0)
            if circles is None:
                print(f"no circles on i = {i}")
                break

            circles = np.uint16(np.around(circles))
            largest_circle = max(circles[0, :], key=lambda c: c[2])
            x, y, r = largest_circle
            if j == 0:
                outer = largest_circle
            else:
                inner = largest_circle

            mask = np.zeros_like(roi_gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            roi_gray = cv2.bitwise_and(roi_gray, roi_gray, mask=mask)
            top = max(y - r, 0)
            bottom = min(y + r, roi_gray.shape[0])
            left = max(x - r, 0)
            right = min(x + r, roi_gray.shape[1])
            roi_gray = roi_gray[top:bottom, left:right]
            offset_x += left
            offset_y += top

        if outer is None or inner is None:
            print("uh huh")
            break
        x1, y1, r1 = outer
        x2, y2, r2 = inner  
        cv2.circle(frame, (x1, y1), (r1+r2)//2, (0, 0, 255), 2)

        i += 1

    return frame


frame = cv2.imread("assets/circles2.png")
#cv2.imshow("original", frame)

start = time.time()
frame2 = detectCircles(frame)
end = time.time()
print(f"time: {(end-start)*1000} ms")

if frame2 is not None:
    cv2.imshow("frame", frame2)

cv2.waitKey(0)