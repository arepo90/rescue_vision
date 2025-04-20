"""
TESTS
WIP
"""

import time
import cv2
import numpy as np
import math

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

    inn_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)
    roi_mask = np.ones(frame_roi.shape, dtype=np.uint8) * 255
    
    if inn_circles is not None:
        inn_circles = inn_circles[0]

        for i in range(len(inn_circles)):
            center = (round(inn_circles[i][0]), round(inn_circles[i][1]))
            radius = round(inn_circles[i][2]) + 5
            cv2.circle(roi_mask, center, radius, 0, 8)
            center_on_frame = (center[0] + ext_box[0], center[1] + ext_box[1])
            #cv2.circle(frame, center_on_frame, radius, (255, 0, 0), 2)
    else:
        print("no inner circles")
        return None
    
    mask_roi = np.zeros(frame_roi.shape, dtype=np.uint8)
    cv2.circle(mask_roi, (round(inn_circles[0][0]), round(inn_circles[0][1])), round(inn_circles[0][2])-10, 255, -1)
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

def detectCircles(frame2):
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
            dis = (frame.shape[1]-center[0]**2) + (frame.shape[0] - center[1])**2

            if dis < min_dis:
                min_dis = dis
                ext_sector = ext_circles[i]

    if min_dis == float('inf'):
        print("no outer circle")
        return frame
    
    ext_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ext_mask, (round(ext_sector[0]), round(ext_sector[1])), round(ext_sector[2]), 255, -1)
    frame_roi = np.zeros_like(gray_frame)
    frame_roi = cv2.bitwise_and(gray_frame, gray_frame, mask=ext_mask)
    x1, y1, w1, h1 = cv2.boundingRect(ext_mask)
    frame_roi = frame_roi[y1:y1+h1, x1:x1+w1]

    #frame2 = np.zeros_like(frame)
    #frame2 = cv2.bitwise_and(frame, frame, mask=ext_mask)
    #frame2 = frame2[y1:y1+h1, x1:x1+w1]

    if frame_roi.size == 0 or frame_roi.shape[0] < 8 or frame_roi.shape[1] < 8:
        print("empty final")
        return frame

    inn_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)
    roi_mask = np.ones(frame_roi.shape, dtype=np.uint8) * 255
    
    if inn_circles is not None:
        inn_circles = inn_circles[0]

        for i in range(len(inn_circles)):
            center = (round(inn_circles[i][0]), round(inn_circles[i][1]))
            radius = round(inn_circles[i][2]) + 5
            cv2.circle(roi_mask, center, radius, 0, 8)
            #center_on_frame = (center[0] + x, center[1] + y)
            #cv2.circle(frame, center_on_frame, radius, (255, 0, 0), 2)
    else:
        print("no inner circles")
        return frame
    
    mask_roi = np.zeros(frame_roi.shape, dtype=np.uint8)
    cv2.circle(mask_roi, (round(inn_circles[0][0]), round(inn_circles[0][1])), round(inn_circles[0][2])-10, 255, -1)
    final = np.zeros_like(frame_roi)
    final = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_roi)
    x2, y2, w2, h2 = cv2.boundingRect(mask_roi)
    final = final[y2:y2+h2, x2:x2+w2]

    #final2 = np.zeros_like(frame2)
    #final2 = cv2.bitwise_and(frame2, frame2, mask=mask_roi)
    #final2 = final2[y2:y2+h2, x2:x2+w2]

    scale = 6
    final = cv2.resize(final, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(final, 110, 255, cv2.THRESH_BINARY) # 70 120
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    filtered_contours.sort(key=cv2.contourArea, reverse=True)
    print(len(filtered_contours))

    #final2 = cv2.resize(final2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    #cv2.drawContours(final2, filtered_contours, -1, (255, 0, 255), 2)

    # 3. Adjust the contours' coordinates to the original frame
    adjusted_contours = []
    for contour in contours:
        adjusted = contour/scale + np.array([x1+x2, y1+y2])
        adjusted = adjusted.astype(np.int32)
        adjusted_contours.append(adjusted)

    cv2.drawContours(frame, adjusted_contours, -1, (255, 0, 255), 2)

    rings = []
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
    for i, cont in enumerate(filtered_contours):
        if i >= 3:
            break
        cx, cy, cw, ch = cv2.boundingRect(cont)
        aspect_ratio = float(cw) / ch
        angles = []

        if 0.95 < aspect_ratio < 1.05:
            #cv2.drawContours(final2, cont, -1, colors[i], 2)
            (rx, ry), r = cv2.minEnclosingCircle(cont)
            center = (int(rx), int(ry))

            for point in cont.squeeze():
                dx = point[0] - center[0]
                dy = point[1] - center[1]
                angles.append(np.arctan2(dy, dx))

            angles = np.array(angles)
            angles = np.sort(angles)
            angle_diffs = np.diff(np.concatenate((angles, [angles[0] + 2*np.pi])))
            gap_mid_angle = angles[np.argmax(angle_diffs)] + np.max(angle_diffs)/2
            gap_x = int(center[0] + r * np.cos(gap_mid_angle))
            gap_y = int(center[1] + r * np.sin(gap_mid_angle))
            #cv2.circle(final2, (gap_x, gap_y), 5, colors[i], -1)
            cv2.line(frame, (int(rx/scale + x1+x2), int(ry/scale + y1+y2)), (int(gap_x/scale + x1+x2), int(gap_y/scale + y1+y2)), colors[i], 2)
            #cv2.line(final2, (int(rx), int(ry)), (gap_x, gap_y), colors[i], 2)

    #cv2.imshow("final", final2)
    return frame

def detectCircles2(frame):
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
            dis = (frame.shape[1]-center[0]**2) + (frame.shape[0] - center[1])**2

            if dis < min_dis:
                min_dis = dis
                ext_sector = ext_circles[i]

    if min_dis == float('inf'):
        print("no outer circle")
        return frame
    
    ext_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ext_mask, (round(ext_sector[0]), round(ext_sector[1])), round(ext_sector[2]), 255, -1)
    frame_roi = np.zeros_like(gray_frame)
    frame_roi = cv2.bitwise_and(gray_frame, gray_frame, mask=ext_mask)
    x1, y1, w1, h1 = cv2.boundingRect(ext_mask)
    frame_roi = frame_roi[y1:y1+h1, x1:x1+w1]

    #frame2 = np.zeros_like(frame)
    #frame2 = cv2.bitwise_and(frame, frame, mask=ext_mask)
    #frame2 = frame2[y1:y1+h1, x1:x1+w1]

    if frame_roi.size == 0 or frame_roi.shape[0] < 8 or frame_roi.shape[1] < 8:
        print("empty final")
        return frame

    inn_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)
    roi_mask = np.ones(frame_roi.shape, dtype=np.uint8) * 255
    
    if inn_circles is not None:
        inn_circles = inn_circles[0]

        for i in range(len(inn_circles)):
            center = (round(inn_circles[i][0]), round(inn_circles[i][1]))
            radius = round(inn_circles[i][2]) + 5
            cv2.circle(roi_mask, center, radius, 0, 8)
            #center_on_frame = (center[0] + x, center[1] + y)
            #cv2.circle(frame, center_on_frame, radius, (255, 0, 0), 2)
    else:
        print("no inner circles")
        return frame
    
    mask_roi = np.zeros(frame_roi.shape, dtype=np.uint8)
    cv2.circle(mask_roi, (round(inn_circles[0][0]), round(inn_circles[0][1])), round(inn_circles[0][2])-10, 255, -1)
    final = np.zeros_like(frame_roi)
    final = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_roi)
    x2, y2, w2, h2 = cv2.boundingRect(mask_roi)
    final = final[y2:y2+h2, x2:x2+w2]

    #final2 = np.zeros_like(frame2)
    #final2 = cv2.bitwise_and(frame2, frame2, mask=mask_roi)
    #final2 = final2[y2:y2+h2, x2:x2+w2]

    scale = 4
    final = cv2.resize(final, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(final, 120, 255, cv2.THRESH_BINARY) # 70 120
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    filtered_contours.sort(key=cv2.contourArea, reverse=True)
    print(len(filtered_contours))

    #final2 = cv2.resize(final2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    #cv2.drawContours(final2, filtered_contours, -1, (255, 0, 255), 2)

    rings = []
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
    for i, cont in enumerate(filtered_contours):
        if i >= 3:
            break
        cx, cy, cw, ch = cv2.boundingRect(cont)
        aspect_ratio = float(cw) / ch
        if 0.95 < aspect_ratio < 1.05:
            #cv2.drawContours(final2, cont, -1, colors[i], 2)
            (rx, ry), radius = cv2.minEnclosingCircle(cont)
            center = (int(rx), int(ry))
            angles = []

            max_empty = 0
            best_angle = 0
            angles = []
            for angle in np.linspace(0, 2*np.pi, 360):
                empty_count = 0
                #for r in np.linspace(radius*0.8, radius*1.2, 10):
                for r in np.linspace(0, radius*1.1, 20):
                    x3 = int(center[0] + r * np.cos(angle))
                    y3 = int(center[1] + r * np.sin(angle))
                    if 0 <= x3 < final.shape[1] and 0 <= y3 < final.shape[0]:
                        if cv2.pointPolygonTest(cont, (x3, y3), measureDist=False) == -1:
                            empty_count += 1
                if empty_count >= max_empty:
                    max_empty = empty_count
                    best_angle = angle
                if empty_count == 20:
                    angles.append(angle)
                

            if len(angles) > 0:
                best_angle = np.mean(angles)
            
            gap_x = int(center[0] + radius * np.cos(best_angle))
            gap_y = int(center[1] + radius * np.sin(best_angle))

            cv2.line(frame, (int(rx/scale + x1+x2), int(ry/scale + y1+y2)), (int(gap_x/scale + x1+x2), int(gap_y/scale + y1+y2)), colors[i], 2)
            #cv2.line(final2, center, (gap_x, gap_y), colors[i], 2)

            """
            for point in cont.squeeze():
                dx = point[0] - center[0]
                dy = point[1] - center[1]
                angles.append(np.arctan2(dy, dx))

            angles = np.array(angles)
            angles = np.sort(angles)
            angle_diffs = np.diff(np.concatenate((angles, [angles[0] + 2*np.pi])))
            gap_mid_angle = angles[np.argmax(angle_diffs)] + np.max(angle_diffs)/2
            gap_x = int(center[0] + r * np.cos(gap_mid_angle))
            gap_y = int(center[1] + r * np.sin(gap_mid_angle))
            cv2.circle(final2, (gap_x, gap_y), 5, colors[i], -1)
            cv2.line(frame, (int(rx/scale + x1+x2), int(ry/scale + y1+y2)), (int(gap_x/scale + x1+x2), int(gap_y/scale + y1+y2)), colors[i], 2)
            cv2.line(final2, (int(rx), int(ry)), (gap_x, gap_y), colors[i], 2)
            """

    #cv2.imshow("final", final2)
    return frame

def detectCircles3(frame):
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
            dis = (frame.shape[1]-center[0]**2) + (frame.shape[0] - center[1])**2

            if dis < min_dis:
                min_dis = dis
                ext_sector = ext_circles[i]

    if min_dis == float('inf'):
        print("no outer circle")
        return frame
    
    ext_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ext_mask, (round(ext_sector[0]), round(ext_sector[1])), round(ext_sector[2]), 255, -1)
    frame_roi = np.zeros_like(gray_frame)
    frame_roi = cv2.bitwise_and(gray_frame, gray_frame, mask=ext_mask)
    x1, y1, w1, h1 = cv2.boundingRect(ext_mask)
    frame_roi = frame_roi[y1:y1+h1, x1:x1+w1]

    #frame2 = np.zeros_like(frame)
    #frame2 = cv2.bitwise_and(frame, frame, mask=ext_mask)
    #frame2 = frame2[y1:y1+h1, x1:x1+w1]

    if frame_roi.size == 0 or frame_roi.shape[0] < 8 or frame_roi.shape[1] < 8:
        print("empty final")
        return frame

    inn_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)
    roi_mask = np.ones(frame_roi.shape, dtype=np.uint8) * 255
    
    if inn_circles is not None:
        inn_circles = inn_circles[0]

        for i in range(len(inn_circles)):
            center = (round(inn_circles[i][0]), round(inn_circles[i][1]))
            radius = round(inn_circles[i][2]) + 5
            cv2.circle(roi_mask, center, radius, 0, 8)
            #center_on_frame = (center[0] + x, center[1] + y)
            #cv2.circle(frame, center_on_frame, radius, (255, 0, 0), 2)
    else:
        print("no inner circles")
        return frame
    
    mask_roi = np.zeros(frame_roi.shape, dtype=np.uint8)
    cv2.circle(mask_roi, (round(inn_circles[0][0]), round(inn_circles[0][1])), round(inn_circles[0][2])-10, 255, -1)
    final = np.zeros_like(frame_roi)
    final = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_roi)
    x2, y2, w2, h2 = cv2.boundingRect(mask_roi)
    final = final[y2:y2+h2, x2:x2+w2]

    #final2 = np.zeros_like(frame2)
    #final2 = cv2.bitwise_and(frame2, frame2, mask=mask_roi)
    #final2 = final2[y2:y2+h2, x2:x2+w2]

    scale = 4
    final = cv2.resize(final, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(final, 120, 255, cv2.THRESH_BINARY) # 70 120
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    filtered_contours.sort(key=cv2.contourArea, reverse=True)
    print(len(filtered_contours))

    #final2 = cv2.resize(final2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    #cv2.drawContours(final2, filtered_contours, -1, (255, 0, 255), 2)

    rings = []
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
    for i, cont in enumerate(filtered_contours):
        if i >= 3:
            break
        cx, cy, cw, ch = cv2.boundingRect(cont)
        aspect_ratio = float(cw) / ch
        if 0.95 < aspect_ratio < 1.05:
            #cv2.drawContours(final2, cont, -1, colors[i], 2)
            (rx, ry), r = cv2.minEnclosingCircle(cont)
            center = (int(rx), int(ry))
            angles = []

            max_empty = 0
            min_touch = 99999999
            best_angle = 0
            angles = []

            start = time.time()
            for angle in np.linspace(0, 2*np.pi, 72):
                mask = np.zeros(final.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cont], -1, 255, -1)
                # Create line mask
                line_mask = np.zeros_like(mask)
                x3 = int(center[0] + r * np.cos(angle))
                y3 = int(center[1] + r * np.sin(angle))
                #print(x3, y3, center, r, angle)
                cv2.line(line_mask, center, (x3, y3), 255, 1)
                overlap = cv2.bitwise_and(mask, line_mask)
                temp = np.count_nonzero(overlap)
                if temp < min_touch:
                    min_touch = temp
                    best_angle = angle
                if temp == 0:
                    angles.append(angle)
            end = time.time()
            print(f"interior: {(end-start)*1000} ms")

            if len(angles) > 0:
                best_angle = np.mean(angles)
            
            gap_x = int(center[0] + r * np.cos(best_angle))
            gap_y = int(center[1] + r * np.sin(best_angle))

            cv2.line(frame, (int(rx/scale + x1+x2), int(ry/scale + y1+y2)), (int(gap_x/scale + x1+x2), int(gap_y/scale + y1+y2)), colors[i], 2)
            #cv2.line(final2, center, (gap_x, gap_y), colors[i], 2)

    #cv2.imshow("final", final2)
    return frame


"""
frame = cv2.imread("assets/cam2.jpg")
#cv2.imshow("original", frame)
start = time.time()
frame2 = detectCircles2(frame)
end = time.time()
print(f"time: {(end-start)*1000} ms")
if frame2 is not None:
        cv2.imshow("frame", frame2)
        cv2.waitKey(0)
"""


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    frame2 = detectCircles2(frame)
    end = time.time()
    print(f"time: {(end-start)*1000} ms")

    if frame2 is not None:
        cv2.imshow("frame", frame2)

    cv2.waitKey(1)

cap.release()
