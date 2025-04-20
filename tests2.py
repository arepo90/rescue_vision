import cv2
import numpy as np
import math
import time

def detect_rings_and_gaps(image):

    image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)


    original = image.copy()
    output = original.copy()

    center = None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = clean.copy()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100  # Adjust as needed
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    filtered_contours.sort(key=cv2.contourArea, reverse=True)

    rings = []
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]

    for i, cont in enumerate(filtered_contours):
        if i >= len(colors):
            break
        x, y, w, h = cv2.boundingRect(cont)
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            cv2.drawContours(output, cont, -1, colors[len(rings)], 2)
            rings.append(cont)

    rings.sort(key=cv2.contourArea, reverse=True)
    for i in range(len(rings)):
        if i >= 3:
            break

        contour = rings[i]
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        angles = []
        for point in contour.squeeze():
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            angles.append(np.arctan2(dy, dx))

        angles = np.array(angles)
        angles = np.sort(angles)
        angle_diffs = np.diff(np.concatenate((angles, [angles[0] + 2*np.pi])))
        gap_mid_angle = angles[np.argmax(angle_diffs)] + np.max(angle_diffs)/2
        gap_x = int(center[0] + radius * np.cos(gap_mid_angle))
        gap_y = int(center[1] + radius * np.sin(gap_mid_angle))
        cv2.line(output, center, (gap_x, gap_y), colors[i], 2)
    else:
        print("early return")

    output = cv2.resize(output, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    return output, []

    #cv2.drawContours(output, contours, -1, (255, 0, 255), 2)


    rings = []

    
    #return output, []

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    filtered_contours.sort(key=cv2.contourArea, reverse=True)

    centx_sum = 0
    centy_sum = 0
    for cont in filtered_contours:
        x, y, w, h = cv2.boundingRect(filtered_contours[0])
        centx_sum += x + w // 2
        centy_sum += y + h // 2
    center = (centx_sum//len(filtered_contours), centy_sum//len(filtered_contours))

    
    cv2.circle(output, center, 2, (255, 0, 255), -1)
    #cv2.line(output, center, (0, 0), (255, 0, 255), 2)
    
    if len(rings) > 0:
        x, y, w, h = cv2.boundingRect(rings[0])
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05 and cv2.contourArea(rings[0]) / (image.shape[0] * image.shape[1]) > 0.9:
            rings = rings[1:]
    
    rings = rings[:3]

    
    ring_results = []
    
    for i, contour in enumerate(rings):
        if i >= len(colors):
            break
            
        M = cv2.moments(contour)
        cX = 0
        cY = 0
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue
        
        cv2.drawContours(output, [contour], -1, colors[i], 2)
        

        contour_points = contour.reshape(-1, 2)
        
        distances = [np.sqrt((x - cX)**2 + (y - cY)**2) for (x, y) in contour_points]
        
        avg_radius = np.mean(distances)
        
        threshold = 0.85  
        gap_points = [pt for pt, dist in zip(contour_points, distances) if dist < avg_radius * threshold]
        
        dx = 2*center[0]-cX - center[0]
        dy = 2*center[1]-cY - center[1]
        scale = 20.0

        x_ext = int(2*center[0]-cX + dx * scale)
        y_ext = int(2*center[1]-cY + dy * scale)
        cv2.circle(output, (2*center[0]-cX, 2*center[1]-cY), 2, colors[i], -1)
        cv2.line(output, center, (x_ext, y_ext), (0, 0, 255), 1)

        """
        # If we found potential gap points
        if gap_points:
            # Calculate the center of the gap points
            gap_cX = np.mean([pt[0] for pt in gap_points])
            gap_cY = np.mean([pt[1] for pt in gap_points])
            
            # Draw the gap center
            cv2.circle(output, (int(gap_cX), int(gap_cY)), 5, colors[i], -1)
            
            # Calculate angle of gap from center
            angle = math.degrees(math.atan2(gap_cY - cY, gap_cX - cX))
            # Normalize to 0-360
            if angle < 0:
                angle += 360
                
            # Add result
            ring_results.append({
                "ring_level": i+1,
                "center": (cX, cY),
                "gap_angle": angle,
                "radius": avg_radius
            })
            
            # Draw line from center to gap
            cv2.line(output, (cX, cY), (int(gap_cX), int(gap_cY)), colors[i], 2)
            #cv2.line(output, center, (int(gap_cX), int(gap_cY)), (255, 0, 255), 10)
        """
        
        #cv2.circle(output, (cX, cY), 5, colors[i], -1)
        
        #cv2.putText(output, f"Ring {i+1}", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    return output, ring_results

def main():
    """
    image_path = 'assets/circles2.png'  # Update this to your image path
    
    # Detect rings and gaps
    result_image, ring_results = detect_rings_and_gaps(image_path)
    
    # Display the result
    cv2.imshow('Detected Rings and Gaps', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        frame2, _ = detect_rings_and_gaps(frame)
        end = time.time()
        print(f"time: {(end-start)*1000} ms")

        if frame2 is not None:
            cv2.imshow("frame", frame2)

        cv2.waitKey(1)

    cap.release()

    
if __name__ == "__main__":
    main()