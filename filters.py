"""
Full filters implementation
Robotec 2025

Deprecated
"""

import cv2
import numpy as np
import socket
import random
import math
import threading
import time

# --- Init settings ---
CFG_PATH = "visual_net/yolo.cfg"
WEIGHTS_PATH = "visual_net/yolo.weights"
LABELS_PATH = "visual_net/labels.names"
CAMERA_INDEX = 0
INPUT_SIZE = (416, 416)
CONF_THRESH = 0.8
NMS_THRESH = 0.4
IP_ADDRESS = "127.0.0.1"
START_PORT = 9000
BUFFER_SIZE = 65535
MAX_PACKET_SIZE = 65507
FRAGMENTATION_FLAG = 0x8000

# --- Helper functions ---
def parsePacket(data):
    if len(data) < 10:
        return None, None, None
    """
    fields = struct.unpack('!5H', data[:10])
    bitfield = fields[0]
    m = fields[1]
    seq = fields[2]
    timestamp = fields[3]
    ssrc = fields[4]
    cc = (bitfield >> 12) & 0x0F
    x = (bitfield >> 11) & 0x01
    p = (bitfield >> 10) & 0x01
    version = (bitfield >> 8) & 0x03
    pt = (bitfield >> 7) & 0x01
    return{
        'cc': cc,
        'x': x,
        'p': p,
        'version': version,
        'pt': pt,
        'm': m,
        'seq': seq,
        'timestamp': timestamp,
        'ssrc': ssrc,
        'is_fragmented': bool(seq & FRAGMENTATION_FLAG),
        'fragment_index': seq & ~FRAGMENTATION_FLAG
    }, timestamp/256, data[10:]
    """
    first_byte = data[0]
    version = first_byte & 0x03  # Lower 2 bits
    p = (first_byte >> 2) & 0x01  # 3rd bit
    x = (first_byte >> 3) & 0x01  # 4th bit
    cc = (first_byte >> 4) & 0x0F  # Upper 4 bits
    
    # Second byte: pt (1 bit)
    second_byte = data[1]
    pt = second_byte & 0x01  # Lower 1 bit
    
    # m, seq, timestamp, ssrc (each 2 bytes)
    m = int.from_bytes(data[2:4], byteorder='little', signed=False)
    seq = int.from_bytes(data[4:6], byteorder='little', signed=False)
    timestamp = int.from_bytes(data[6:8], byteorder='little', signed=False)
    ssrc = int.from_bytes(data[8:10], byteorder='little', signed=False)
    
    return{
        'cc': cc,
        'x': x,
        'p': p,
        'version': version,
        'pt': pt,
        'm': m,
        'seq': seq,
        'timestamp': timestamp,
        'ssrc': ssrc,
        'is_fragmented': bool(seq & FRAGMENTATION_FLAG),
        'fragment_index': seq & ~FRAGMENTATION_FLAG
    }, timestamp, data[10:]

def placeText(text, frame):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 3)
    cv2.putText(frame, text, ((frame.shape[1]-text_size[0])//2, (frame.shape[0]-text_size[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 255), 3)
    return frame
    
# --- Stream handler class ---
class RTPStreamHandler:
    def __init__(self, idx):
        print("channel: ", START_PORT + idx*2 + 1)
        self.idx = idx
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind((IP_ADDRESS, START_PORT + idx*2 + 1))
        self.recv_socket.settimeout(1.0)
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.is_recv_running = threading.Event()
        self.is_recv_running.set()

        self.frame_lock = threading.Lock()
        self.recv_lock = threading.Lock()
        self.latest_frame = []
        self.latest_compressed = []

        self.recv_thread = threading.Thread(target=self.recvPacket)
        self.recv_thread.start()

    def sendPacket(self, frame):
        if frame is not None:
            _, compressed = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            max_size = MAX_PACKET_SIZE - 10
            num_fragments = (len(compressed) + max_size - 1) // max_size
            ssrc = random.randint(1000, 9999)
            for i in range(num_fragments):
                payload = compressed[i*max_size : min((i+1)*max_size, len(compressed))]
                
                seq = i & 0x7FFF
                header = bytearray(10)
                header[0] = ((4 & 0x0F) << 4) | ((1 & 0x01) << 3) | ((1 & 0x01) << 2) | (2 & 0x03)
                header[1] = (1 & 0x01)
                header[2:4] = (num_fragments & 0xFFFF).to_bytes(2, byteorder='little', signed=False)
                if num_fragments > 1:
                    seq |= FRAGMENTATION_FLAG
                
                header[4:6] = seq.to_bytes(2, byteorder='little', signed=False)
                header[6:8] = (0 & 0xFFFF).to_bytes(2, byteorder='little', signed=False)
                header[8:10] = (ssrc & 0xFFFF).to_bytes(2, byteorder='little', signed=False)

                packet = header + payload.tobytes()
                self.send_socket.sendto(packet, (IP_ADDRESS, START_PORT + self.idx*2))

    def recvPacket(self):
        while self.is_recv_running.is_set():
            packet = bytearray()
            fragments = []
            marker = None
            ssrc = 0
            i = 0

            while True:
                try:
                    packet, _ = self.recv_socket.recvfrom(BUFFER_SIZE)
                except socket.timeout:
                    continue 
                
                if len(packet) < 10: 
                    print("packer size error")   
                    packet = None
                    break

                header, marker, payload = parsePacket(packet)
                index = header['fragment_index']

                if header is None:
                    print("no header")
                    break
                elif marker == 0:
                    print("shutdown received")
                    packet = 0
                    is_active.clear()
                    break
                elif not header['is_fragmented']:
                    packet = payload
                    break

                if i == 0:
                    num_fragments = header['m']
                    fragments = [b''] * num_fragments
                    ssrc = header['ssrc']
                elif ssrc != header['ssrc']:
                    i = 0
                    num_fragments = header['m']
                    fragments = [b''] * num_fragments
                    ssrc = header['ssrc']

                if index >= num_fragments:
                    print("frag index error")
                    break

                fragments[index] = payload

                if i == num_fragments - 1:
                    packet = bytearray()
                    for frag in fragments:
                        packet.extend(frag)
                    break

                i += 1


            if packet:
                np_data = np.frombuffer(packet, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    if marker == 1:
                        frame = detectQR(frame)
                    elif marker == 2:
                        frame = detectHazmat(frame)
                    elif marker == 3:
                        frame = detectShapeHough(frame)
                    elif marker == 4:
                        frame = detectCircles1(frame)
                    elif marker == 5:
                        frame = detectCircles2(frame)
                    else:
                        print("[w] Recv socket marker error: ", marker, type(marker))
                        continue

                    self.sendPacket(frame)
                else:
                    print("[w] Recv error. Frame decode")

            elif packet == 0:
                print("[i] GUI closed")
                break
            else:
                print("[w] Recv packet no payload")
                continue

    def close(self):
        print(f"[i] Closing channel {self.idx}...")
        self.is_recv_running.clear()
        self.recv_thread.join()
        self.recv_socket.close()
        self.send_socket.close()
    
# --- Filter functions ---
def detectQR(frame):
    qr_detector = cv2.QRCodeDetector()
    decoded_text, points, _ = qr_detector.detectAndDecode(frame)

    if points is not None:
        points = points.astype(int).reshape((-1, 1, 2))

        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=5)
        cv2.putText(frame, decoded_text, (points[0][0][0], points[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    return frame

def detectHazmat(frame):
    classIDs = []
    confidences = []
    boxes = []
    with model_lock:
        classIDs, confidences, boxes = model.detect(frame, CONF_THRESH, NMS_THRESH)
    
    for cid, conf, box in zip(classIDs, confidences, boxes):
        x, y, w, h = box
        color = [int(c) for c in COLORS[int(cid)]]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{LABELS[int(cid)]}: {conf:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    return frame

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
            dis = center[0]**2 + (frame.shape[0] - center[1])**2

            if dis < min_dis:
                min_dis = dis
                ext_sector = ext_circles[i]

    if min_dis == float('inf'):
        return frame
    
    ext_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ext_mask, (round(ext_sector[0]), round(ext_sector[1])), round(ext_sector[2]), 255, -1)
    temp = np.zeros_like(gray_frame)
    temp = cv2.bitwise_and(gray_frame, gray_frame, mask=ext_mask)
    ext_box = cv2.boundingRect(ext_mask)
    frame_roi = temp[ext_box[1]:ext_box[1]+ext_box[3], ext_box[0]:ext_box[0]+ext_box[2]]

    if frame_roi.size == 0 or frame_roi.shape[0] < 8 or frame_roi.shape[1] < 8:
        return frame

    ext_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)    

    if ext_circles is not None:
        ext_circles = ext_circles[0]  
    else:
        return frame
    
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
            box = (box[0] + ext_box[0] - 10, box[1] + ext_box[1] - 10, box[2] + 20, box[3] + 20)
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 8)
    
    return frame

def detectCircles1(frame):
    scale = 4
    rad_checks = 20

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = cv2.resize(gray_frame, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_AREA)
    ext_circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, temp.shape[0]//8, param1=100, param2=50, minRadius=temp.shape[0]//8, maxRadius=temp.shape[0]//4)
    
    min_dis = float('inf')
    ext_sector = None
    if ext_circles is not None:
        ext_circles = ext_circles[0]

        for i in range(len(ext_circles)):
            ext_circles[i][0] *= scale
            ext_circles[i][1] *= scale
            ext_circles[i][2] *= scale
            center = (round(ext_circles[i][0]), round(ext_circles[i][1]))
            radius = round(ext_circles[i][2])
            dis = (frame.shape[1]-center[0]**2) + (frame.shape[0] - center[1])**2

            if dis < min_dis:
                min_dis = dis
                ext_sector = ext_circles[i]

    if min_dis == float('inf'):
        return frame
        #return placeText("MISSING TASK SECTOR", frame)
    
    ext_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ext_mask, (round(ext_sector[0]), round(ext_sector[1])), round(ext_sector[2]), 255, -1)
    frame_roi = np.zeros_like(gray_frame)
    frame_roi = cv2.bitwise_and(gray_frame, gray_frame, mask=ext_mask)
    x1, y1, w1, h1 = cv2.boundingRect(ext_mask)
    frame_roi = frame_roi[y1:y1+h1, x1:x1+w1]

    if frame_roi.size == 0 or frame_roi.shape[0] < 8 or frame_roi.shape[1] < 8:
        return frame
        #return placeText("MISSING TASK SECTOR", frame)

    inn_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)
    
    if inn_circles is not None:
        inn_circles = inn_circles[0]      
    else:
        return frame
        #return placeText("MISSING INNER ROI", frame)
    
    mask_roi = np.zeros(frame_roi.shape, dtype=np.uint8)
    cv2.circle(mask_roi, (round(inn_circles[0][0]), round(inn_circles[0][1])), round(inn_circles[0][2])-10, 255, -1)
    final = np.zeros_like(frame_roi)
    final = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_roi)
    x2, y2, w2, h2 = cv2.boundingRect(mask_roi)
    final = final[y2:y2+h2, x2:x2+w2]

    final = cv2.resize(final, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(final, 120, 255, cv2.THRESH_BINARY) # 70 120
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    filtered_contours.sort(key=cv2.contourArea, reverse=True)

    mini = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
    for i, cont in enumerate(filtered_contours):
        if i >= 3:
            break

        cx, cy, cw, ch = cv2.boundingRect(cont)
        aspect_ratio = float(cw) / ch

        if 0.95 < aspect_ratio < 1.05:
            (rx, ry), radius = cv2.minEnclosingCircle(cont)
            center = (int(rx), int(ry))
            angles = []
            max_empty = 0
            best_angle = 0

            for angle in np.linspace(0, 2*np.pi, 360):
                empty_count = 0

                for r in np.linspace(0, radius*1.1, rad_checks):
                    x3 = int(center[0] + r * np.cos(angle))
                    y3 = int(center[1] + r * np.sin(angle))

                    if 0 <= x3 < final.shape[1] and 0 <= y3 < final.shape[0] and cv2.pointPolygonTest(cont, (x3, y3), False) == -1:
                        empty_count += 1

                if empty_count >= max_empty:
                    max_empty = empty_count
                    best_angle = angle

                if empty_count == rad_checks:
                    angles.append(angle)

            if len(angles) > 0:
                best_angle = np.mean(angles)
            
            gap_x = int(center[0] + radius * np.cos(best_angle))
            gap_y = int(center[1] + radius * np.sin(best_angle))
            cv2.line(frame, (int(rx/scale + x1+x2), int(ry/scale + y1+y2)), (int(gap_x/scale + x1+x2), int(gap_y/scale + y1+y2)), colors[i], 8)
            cv2.line(mini, center, (int(gap_x), int(gap_y)), colors[i], 2)
    else:
        return frame
        #return placeText("MISSING RING CONTOURS", frame)

    cv2.imshow("bruh", mini)
    return frame

def detectCircles2(frame):
    scale = 4
    rad_checks = 72

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = cv2.resize(gray_frame, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_AREA)
    ext_circles = cv2.HoughCircles(temp, cv2.HOUGH_GRADIENT, 1, temp.shape[0]//8, param1=100, param2=50, minRadius=temp.shape[0]//8, maxRadius=temp.shape[0]//4)
    min_dis = float('inf')
    ext_sector = None

    if ext_circles is not None:
        ext_circles = ext_circles[0]

        for i in range(len(ext_circles)):
            ext_circles[i][0] *= scale
            ext_circles[i][1] *= scale
            ext_circles[i][2] *= scale
            center = (round(ext_circles[i][0]), round(ext_circles[i][1]))
            radius = round(ext_circles[i][2])
            dis = (frame.shape[1]-center[0]**2) + (frame.shape[0] - center[1])**2

            if dis < min_dis:
                min_dis = dis
                ext_sector = ext_circles[i]

    if min_dis == float('inf'):
        print("no outer circle")
        return frame
    
    print("flag 1")
    ext_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ext_mask, (round(ext_sector[0]), round(ext_sector[1])), round(ext_sector[2]), 255, -1)
    frame_roi = np.zeros_like(gray_frame)
    frame_roi = cv2.bitwise_and(gray_frame, gray_frame, mask=ext_mask)
    x1, y1, w1, h1 = cv2.boundingRect(ext_mask)
    frame_roi = frame_roi[y1:y1+h1, x1:x1+w1]

    if frame_roi.size == 0 or frame_roi.shape[0] < 8 or frame_roi.shape[1] < 8:
        return frame

    inn_circles = cv2.HoughCircles(frame_roi, cv2.HOUGH_GRADIENT, 1, frame_roi.shape[0]//8, param1=100, param2=50, minRadius=frame_roi.shape[0]//8, maxRadius=frame_roi.shape[0]//3)
    roi_mask = np.ones(frame_roi.shape, dtype=np.uint8) * 255
    
    print("flag 2")
    if inn_circles is not None:
        inn_circles = inn_circles[0]

        for i in range(len(inn_circles)):
            center = (round(inn_circles[i][0]), round(inn_circles[i][1]))
            radius = round(inn_circles[i][2]) + 5
            cv2.circle(roi_mask, center, radius, 0, 8)

    else:
        return frame
    
    mask_roi = np.zeros(frame_roi.shape, dtype=np.uint8)
    cv2.circle(mask_roi, (round(inn_circles[0][0]), round(inn_circles[0][1])), round(inn_circles[0][2])-10, 255, -1)
    final = np.zeros_like(frame_roi)
    final = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_roi)
    x2, y2, w2, h2 = cv2.boundingRect(mask_roi)
    final = final[y2:y2+h2, x2:x2+w2]

    print("flag 3")
    final = cv2.resize(final, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(final, 120, 255, cv2.THRESH_BINARY) # 70 120
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    filtered_contours.sort(key=cv2.contourArea, reverse=True)

    #mini = frame.copy()    
    #mini = mini[y1+y2:y1+y2+h2, x1+x2:x1+x2+w2]

    print("flag 4")
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 255), (255, 255, 255)]
    for i, cont in enumerate(filtered_contours):
        if i >= 3:
            break
        cx, cy, cw, ch = cv2.boundingRect(cont)
        aspect_ratio = float(cw) / ch
        if 0.95 < aspect_ratio < 1.05:
            (rx, ry), r = cv2.minEnclosingCircle(cont)
            center = (int(rx), int(ry))
            angles = []

            min_touch = math.inf
            best_angle = 0
            angles = []

            for angle in np.linspace(0, 2*np.pi, rad_checks):
                mask = np.zeros(final.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cont], -1, 255, -1)
                # Create line mask
                line_mask = np.zeros_like(mask)
                x3 = int(center[0] + r * np.cos(angle))
                y3 = int(center[1] + r * np.sin(angle))
                cv2.line(line_mask, center, (x3, y3), 255, 1)
                overlap = cv2.bitwise_and(mask, line_mask)
                temp = np.count_nonzero(overlap)
                if temp < min_touch:
                    min_touch = temp
                    best_angle = angle
                if temp == 0:
                    angles.append(angle)

            if len(angles) > 0:
                best_angle = np.mean(angles)
            
            gap_x = int(center[0] + r * np.cos(best_angle))
            gap_y = int(center[1] + r * np.sin(best_angle))

            cv2.line(frame, (int(rx/scale + x1+x2), int(ry/scale + y1+y2)), (int(gap_x/scale + x1+x2), int(gap_y/scale + y1+y2)), colors[i], 8)
            #cv2.line(mini, center, (gap_x), colors[i], 2)
            
    return frame


print("[i] Hi")
is_active = threading.Event()
is_active.set()
with open(LABELS_PATH) as f:
    LABELS = [l.strip() for l in f if l.strip()]
np.random.seed(42)
COLORS = np.random.randint(0, 255, (len(LABELS), 3), dtype="uint8")

model_lock = threading.Lock()
model = cv2.dnn_DetectionModel(CFG_PATH, WEIGHTS_PATH)
model.setInputParams(scale=1/255.0, size=INPUT_SIZE, swapRB=True)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

filter_channels = [RTPStreamHandler(i) for i in range(4)]

while is_active.is_set():
    time.sleep(1)

print("[i] Closing program...")
for channel in filter_channels:
    channel.close()
print("[i] Bye")



"""
# --- Sockets ---
recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_socket.bind((IP_ADDRESS, START_PORT + 1))
recv_socket.settimeout(1.0)
send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- DNN model setup ---
with open(LABELS_PATH) as f:
    LABELS = [l.strip() for l in f if l.strip()]
np.random.seed(42)
COLORS = np.random.randint(0, 255, (len(LABELS), 3), dtype="uint8")
model = cv2.dnn_DetectionModel(CFG_PATH, WEIGHTS_PATH)
model.setInputParams(scale=1/255.0, size=INPUT_SIZE, swapRB=True)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# --- Processing loop ---
flag = True
#while flag:
# -- Recv + parse packet --
fragments = []
packet = bytearray()
num_fragments = -1
ssrc = -1
i = 0
frame = []

# -- Fragmentation --
while True:
    try:
        packet, _ = recv_socket.recvfrom(BUFFER_SIZE)
    except socket.timeout:
        continue 
    if len(packet) < 10:
        break
    print(flag)
    header, flag, payload = parsePacket(packet)
    print(flag)
    input()
    if header is None:
        break
    elif len(payload) < 2:
        packet = None
        flag = False
        break
    elif not header['is_fragmented']:
        packet = payload
        break

    if i == 0:
        num_fragments = header['m']
        ssrc = header['ssrc']
        fragments = [b''] * num_fragments
    if ssrc != header['ssrc']:
        i = 0
        fragments.clear()
        num_fragments = header['m']
        ssrc = header['ssrc']
        fragments = [b''] * num_fragments

    index = header['fragment_index']
    if index >= num_fragments:
        break
    fragments[index] = payload
    if i == num_fragments - 1:
        packet = bytearray()
        for frag in fragments:
            packet.extend(frag)
        break

    i += 1

# -- Build frame --
if packet:
    np_data = np.frombuffer(packet, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if frame is None:
        print("no frame")
        #continue
else:
    print("no packet")
    #continue

# filter here

_, compressed = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

# -- Send --
max_size = MAX_PACKET_SIZE - 10
num_fragments = (len(compressed) + max_size - 1) // max_size
ssrc = 2
for i in range(num_fragments):
    payload = compressed[i*max_size : min(max_size, len(compressed)-(i*max_size))]
    cc = 4
    x = 1
    p = 1
    pt = 1
    version = 2
    timestamp = 0
    m = num_fragments
    seq = i

    bitfield = ((cc & 0x0F) << 12) | ((x & 0x01) << 11) | ((p & 0x01) << 10) | ((version & 0x03) << 8) | ((pt & 0x01) << 7)
    if num_fragments > 1:
        seq |= FRAGMENTATION_FLAG
    header = struct.pack('!5H', bitfield, num_fragments, seq, timestamp, ssrc)

    packet = header + payload.tobytes()
    send_socket.sendto(packet, (IP_ADDRESS, START_PORT))
    print(f"total: {len(packet)} payload: {len(payload)}")

recv_socket.close()
send_socket.close()
"""