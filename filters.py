"""
WIP
MAY NOT EVEN RUN
"""

import cv2
import numpy as np
import socket
import struct
import random
import math

# --- Init settings ---
CFG_PATH = "../../net/yolo.cfg"
WEIGHTS_PATH = "../../net/yolo.weights"
LABELS_PATH = "../../net/labels.names"
CAMERA_INDEX = 0
INPUT_SIZE = (416, 416)
CONF_THRESH = 0.8
NMS_THRESH = 0.4
IP_ADDRESS = "127.0.0.1"
START_PORT = 9000
BUFFER_SIZE = 65535
MAX_PACKET_SIZE = 65507
FRAGMENTATION_FLAG = 0x8000

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

def detectHazmat(frame):
    H, W = frame.shape[:2]
    classIDs, confidences, boxes = model.detect(frame, CONF_THRESH, NMS_THRESH)

    for cid, conf, box in zip(classIDs, confidences, boxes):
        x, y, w, h = box
        cx = int(x + w/2)
        cy = int(y + h/2)
        ncx = cx / W
        ncy = cy / H
        color = [int(c) for c in COLORS[int(cid)]]
        label = f"{LABELS[int(cid)]}: {conf:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        text = f"({cx},{cy})  [{ncx:.2f},{ncy:.2f}]"
        cv2.putText(frame, text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    return frame

def detectCircles(frame):

    ext_circles[i][0] *= 4.0
            ext_circles[i][1] *= 4.0
            ext_circles[i][2] *= 4.0
            center = (round(ext_circles[i][0]), round(ext_circles[i][1]))
            radius = round(ext_circles[i][2])

# --- Helper function ---
def parsePacket(data):
    if len(data) < 10:
        return None, None
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
    }, bool(data[10] == 0x01), data[10:]

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
