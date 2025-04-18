import cv2
import numpy as np
import socket
import struct
import random

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

# --- Model setup ---
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

# -- Hazmat model --
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
