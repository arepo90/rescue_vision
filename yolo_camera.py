import cv2
import numpy as np
import time

# Conf.
CFG_PATH     = "net/yolo.cfg"
WEIGHTS_PATH = "net/yolo.weights"
LABELS_PATH  = "net/labels.names"
CAMERA_INDEX = 2
INPUT_SIZE   = (416, 416)
CONF_THRESH  = 0.8
NMS_THRESH   = 0.4

# Clase | Color
with open(LABELS_PATH) as f:
    LABELS = [l.strip() for l in f if l.strip()]
np.random.seed(42)
COLORS = np.random.randint(0, 255, (len(LABELS), 3), dtype="uint8")

# Modelo
model = cv2.dnn_DetectionModel(CFG_PATH, WEIGHTS_PATH)
model.setInputParams(scale=1/255.0, size=INPUT_SIZE, swapRB=True)

# OpenVINO:
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# CPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Cámara
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    t0 = time.time()
    classIDs, confidences, boxes = model.detect(frame, CONF_THRESH, NMS_THRESH)
    t1 = time.time()

    for cid, conf, box in zip(classIDs, confidences, boxes):
        x, y, w, h = box
        # Centroide
        cx = int(x + w/2)
        cy = int(y + h/2)
        # Normalizar
        ncx = cx / W
        ncy = cy / H

        color = [int(c) for c in COLORS[int(cid)]]
        label = f"{LABELS[int(cid)]}: {conf:.2f}"
        # Dibujar bbox + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Dibujar Centroide
        cv2.circle(frame, (cx, cy), 4, color, -1)
        # Anotar coordenadas
        txt = f"({cx},{cy})  [{ncx:.2f},{ncy:.2f}]"
        cv2.putText(frame, txt, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # FPS
    fps = 1.0 / (t1 - t0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Hazmat", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
