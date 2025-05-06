"""
Experimental
WIP

"""

import threading
import cv2
import numpy as np
import time
import win32event
import win32con
import mmap
import win32api
import threading
import random
import opuslib
import queue
import vosk
import json

CFG_PATH = "visual_net/yolo.cfg"
WEIGHTS_PATH = "visual_net/yolo.weights"
LABELS_PATH = "visual_net/labels.names"
VOSK_PATH = "audio_net"
INPUT_SIZE = (416, 416)
CONF_THRESH = 0.8
NMS_THRESH = 0.4
SAMPLE_RATE = 16000             # 16 kHz
FRAME_SIZE = 960                # 960 bytes
DEVICE = None
FRAGMENTATION_FLAG = 0x8000

# --- Visual model ---
class HazmatModel:
    def __init__(self):
        self.model = cv2.dnn_DetectionModel(CFG_PATH, WEIGHTS_PATH)
        self.model.setInputParams(scale=1/255.0, size=INPUT_SIZE, swapRB=True)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detectHazmat(self, frame):
        classIDs = []
        confidences = []
        boxes = []
        classIDs, confidences, boxes = self.model.detect(frame, CONF_THRESH, NMS_THRESH)
        
        for cid, conf, box in zip(classIDs, confidences, boxes):
            x, y, w, h = box
            color = [int(c) for c in COLORS[int(cid)]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{LABELS[int(cid)]}: {conf:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        return frame
    
# --- Speech-to-text model ---
class VoskModel:
    def __init__(self):
        self.model = vosk.Model(VOSK_PATH)
        self.rec = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        self.decoder = opuslib.Decoder(SAMPLE_RATE, 1)
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_running = threading.Event()
        self.is_running.set()
        self.processing_thread = threading.Thread(target=self.processingLoop)
        self.processing_thread.start()

    def destroy(self):
        self.is_running.clear()
        self.processing_thread.join()

    def processingLoop(self):
        while self.is_running.is_set():
            try:
                packet = self.audio_queue.get(timeout=0.1)
                pcm_data = None
                try:
                    pcm_data = self.decoder.decode(packet, FRAME_SIZE)
                except opuslib.OpusError as e:
                    print(f"Opus decoding error: {e}")
                    continue

                if self.rec.AcceptWaveform(pcm_data):
                    result = json.loads(self.rec.Result())
                    if 'text' in result and result['text']:
                        self.text_queue.put(result['text'])
            except queue.Empty:
                continue

def parsePacket(data):
    if len(data) < 10:
        return None, None, None
    
    first_byte = data[0]
    second_byte = data[1]
    version = first_byte & 0x03
    p = (first_byte >> 2) & 0x01
    x = (first_byte >> 3) & 0x01
    cc = (first_byte >> 4) & 0x0F
    pt = second_byte & 0x01    
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

class CommsHandler:
    def __init__(self, name, size):
        self.size = size
        self.name = name
        self.mm = mmap.mmap(-1, size, tagname=name)
        self.mutex = win32event.OpenMutex(win32con.SYNCHRONIZE, False, name + "_mtx")

    def write(self, data: bytes):
        win32event.WaitForSingleObject(self.mutex, 100)
        self.mm.seek(0)
        self.mm.write(data)
        win32event.ReleaseMutex(self.mutex)

    def read(self):
        win32event.WaitForSingleObject(self.mutex, 100)
        self.mm.seek(0)
        data = self.mm.read(self.size)
        win32event.ReleaseMutex(self.mutex)
        return data

class Companion:
    def __init__(self, idx):
        self.idx = idx
        if idx == 4:
            self.stt = VoskModel()
        else:
            self.hazmat = HazmatModel()
        self.latest_filter = None
        self.latest_ssrc = -1
        self.filter_mutex = threading.Lock()
        self.is_send_running = threading.Event()
        self.is_recv_running = threading.Event()
        self.is_send_running.set()
        self.is_recv_running.set()
        self.handler = CommsHandler("SharedChannel"+str(idx), 100000)
        self.recv_thread = threading.Thread(target=self.recvLoop)
        self.send_thread = threading.Thread(target=self.sendLoop)
        self.recv_thread.start()
        self.send_thread.start()

    def recvLoop(self):
        while self.is_recv_running.is_set():
            packet = self.handler.read()
            header, size, payload = parsePacket(packet)
            if header["ssrc"] == self.latest_ssrc:
                time.sleep(0.5)
                continue
            self.latest_ssrc = header["ssrc"]
            if header["m"] == -1:
                print("recv disconnect")
                packet = 0
                is_active.clear()
                break
            elif header["m"] != 1 and header["m"] != 2:
                print(f"[w] Invalid marker received {header['m']}")
                continue

            payload = payload[:size]
            print(f"recv, size: {size}")

            if header["m"] == 1:
                print("recv video")
                if self.idx == 4:
                    print("[w] Video marker called on audio channel")
                    continue
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                frame = self.hazmat.detectHazmat(frame)
                with self.filter_mutex:
                    self.latest_filter = frame
            else:
                print("recv audio")
                if self.idx != 4:
                    print("[w] Audio marker called on video channel")
                    continue
                self.stt.audio_queue.put(payload)
            time.sleep(0.01)

    def sendLoop(self):
        while self.is_send_running.is_set():
            ssrc = random.randint(10000, 99999)
            header = bytearray(10)
            header[0] = ((4 & 0x0F) << 4) | ((1 & 0x01) << 3) | ((1 & 0x01) << 2) | (2 & 0x03)
            header[1] = (1 & 0x01)
            header[2:4] = (1 & 0xFFFF).to_bytes(2, byteorder='little', signed=False)
            header[4:6] = (0 & 0x7FFF).to_bytes(2, byteorder='little', signed=False)
            header[8:10] = (ssrc & 0xFFFF).to_bytes(2, byteorder='little', signed=False)
            if self.idx != 4:
                with self.filter_mutex:
                    if self.latest_filter is not None:
                        _, encoded = cv2.imencode('.jpg', self.latest_filter, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        header[6:8] = ((10+len(encoded)) & 0xFFFF).to_bytes(2, byteorder='little', signed=False)
                        packet = header + encoded.tobytes()
                        self.handler.write(packet)
                        print("sent video")
                time.sleep(0.5)
            else:
                try:
                    text = self.stt.text_queue.get(timeout=0.1)
                    if text:
                        header[6:8] = (10+len(text) & 0xFFFF).to_bytes(2, byteorder='little', signed=False)
                        packet = header + text.encode("utf-8")
                        self.handler.write(packet)
                        print(f"Sent text: {text}, text size: {len(text)}, total: {10+len(text)}, ssrc: {ssrc}")
                        time.sleep(1)
                except queue.Empty:
                    continue

    def destroy(self):
        self.is_send_running.clear()
        self.is_recv_running.clear()
        self.recv_thread.join()
        self.send_thread.join()
        if self.idx == 4:
            self.stt.destroy()
        print(f"destroyed {self.idx}")


with open(LABELS_PATH) as f:
    LABELS = [l.strip() for l in f if l.strip()]
np.random.seed(42)
COLORS = np.random.randint(0, 255, (len(LABELS), 3), dtype="uint8")
is_active = threading.Event()
is_active.set()
channels = [Companion(i) for i in range(5)]
print("start")

while is_active.is_set():
    time.sleep(1)

#time.sleep(10)
print("end")
for channel in channels:
    channel.destroy() 
print("done")