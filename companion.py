"""
    Post-processing companion script
    Robotec 2025
"""

import cv2
import numpy as np
import socket
import random
import threading
import time
import queue
import vosk
import json
import opuslib

# --- Init settings ---
VOSK_PATH = "audio_net"
SAMPLE_RATE = 16000             # 16 kHz
FRAME_SIZE = 960                # 960 bytes
DEVICE = None
IP_ADDRESS = "127.0.0.1"
START_PORT = 9000
BUFFER_SIZE = 65535             # 65535 bytes
MAX_PACKET_SIZE = 65507         # 65507 bytes
FRAGMENTATION_FLAG = 0x8000     # RTP Header flag

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
        self.send_thread = threading.Thread(target=self.processingLoop)
        self.send_thread.start()

    def destroy(self):
        self.is_running.clear()
        self.send_thread.join()

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

# --- Stream handler class ---
class RTPStreamHandler:
    def __init__(self):
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind((IP_ADDRESS, START_PORT + 1))
        self.recv_socket.settimeout(1.0)
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.is_recv_running = threading.Event()
        self.is_send_running = threading.Event()
        self.is_recv_running.set()
        self.is_send_running.set()
        self.recv_thread = threading.Thread(target=self.recvLoop)
        self.send_thread = threading.Thread(target=self.sendLoop)
        self.recv_thread.start()
        self.send_thread.start()
        print(f"[i] Channel created, bound to ports ({START_PORT}), ({START_PORT+1})")

    def destroy(self):
        print(f"[i] Closing channel ({START_PORT}), ({START_PORT+1})")
        self.is_recv_running.clear()
        self.is_send_running.clear()
        self.recv_thread.join()
        self.send_thread.join()
        self.recv_socket.close()
        self.send_socket.close()

    def sendLoop(self):
        while self.is_send_running.is_set():
            try:
                text = stt.text_queue.get(timeout=0.1)
                if text:
                    self.sendPacket(text.encode('utf-8'))
                    print(f"[i] Sent text: {text}")
            except queue.Empty:
                continue

    def parsePacket(self, data):
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

    def sendPacket(self, frame):
        compressed = frame
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

            packet = bytearray()
            packet = header + payload   
            try:
                self.send_socket.sendto(packet, (IP_ADDRESS, START_PORT))
            except socket.error as e:
                print(f"[w] Packet send failed on segment {i}. Socket error: {e}")
                
    def recvLoop(self):
        while self.is_recv_running.is_set():
            packet = bytearray()
            fragments = []
            marker = None
            ssrc = 0
            i = 0

            while self.is_recv_running.is_set():
                try:
                    packet, _ = self.recv_socket.recvfrom(BUFFER_SIZE)
                except socket.timeout:
                    continue 
                except socket.error as e:
                    print(f"[e] Packet recv failed. Socket error: {e}")
                    return
                                
                if len(packet) < 10: 
                    print("[w] Empty packet received")   
                    packet = None
                    break

                header, marker, payload = self.parsePacket(packet)
                index = header['fragment_index']

                if marker == -1:
                    packet = 0
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

                fragments[index] = payload

                if i == num_fragments - 1:
                    packet = bytearray()
                    for frag in fragments:
                        packet.extend(frag)
                    break

                i += 1

            if packet:
                stt.audio_queue.put(packet)
            elif packet == 0:
                print("[i] GUI disconnected")
                is_active.clear()
                break
            else:
                print("[w] Recv error. No payload")
                continue

# --- Main ---
print("[i] Hi")
stt = VoskModel()
channel = RTPStreamHandler()
is_active = threading.Event()
is_active.set()
print("[i] Setup done")

try:
    while is_active.is_set():
        time.sleep(1)
    else:
        print("[i] Closing program...")
        channel.destroy()
        stt.destroy()
        print("[i] Bye")
except KeyboardInterrupt:
    print("[i] Closing program...")
    channel.destroy()
    stt.destroy()
    print("[i] Bye")
