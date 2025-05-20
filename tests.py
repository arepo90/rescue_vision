import cv2
import numpy as np

# Load the image
image = cv2.imread('qr2.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding (adjust the threshold value as needed)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Use QR code detector
detector = cv2.QRCodeDetector()
retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(thresh)

if retval:
    print(f"Decoded QR Code: {decoded_info}")
else:
    print("QR Code not detected.")
