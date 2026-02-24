"""
Simplified QR Decoder (Deprecated - use qrdetect.QRSizeDetector.detect_and_decode_qr instead)

This module is kept for backward compatibility only.
The new pyzbar-based detector in qrdetect.py combines detection and decoding
in one pass, which is much faster for Raspberry Pi.

For new code, use:
    detector = QRSizeDetector()
    bbox, area, width, threshold_met, decoded_data = detector.detect_and_decode_qr(frame)
"""

import cv2
import time
from config import (
    DECODE_TIMEOUT_MS,
    MAX_DECODE_ATTEMPTS
)

class QRReader:
    """
    Legacy QR decoder using OpenCV.
    NOTE: This is slower than the new pyzbar-based approach in qrdetect.py
    """
    def __init__(self):
        self.detector = cv2.QRCodeDetector()
        self.decode_count = 0
        self.failed_decode_count = 0
        self.last_decoded_id = None
        print("[QRReader] DEPRECATED: Consider using QRSizeDetector.detect_and_decode_qr() for better performance")
        
    def decode_qr(self, frame, bbox):
        """
        Decode QR code from ROI defined by bbox.
        Returns decoded string or None.
        
        WARNING: This method is slower than pyzbar. Use QRSizeDetector.detect_and_decode_qr() instead.
        """
        start_time = time.time()
        
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return None
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        
        material_id = None
        attempts = 0
        
        # Simplified: Just try grayscale conversion (fastest approach for RPi)
        while attempts < min(MAX_DECODE_ATTEMPTS, 2):  # Limit to 2 attempts for RPi
            attempts += 1
            
            if attempts == 1:
                # Try on original ROI
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                data, _, _ = self.detector.detectAndDecode(gray)
            else:
                # Try with adaptive thresholding
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                data, _, _ = self.detector.detectAndDecode(gray)
            
            if data:
                material_id = data
                break
            
            elapsed = (time.time() - start_time) * 1000
            if elapsed > DECODE_TIMEOUT_MS:
                break
        
        if material_id:
            self.decode_count += 1
            self.last_decoded_id = material_id
        else:
            self.failed_decode_count += 1
        
        return material_id
    
    def get_decode_success_rate(self):
        total = self.decode_count + self.failed_decode_count
        if total == 0:
            return 0.0
        return (self.decode_count / total) * 100