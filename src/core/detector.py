import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import os
import logging
from ..configs.config import Config
from .model_manager import ModelManager
from .facial_analyzer import FacialAnalyzer
from .alert_system import AlertSystem

class DrowsinessDetector:
    def __init__(self):
        self.config = Config()
        self.model_manager = ModelManager(self.config)
        self.analyzer = FacialAnalyzer(self.config)
        self.alert_system = AlertSystem(self.config)
        
        self.camera = None
        self.EYE_AR_THRESH = self.config.EAR_THRESHOLD
        self.EYE_AR_CONSEC_FRAMES = self.config.EAR_CONSEC_FRAMES
        self.counter = 0
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self.model_manager.download_model())

    def start_camera(self):
        # Khởi tạo camera
        print("Khởi tạo camera...")
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise IOError("Không thể mở camera!")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        print("Camera khởi tạo thành công")

    def stop_camera(self):
        # Dừng camera và giải phóng tài nguyên
        if self.camera:
            self.camera.release()
            self.analyzer.reset_display()
            self.camera = None

    def process_frame(self):
        if not self.camera:
            logging.error("Camera chưa khởi tạo")
            return None, False
        
        ret, frame = self.camera.read()
        if not ret or frame is None:
            logging.error("Không thể lấy frame")
            return None, False

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if not faces:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.analyzer.show_camera_feed(frame)
            return frame, False

        drowsiness_detected = False
        for face in faces:
            shape = self.landmark_predictor(gray, face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.draw_facial_ratios(frame, shape_np)
            
            left_eye, right_eye = shape_np[36:42], shape_np[42:48]
            ear = (self.analyzer.calculate_ear(left_eye) + self.analyzer.calculate_ear(right_eye)) / 2.0
            
            if ear < self.EYE_AR_THRESH:
                self.counter += 1
                if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                    drowsiness_detected = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.counter = 0

        self.analyzer.show_camera_feed(frame)
        return frame, drowsiness_detected

    def draw_facial_ratios(self, frame, shape_np):
        # Vẽ các đặc điểm khuôn mặt
        for i in range(68):
            x, y = shape_np[i]
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        cv2.polylines(frame, [shape_np[36:42]], True, (0, 255, 0), 1)  # Mắt trái
        cv2.polylines(frame, [shape_np[42:48]], True, (0, 255, 0), 1)  # Mắt phải
        cv2.polylines(frame, [shape_np[48:60]], True, (0, 255, 0), 1)  # Miệng
        cv2.polylines(frame, [shape_np[27:36]], True, (0, 255, 0), 1)  # Mũi
        
        face_width = np.linalg.norm(shape_np[0] - shape_np[16])
        face_height = np.linalg.norm(shape_np[27] - shape_np[8])
        ratio = face_width / face_height if face_height > 0 else 0
        cv2.putText(frame, f"Ratio: {ratio:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def __del__(self):
        self.stop_camera()