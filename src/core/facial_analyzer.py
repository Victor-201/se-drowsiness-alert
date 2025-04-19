import numpy as np
from scipy.spatial import distance as dist
import cv2
import logging
from math import atan2, degrees
from src.configs.config import Config

logger = logging.getLogger(__name__)

class FacialAnalyzer:
    def __init__(self):
        # Khởi tạo bộ phân tích khuôn mặt
        self.config = Config()
        self.min_ear = 0.15
        self.max_ear = 0.40

    def calculate_ear(self, eye_points):
        # Tính tỷ lệ khía cạnh mắt (EAR)
        points = np.array(eye_points, dtype=np.float32)
        A = dist.euclidean(points[1], points[5])
        B = dist.euclidean(points[2], points[4])
        C = dist.euclidean(points[0], points[3])
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return np.clip(ear, self.min_ear, self.max_ear)

    def calculate_mar(self, mouth_points):
        # Tính tỷ lệ khía cạnh miệng (MAR)
        A = dist.euclidean(mouth_points[13], mouth_points[19])
        B = dist.euclidean(mouth_points[14], mouth_points[18])
        C = dist.euclidean(mouth_points[15], mouth_points[17])
        D = dist.euclidean(mouth_points[12], mouth_points[16])
        return (A + B + C) / (3.0 * D) if D > 0 else 0.0

    def eye_aspect_ratio_variance(self, ear_history, window_size=30):
        # Tính độ biến thiên EAR
        if len(ear_history) < window_size:
            return 0.0
        return np.var(ear_history[-window_size:])

    def enhance_eye_region(self, frame, eye_points):
        # Cải thiện vùng mắt để phát hiện chính xác
        eye_region = np.array(eye_points, dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [eye_region], (225, 225, 225))
        eye_frame = cv2.bitwise_and(frame, frame, mask=mask)
        eye_frame = cv2.equalizeHist(cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY))
        eye_frame = cv2.GaussianBlur(eye_frame, (3, 3), 0)
        return cv2.cvtColor(eye_frame, cv2.COLOR_GRAY2BGR)

    def calculate_head_pose(self, shape_np):
        # Tính góc nghiêng và cúi đầu
        left_eye_center = np.mean(shape_np[36:42], axis=0)
        right_eye_center = np.mean(shape_np[42:48], axis=0)
        nose_tip = shape_np[30]
        left_mouth, right_mouth = shape_np[48], shape_np[54]
        eye_vector = right_eye_center - left_eye_center
        roll_angle = degrees(atan2(eye_vector[1], eye_vector[0]))
        mouth_center = (left_mouth + right_mouth) / 2
        nose_to_mouth = mouth_center - nose_tip
        pitch_angle = degrees(atan2(nose_to_mouth[1], nose_to_mouth[0])) - 90
        return abs(roll_angle), abs(pitch_angle)

    def show_camera_feed(self, frame):
        # Hiển thị khung hình từ camera
        if frame is None:
            logger.warning("Khung hình rỗng")
            return
        cv2.imshow("Phát hiện buồn ngủ", frame)
        return cv2.waitKey(1) & 0xFF

    def reset_display(self):
        # Đóng tất cả cửa sổ hiển thị
        cv2.destroyAllWindows()
        logger.info("Reset hiển thị")