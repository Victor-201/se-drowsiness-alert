import numpy as np
from scipy.spatial import distance as dist
import cv2
import logging
from ..configs.config import Config

class FacialAnalyzer:
    def __init__(self):
        self.config = Config()
        self.min_ear = 0.15
        self.max_ear = 0.40

    def process_frame(self, frame):
        if frame is None:
            logging.warning("Frame rỗng")
            return None
        return frame

    def calculate_ear(self, eye_points):
        # Tính tỷ lệ khía cạnh mắt (EAR)
        points = np.array(eye_points)
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
        # Cải thiện vùng mắt để phát hiện chính xác hơn
        eye_region = np.array(eye_points, dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [eye_region], 255)
        eye_frame = cv2.bitwise_and(frame, frame, mask=mask)
        eye_frame = cv2.equalizeHist(cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY))
        eye_frame = cv2.GaussianBlur(eye_frame, (3, 3), 0)
        return cv2.cvtColor(eye_frame, cv2.COLOR_GRAY2BGR)

    def show_camera_feed(self, frame):
        # Hiển thị luồng video
        if frame is None:
            logging.warning("Frame rỗng")
            return None
        return cv2.waitKey(1) & 0xFF

    def reset_display(self):
        # Reset cửa sổ hiển thị
        cv2.destroyAllWindows()
        logging.info("Reset hiển thị thành công")