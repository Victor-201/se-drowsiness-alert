import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
from collections import deque
from math import atan2, degrees


class HeadPoseBlinkDetector:
    def __init__(self, config, alert_system):
        self.config = config
        self.alert_system = alert_system
        self.landmark_predictor = None

        # Cấu hình cho phát hiện nghiêng đầu
        self.HEAD_TILT_THRESHOLD = 15  # Ngưỡng độ nghiêng đầu (độ)
        self.HEAD_TILT_FRAMES = 20  # Số frame liên tiếp để cảnh báo nghiêng đầu
        self.head_tilt_counter = 0

        # Cấu hình cho phát hiện nháy mắt nhiều
        self.BLINK_THRESHOLD = 0.25  # Ngưỡng phát hiện nháy mắt (tỷ lệ khía cạnh mắt)
        self.BLINK_CONSEC_FRAMES = 3  # Số frame liên tiếp để tính là một lần nháy mắt
        self.BLINK_FREQUENCY_THRESHOLD = 5  # Số lần nháy mắt trong khoảng thời gian BLINK_TIME_WINDOW
        self.BLINK_TIME_WINDOW = 3.0  # Khoảng thời gian tính tần suất nháy mắt (giây)

        self.blink_counter = 0
        self.blink_total = 0
        self.blink_times = deque(maxlen=20)
        self.last_blink_time = time.time()
        self.eye_closed = False

        # Trạng thái cảnh báo
        self.head_tilt_alert = False
        self.rapid_blink_alert = False

        # Lịch sử lưu EAR để phát hiện nháy mắt chính xác hơn
        self.ear_history = deque(maxlen=10)

    def set_landmark_predictor(self, predictor):
        self.landmark_predictor = predictor

    @staticmethod
    def calculate_head_pose(shape_np):
        """Tính góc nghiêng đầu dựa trên các điểm landmarks"""
        # Lấy tọa độ các điểm mốc cho việc tính góc nghiêng
        left_eye_center = np.mean(shape_np[36:42], axis=0)
        right_eye_center = np.mean(shape_np[42:48], axis=0)
        nose_tip = shape_np[30]
        left_mouth = shape_np[48]
        right_mouth = shape_np[54]

        # Tính góc nghiêng dựa trên vector giữa hai mắt
        eye_vector = right_eye_center - left_eye_center
        roll_angle = degrees(atan2(eye_vector[1], eye_vector[0]))

        # Tính góc nghiêng dựa trên mũi và miệng
        mouth_center = (left_mouth + right_mouth) / 2
        nose_to_mouth = mouth_center - nose_tip
        pitch_angle = degrees(atan2(nose_to_mouth[1], nose_to_mouth[0])) - 90

        return abs(roll_angle), abs(pitch_angle)

    def detect_blink(self, ear):
        """Phát hiện nháy mắt và tính tần suất nháy mắt"""
        self.ear_history.append(ear)

        # Đảm bảo có đủ dữ liệu để tính toán
        if len(self.ear_history) < 3:
            return False

        # Phát hiện nháy mắt dựa trên giá trị EAR
        if not self.eye_closed and ear < self.BLINK_THRESHOLD:
            self.blink_counter += 1
            self.eye_closed = True
        elif self.eye_closed and ear > self.BLINK_THRESHOLD:
            self.eye_closed = False

            # Nếu số frame mắt nhắm vượt ngưỡng, đó là một lần nháy mắt
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                current_time = time.time()
                self.blink_total += 1
                self.blink_times.append(current_time)
                self.last_blink_time = current_time
                self.blink_counter = 0
                return True

            self.blink_counter = 0

        return False

    def rapid_blink_detected(self):
        """Kiểm tra xem có đang nháy mắt quá nhanh không"""
        if len(self.blink_times) < 2:
            return False

        current_time = time.time()
        # Đếm số lần nháy mắt trong khoảng thời gian BLINK_TIME_WINDOW
        blinks_in_window = sum(1 for t in self.blink_times
                               if current_time - t <= self.BLINK_TIME_WINDOW)

        return blinks_in_window >= self.BLINK_FREQUENCY_THRESHOLD

    def analyze_frame(self, frame, shape_np):
        """Phân tích frame để phát hiện nghiêng đầu và nháy mắt nhiều"""
        if shape_np is None or len(shape_np) != 68:
            return frame, False, False

        # Tính góc nghiêng đầu
        roll_angle, pitch_angle = self.calculate_head_pose(shape_np)
        head_tilted = roll_angle > self.HEAD_TILT_THRESHOLD or pitch_angle > self.HEAD_TILT_THRESHOLD

        # Tính tỷ lệ khía cạnh mắt (EAR)
        left_eye, right_eye = shape_np[36:42], shape_np[42:48]
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Phát hiện nháy mắt
        blink_detected = self.detect_blink(ear)
        rapid_blinking = self.rapid_blink_detected()

        # Xử lý cảnh báo nghiêng đầu
        if head_tilted:
            self.head_tilt_counter += 1
            if self.head_tilt_counter >= self.HEAD_TILT_FRAMES and not self.head_tilt_alert:
                self.head_tilt_alert = True
        else:
            self.head_tilt_counter = max(0, self.head_tilt_counter - 1)
            if self.head_tilt_counter == 0:
                self.head_tilt_alert = False

        # Xử lý cảnh báo nháy mắt nhanh
        self.rapid_blink_alert = rapid_blinking

        # Vẽ thông tin lên frame
        self.draw_info(frame, ear, roll_angle, pitch_angle, blink_detected)

        return frame, self.head_tilt_alert, self.rapid_blink_alert

    @staticmethod
    def calculate_ear(eye_points):
        """Tính tỷ lệ khía cạnh mắt (EAR)"""
        points = np.array(eye_points)
        a = dist.euclidean(points[1], points[5])
        b = dist.euclidean(points[2], points[4])
        c = dist.euclidean(points[0], points[3])
        ear = (a + b) / (2.0 * c) if c > 0 else 0.0
        return ear

    def draw_info(self, frame, ear, roll_angle, pitch_angle, blink_detected):
        """Vẽ thông tin lên frame"""
        # Hiển thị thông số chính
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Head Roll: {roll_angle:.1f} deg", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Head Pitch: {pitch_angle:.1f} deg", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Hiển thị thông tin nháy mắt
        blink_color = (0, 0, 255) if self.rapid_blink_alert else (255, 255, 255)
        cv2.putText(frame, f"Blinks: {self.blink_total}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 1)

        # Hiển thị cảnh báo khi nghiêng đầu
        if self.head_tilt_alert:
            cv2.putText(frame, "HEAD TILT ALERT!", (frame.shape[1] // 4, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return frame

    def render_head_tilt_alert(self, frame):
        """Hiển thị cảnh báo nghiêng đầu"""
        if not self.head_tilt_alert:
            return frame

        self.alert_system.play_alert_sound()
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 6)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 165, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        alert_text = "HEAD POSITION ALERT!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, alert_text, (text_x, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        return frame

    def render_rapid_blink_alert(self, frame):
        """Hiển thị cảnh báo nháy mắt nhanh"""
        if not self.rapid_blink_alert:
            return frame

        self.alert_system.play_alert_sound()
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 8)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 165, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        alert_text = "RAPID BLINKING ALERT!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, alert_text, (text_x, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        info_text = f"Blinks: {len(self.blink_times)} in {self.BLINK_TIME_WINDOW}s"
        cv2.putText(frame, info_text, (text_x + 50, frame.shape[0] // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def reset(self):
        """Reset trạng thái của detector"""
        self.head_tilt_counter = 0
        self.head_tilt_alert = False
        self.blink_counter = 0
        self.blink_total = 0
        self.blink_times.clear()
        self.rapid_blink_alert = False
        self.eye_closed = False
        self.ear_history.clear()