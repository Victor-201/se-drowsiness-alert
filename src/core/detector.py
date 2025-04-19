import cv2
import numpy as np
import time
import logging
from collections import deque
from src.configs.config import Config
from src.core.model_manager import ModelManager
from src.core.facial_analyzer import FacialAnalyzer
from src.core.alert_system import AlertSystem

logger = logging.getLogger(__name__)

class DrowsinessDetector:
    def __init__(self):
        # Khởi tạo bộ phát hiện buồn ngủ
        self.config = Config()
        self.model_manager = ModelManager()
        self.analyzer = FacialAnalyzer()
        self.alert_system = AlertSystem()
        self.camera = None
        self.EYE_AR_THRESH = self.config.EAR_THRESHOLD
        self.EYE_AR_CONSEC_FRAMES = self.config.EAR_CONSEC_FRAMES
        self.BLINK_CONSEC_FRAMES = self.config.BLINK_CONSEC_FRAMES
        self.NO_FACE_CONSEC_FRAMES = self.config.NO_FACE_ALERT_FRAMES
        self.eye_counter = 0
        self.no_face_counter = 0
        self.face_detected = False
        self.drowsiness_start_time = None
        self.face_detector = self.model_manager.detector
        self.landmark_predictor = self.model_manager.predictor
        self.head_tilt_threshold = self.config.HEAD_TILT_THRESHOLD
        self.head_tilt_frames = self.config.HEAD_TILT_FRAMES
        self.head_tilt_counter = 0
        self.BLINK_THRESHOLD = 0.25
        self.BLINK_FREQUENCY_THRESHOLD = 5
        self.BLINK_TIME_WINDOW = 3.0
        self.blink_counter = 0
        self.blink_total = 0
        self.blink_times = deque(maxlen=20)
        self.last_blink_time = time.time()
        self.eye_closed = False
        self.rapid_blink_alert = False
        self.ear_history = deque(maxlen=30)

    def start_camera(self):
        # Khởi tạo camera
        logger.info("Khởi tạo camera...")
        try:
            self.camera = cv2.VideoCapture(self.config.CAMERA_ID)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                raise IOError("Không thể mở camera")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
            logger.info("Khởi tạo camera thành công")
        except Exception as e:
            logger.error(f"Khởi tạo camera thất bại: {e}")
            raise

    def stop_camera(self):
        # Dừng và giải phóng camera
        if self.camera:
            self.camera.release()
            self.analyzer.reset_display()
            self.camera = None
            logger.info("Dừng camera")

    def detect_blink(self, ear):
        # Phát hiện nháy mắt
        self.ear_history.append(ear)
        if len(self.ear_history) < 3:
            return False

        if not self.eye_closed and ear < self.BLINK_THRESHOLD:
            self.blink_counter += 1
            self.eye_closed = True
        elif self.eye_closed and ear > self.BLINK_THRESHOLD:
            self.eye_closed = False
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                current_time = time.time()
                self.blink_total += 1
                self.blink_times.append(current_time)
                self.last_blink_time = current_time
                self.blink_counter = 0
                return True
            self.blink_counter = 0
        return False

    def check_rapid_blinking(self):
        # Kiểm tra nháy mắt nhanh
        if len(self.blink_times) < 2:
            return False
        current_time = time.time()
        blinks_in_window = sum(1 for t in self.blink_times
                              if current_time - t <= self.BLINK_TIME_WINDOW)
        return blinks_in_window >= self.BLINK_FREQUENCY_THRESHOLD

    @staticmethod
    def find_largest_face(faces):
        # Tìm khuôn mặt lớn nhất
        if not faces:
            return None
        largest_face, largest_area = None, 0
        for face in faces:
            width = face.right() - face.left()
            height = face.bottom() - face.top()
            area = width * height
            if area > largest_area:
                largest_area = area
                largest_face = face
        return largest_face

    def process_frame(self):
        # Xử lý từng khung hình từ camera
        if not self.camera:
            logger.error("Camera chưa khởi tạo")
            return None, False

        ret, frame = self.camera.read()
        if not ret or frame is None:
            logger.error("Không thể lấy khung hình")
            return None, False

        frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)

        if not faces:
            self.no_face_counter += 1
            self.face_detected = False
            if self.no_face_counter >= self.NO_FACE_CONSEC_FRAMES:
                frame = self.alert_system.render_distraction_alert(frame)
                self.analyzer.show_camera_feed(frame)
                return frame, True
            cv2.putText(frame, "Không phát hiện khuôn mặt", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.ALERT_COLOR, 2)
            self.analyzer.show_camera_feed(frame)
            return frame, False

        self.no_face_counter = 0
        self.face_detected = True
        drowsiness_detected = False
        head_tilt_detected = False
        rapid_blink_detected = False

        largest_face = self.find_largest_face(faces)
        if largest_face:
            shape = self.landmark_predictor(gray, largest_face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            x1, y1, x2, y2 = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.PRIMARY_COLOR, 2)  # Vẽ khung khuôn mặt
            self.draw_facial_ratios(frame, shape_np)

            left_eye, right_eye = shape_np[36:42], shape_np[42:48]
            left_ear = self.analyzer.calculate_ear(left_eye)
            right_ear = self.analyzer.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            blink_detected = self.detect_blink(ear)
            rapid_blink_detected = self.check_rapid_blinking()

            roll_angle, pitch_angle = self.analyzer.calculate_head_pose(shape_np)
            head_tilted = roll_angle > self.head_tilt_threshold or pitch_angle > self.head_tilt_threshold

            if head_tilted:
                self.head_tilt_counter += 1
                if self.head_tilt_counter >= self.head_tilt_frames:
                    head_tilt_detected = True
            else:
                self.head_tilt_counter = max(0, self.head_tilt_counter - 1)

            # Hiển thị thông tin trạng thái
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)
            cv2.putText(frame, f"Góc nghiêng: {roll_angle:.1f} độ", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.config.TEXT_COLOR, 1)
            cv2.putText(frame, f"Góc cúi: {pitch_angle:.1f} độ", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.config.TEXT_COLOR, 1)
            blink_color = self.config.ALERT_COLOR if rapid_blink_detected else self.config.TEXT_COLOR
            cv2.putText(frame, f"Nháy mắt: {self.blink_total}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 1)

            if ear < self.EYE_AR_THRESH:
                self.eye_counter += 1
                if self.eye_counter == 1:
                    self.drowsiness_start_time = time.time()
                if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                    drowsiness_detected = True
                    drowsiness_duration = time.time() - self.drowsiness_start_time
                    frame = self.alert_system.render_drowsiness_alert(frame, drowsiness_duration)
            else:
                self.eye_counter = 0
                self.drowsiness_start_time = None

            frame = self.alert_system.render_status_bar(frame, ear, self.EYE_AR_THRESH)
            metrics = [
                f"EAR: {ear:.2f} (Ngưỡng: {self.EYE_AR_THRESH:.2f})",
                f"Nháy mắt: {self.blink_total}"
            ]
            status_text = "Trạng thái: Bình thường"
            if drowsiness_detected:
                status_text = "Trạng thái: BUỒN NGỦ"
            elif head_tilt_detected:
                status_text = "Trạng thái: NGHIÊNG ĐẦU"
            elif rapid_blink_detected:
                status_text = "Trạng thái: NHÁY MẮT NHANH"
            frame = self.alert_system.render_metrics(frame, metrics, status_text)

            if head_tilt_detected:
                frame = self.alert_system.render_head_tilt_alert(frame)
            if rapid_blink_detected:
                frame = self.alert_system.render_fatigue_alert(frame)

        cv2.putText(frame, f"Khuôn mặt: {len(faces)} (Xử lý lớn nhất)",
                    (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)
        self.analyzer.show_camera_feed(frame)
        return frame, drowsiness_detected or head_tilt_detected or rapid_blink_detected

    def draw_facial_ratios(self, frame, shape_np):
        # Vẽ các đặc điểm khuôn mặt
        for i in range(68):
            cv2.circle(frame, tuple(shape_np[i]), 1, self.config.PRIMARY_COLOR, -1)
        cv2.polylines(frame, [shape_np[36:42]], True, self.config.PRIMARY_COLOR, 1)  # Mắt trái
        cv2.polylines(frame, [shape_np[42:48]], True, self.config.PRIMARY_COLOR, 1)  # Mắt phải
        cv2.polylines(frame, [shape_np[48:60]], True, self.config.PRIMARY_COLOR, 1)  # Miệng
        cv2.polylines(frame, [shape_np[27:36]], True, self.config.PRIMARY_COLOR, 1)  # Mũi
        face_width = np.linalg.norm(shape_np[0] - shape_np[16])
        face_height = np.linalg.norm(shape_np[27] - shape_np[8])
        ratio = face_width / face_height if face_height > 0 else 0
        cv2.putText(frame, f"Tỷ lệ khuôn mặt: {ratio:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.TEXT_COLOR, 1)
        return frame

    def calibrate(self, duration=5):
        # Hiệu chỉnh ngưỡng EAR
        if not self.camera:
            self.start_camera()
        logger.info(f"Bắt đầu hiệu chỉnh trong {duration} giây")
        ear_values = []
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = self.camera.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            largest_face = self.find_largest_face(faces)
            if largest_face:
                shape = self.landmark_predictor(gray, largest_face)
                shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                left_eye, right_eye = shape_np[36:42], shape_np[42:48]
                left_ear = self.analyzer.calculate_ear(left_eye)
                right_ear = self.analyzer.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                ear_values.append(ear)
                cv2.putText(frame, f"Đang hiệu chỉnh... {int(duration - (time.time() - start_time))}s",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR hiện tại: {ear:.2f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.draw_facial_ratios(frame, shape_np)
            cv2.imshow("Hiệu chỉnh", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow("Hiệu chỉnh")
        if ear_values:
            avg_ear = np.mean(ear_values)
            new_threshold = avg_ear * 0.9
            logger.info(f"Hiệu chỉnh hoàn tất. Ngưỡng EAR mới: {new_threshold:.3f}")
            self.EYE_AR_THRESH = new_threshold
            self.config.save_calibration(new_threshold)
            return True
        logger.error("Hiệu chỉnh thất bại: Không phát hiện khuôn mặt")
        return False

    def __del__(self):
        # Dọn dẹp tài nguyên
        self.stop_camera()