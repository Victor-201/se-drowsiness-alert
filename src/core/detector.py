import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import logging
from collections import deque
from math import atan2, degrees
from src.configs.config import Config
from src.core.model_manager import ModelManager
from src.core.facial_analyzer import FacialAnalyzer
from src.core.alert_system import AlertSystem


class DrowsinessDetector:
    def __init__(self):
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

        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self.model_manager.download_model())

        # Các cấu hình và biến cho phát hiện nghiêng đầu
        self.HEAD_TILT_THRESHOLD = 15  # Ngưỡng độ nghiêng đầu (độ)
        self.HEAD_TILT_FRAMES = 20  # Số frame liên tiếp để cảnh báo nghiêng đầu
        self.head_tilt_counter = 0
        self.head_tilt_alert = False

        # Các cấu hình và biến cho phát hiện nháy mắt nhiều
        self.BLINK_THRESHOLD = 0.25  # Ngưỡng phát hiện nháy mắt (tỷ lệ khía cạnh mắt)
        self.BLINK_FREQUENCY_THRESHOLD = 5  # Số lần nháy mắt trong khoảng thời gian BLINK_TIME_WINDOW
        self.BLINK_TIME_WINDOW = 3.0  # Khoảng thời gian tính tần suất nháy mắt (giây)

        self.blink_counter = 0
        self.blink_total = 0
        self.blink_times = deque(maxlen=20)
        self.last_blink_time = time.time()
        self.eye_closed = False
        self.rapid_blink_alert = False

        # Lịch sử lưu EAR để phát hiện nháy mắt chính xác hơn
        self.ear_history = deque(maxlen=30)

    def start_camera(self):
        # Khởi tạo camera
        print("Khởi tạo camera...")
        self.camera = cv2.VideoCapture(self.config.CAMERA_ID)
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

    @staticmethod
    def calculate_ear(eye_points):
        """Tính tỷ lệ khía cạnh mắt (EAR)"""
        points = np.array(eye_points)
        a = distance.euclidean(points[1], points[5])
        b = distance.euclidean(points[2], points[4])
        c = distance.euclidean(points[0], points[3])
        ear = (a + b) / (2.0 * c) if c > 0 else 0.0
        return ear

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

    def check_rapid_blinking(self):
        """Kiểm tra xem có đang nháy mắt quá nhanh không"""
        if len(self.blink_times) < 2:
            return False

        current_time = time.time()
        # Đếm số lần nháy mắt trong khoảng thời gian BLINK_TIME_WINDOW
        blinks_in_window = sum(1 for t in self.blink_times
                               if current_time - t <= self.BLINK_TIME_WINDOW)

        return blinks_in_window >= self.BLINK_FREQUENCY_THRESHOLD

    @staticmethod
    def find_largest_face(faces):
        """Tìm khuôn mặt lớn nhất trong danh sách faces được phát hiện"""
        if not faces:
            return None

        # Tìm khuôn mặt có diện tích lớn nhất
        largest_face = None
        largest_area = 0

        for face in faces:
            # Tính diện tích khuôn mặt
            width = face.right() - face.left()
            height = face.bottom() - face.top()
            area = width * height

            if area > largest_area:
                largest_area = area
                largest_face = face

        return largest_face

    def process_frame(self):
        if not self.camera:
            logging.error("Camera chưa khởi tạo")
            return None, False

        ret, frame = self.camera.read()
        if not ret or frame is None:
            logging.error("Không thể lấy frame")
            return None, False

        frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)

        # Kiểm tra có khuôn mặt trong frame không
        if not faces:
            self.no_face_counter += 1
            self.face_detected = False

            # Hiện cảnh báo nếu không có khuôn mặt quá lâu
            if self.no_face_counter >= self.NO_FACE_CONSEC_FRAMES:
                frame = self.alert_system.render_distraction_alert(frame)
                self.analyzer.show_camera_feed(frame)
                return frame, True  # Coi như đã phát hiện tình trạng nguy hiểm

            cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.ALERT_COLOR, 2)
            self.analyzer.show_camera_feed(frame)
            return frame, False

        # Reset bộ đếm khi phát hiện được khuôn mặt
        self.no_face_counter = 0
        self.face_detected = True

        drowsiness_detected = False
        head_tilt_detected = False
        rapid_blink_detected = False

        # Tìm khuôn mặt lớn nhất trong khung hình
        largest_face = self.find_largest_face(faces)

        if largest_face:
            # Lấy facial landmarks cho khuôn mặt lớn nhất
            shape = self.landmark_predictor(gray, largest_face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])

            # Vẽ khung khuôn mặt
            x1, y1, x2, y2 = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.PRIMARY_COLOR, 2)

            # Vẽ facial landmarks
            self.draw_facial_ratios(frame, shape_np)

            # Tính EAR và phát hiện nháy mắt
            left_eye, right_eye = shape_np[36:42], shape_np[42:48]
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Phát hiện nháy mắt và tần suất nháy mắt
            blink_detected = self.detect_blink(ear)
            rapid_blink_detected = self.check_rapid_blinking()

            # Tính góc nghiêng đầu
            roll_angle, pitch_angle = self.calculate_head_pose(shape_np)
            head_tilted = roll_angle > self.HEAD_TILT_THRESHOLD or pitch_angle > self.HEAD_TILT_THRESHOLD

            # Xử lý cảnh báo nghiêng đầu
            if head_tilted:
                self.head_tilt_counter += 1
                if self.head_tilt_counter >= self.HEAD_TILT_FRAMES:
                    head_tilt_detected = True
            else:
                self.head_tilt_counter = max(0, self.head_tilt_counter - 1)
                head_tilt_detected = False

            # Hiển thị thông tin về phát hiện nghiêng đầu và nháy mắt
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)
            cv2.putText(frame, f"Head Roll: {roll_angle:.1f} deg", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.config.TEXT_COLOR, 1)
            cv2.putText(frame, f"Head Pitch: {pitch_angle:.1f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.config.TEXT_COLOR, 1)

            blink_color = self.config.ALERT_COLOR if rapid_blink_detected else self.config.TEXT_COLOR
            cv2.putText(frame, f"Blinks: {self.blink_total}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 1)

            # Phát hiện buồn ngủ
            if ear < self.EYE_AR_THRESH:
                self.eye_counter += 1
                # Bắt đầu đếm thời gian buồn ngủ
                if self.eye_counter == 1:
                    self.drowsiness_start_time = time.time()

                # Nếu đóng mắt đủ lâu, cảnh báo buồn ngủ
                if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                    drowsiness_detected = True
                    drowsiness_duration = time.time() - self.drowsiness_start_time
                    # Hiển thị cảnh báo buồn ngủ
                    frame = self.alert_system.render_drowsiness_alert(frame, drowsiness_duration)
            else:
                self.eye_counter = 0
                self.drowsiness_start_time = None

            # Cập nhật thanh trạng thái EAR
            frame = self.alert_system.render_status_bar(frame, ear, self.EYE_AR_THRESH)

            # Hiển thị thông tin trạng thái
            metrics = [
                f"EAR: {ear:.2f} (Threshold: {self.EYE_AR_THRESH:.2f})",
                f"Blinks: {self.blink_total}"
            ]
            status_text = "Status: Normal"
            if drowsiness_detected:
                status_text = "Status: DROWSY"
            elif head_tilt_detected:
                status_text = "Status: HEAD TILTED"
            elif rapid_blink_detected:
                status_text = "Status: RAPID BLINKING"

            frame = self.alert_system.render_metrics(frame, metrics, status_text)

            # Hiển thị cảnh báo nghiêng đầu
            if head_tilt_detected:
                self.render_head_tilt_alert(frame)

            # Hiển thị cảnh báo nháy mắt nhanh
            if rapid_blink_detected:
                self.render_rapid_blink_alert(frame)

        # Hiển thị số lượng khuôn mặt phát hiện được
        cv2.putText(frame, f"Faces: {len(faces)} (Processing largest)",
                    (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)

        # Hiển thị frame
        self.analyzer.show_camera_feed(frame)

        # Trả về trạng thái nguy hiểm nếu phát hiện bất kỳ tình trạng nào
        return frame, drowsiness_detected or head_tilt_detected or rapid_blink_detected

    def render_head_tilt_alert(self, frame):
        """Hiển thị cảnh báo nghiêng đầu"""
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

    def draw_facial_ratios(self, frame, shape_np):
        # Vẽ các đặc điểm khuôn mặt
        for i in range(68):
            x, y = shape_np[i]
            cv2.circle(frame, (x, y), 1, self.config.PRIMARY_COLOR, -1)

        # Vẽ contour cho các phần chính của khuôn mặt
        cv2.polylines(frame, [np.array(shape_np[36:42], dtype=np.int32)], True, self.config.PRIMARY_COLOR,
                      1)  # Mắt trái
        cv2.polylines(frame, [np.array(shape_np[42:48], dtype=np.int32)], True, self.config.PRIMARY_COLOR,
                      1)  # Mắt phải
        cv2.polylines(frame, [np.array(shape_np[48:60], dtype=np.int32)], True, self.config.PRIMARY_COLOR, 1)  # Miệng
        cv2.polylines(frame, [np.array(shape_np[27:36], dtype=np.int32)], True, self.config.PRIMARY_COLOR, 1)  # Mũi

        # Tính và hiển thị tỷ lệ khuôn mặt
        face_width = np.linalg.norm(shape_np[0] - shape_np[16])
        face_height = np.linalg.norm(shape_np[27] - shape_np[8])
        ratio = face_width / face_height if face_height > 0 else 0
        cv2.putText(frame, f"Face Ratio: {ratio:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.TEXT_COLOR,
                    1)
        return frame

    def calibrate(self, duration=5):
        """Hiệu chỉnh ngưỡng EAR tự động dựa trên người sử dụng"""
        if not self.camera:
            self.start_camera()

        print(f"Bắt đầu hiệu chỉnh trong {duration} giây...")
        print("Hãy nhìn thẳng vào camera và giữ mắt mở bình thường")

        ear_values = []
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = self.camera.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)

            # Chỉ xử lý khuôn mặt lớn nhất
            largest_face = self.find_largest_face(faces)

            if largest_face:
                shape = self.landmark_predictor(gray, largest_face)
                shape_np = np.array([[p.x, p.y] for p in shape.parts()])

                left_eye, right_eye = shape_np[36:42], shape_np[42:48]
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                ear_values.append(ear)

                cv2.putText(frame, f"Calibrating... {int(duration - (time.time() - start_time))}s",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Current EAR: {ear:.2f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Vẽ facial landmarks
                self.draw_facial_ratios(frame, shape_np)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

        cv2.destroyWindow("Calibration")

        if ear_values:
            # Tính toán ngưỡng EAR mới (90% giá trị trung bình)
            avg_ear = np.mean(ear_values)
            new_threshold = avg_ear * 0.9
            print(f"Hiệu chỉnh hoàn tất. Ngưỡng EAR mới: {new_threshold:.3f}")

            # Cập nhật ngưỡng và lưu cấu hình
            self.EYE_AR_THRESH = new_threshold
            self.config.save_calibration(new_threshold)
            return True
        else:
            print("Hiệu chỉnh thất bại. Không phát hiện được khuôn mặt.")
            return False

    def __del__(self):
        self.stop_camera()