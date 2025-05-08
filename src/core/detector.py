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
        """Khởi tạo bộ phát hiện buồn ngủ với các cấu hình và mô hình cần thiết."""
        self.config = Config()
        self.model_manager = ModelManager()
        self.analyzer = FacialAnalyzer()
        self.alert_system = AlertSystem()
        self.camera = None
        self.ear_threshold = self.config.EAR_THRESHOLD
        self.ear_consec_frames = self.config.EAR_CONSEC_FRAMES
        self.blink_consec_frames = self.config.BLINK_CONSEC_FRAMES
        self.no_face_alert_frames = self.config.NO_FACE_ALERT_FRAMES
        self.eye_counter = 0
        self.no_face_counter = 0
        self.face_detected = False
        self.drowsiness_start_time = None
        self.face_detector = self.model_manager.detector
        self.landmark_predictor = self.model_manager.predictor
        self.head_tilt_threshold = self.config.HEAD_TILT_THRESHOLD
        self.head_tilt_frames = self.config.HEAD_TILT_FRAMES
        self.head_tilt_counter = 0
        self.blink_total = 0
        self.blink_per_minute_threshold = self.config.BLINK_PER_MINUTE_THRESHOLD
        self.yawn_threshold = self.config.YAWN_THRESHOLD
        self.yawn_consec_frames = self.config.YAWN_CONSEC_FRAMES
        self.yawn_counter = 0
        self.yawn_total = 0
        self.yawn_times = deque(maxlen=100)
        self.yawn_per_minute_threshold = self.config.YAWN_PER_MINUTE_THRESHOLD
        self.mouth_open = False
        self.eye_closed = False
        self.ear_history = deque(maxlen=30)
        self.blink_times = deque(maxlen=100)
        self.fatigue_alert = False
        self.fatigue_start_time = None
        self.last_reset_time = time.time()
        self.calibration_ear_values = []
        self.fatigue_alert_count = 0
        self.notification_duration = self.config.NOTIFICATION_DURATION

    def start_camera(self):
        """Khởi động camera với các thông số cấu hình."""
        logger.info("Khởi tạo camera...")
        if self.camera and self.camera.isOpened():
            logger.info("Camera đã được khởi tạo, bỏ qua khởi tạo lại")
            return
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
        """Dừng và giải phóng camera."""
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.analyzer.reset_display()
            self.camera = None
            logger.info("Dừng camera")

    def detect_blink(self, ear):
        """
        Phát hiện nháy mắt dựa trên EAR (Eye Aspect Ratio).
        Trả về True nếu phát hiện nháy mắt, False nếu không.
        """
        self.ear_history.append(ear)
        
        # Đảm bảo có đủ dữ liệu để xử lý
        if len(self.ear_history) < self.blink_consec_frames:
            return False
        
        # Tính ngưỡng động dựa trên EAR trung bình gần đây
        recent_ear_avg = np.mean(list(self.ear_history)[-10:]) if len(self.ear_history) >= 10 else np.mean(self.ear_history)
        dynamic_threshold = min(self.ear_threshold, recent_ear_avg * 0.8)
        
        # Phát hiện nháy mắt khi trạng thái mắt thay đổi từ đóng sang mở
        if not self.eye_closed and ear < dynamic_threshold:
            self.eye_closed = True
            return False
        elif self.eye_closed and ear >= dynamic_threshold:
            self.eye_closed = False
            current_time = time.time()
            self.blink_total += 1
            self.blink_times.append(current_time)
            logger.debug(f"Nháy mắt phát hiện, tổng số: {self.blink_total}")
            return True
        
        return False

    def detect_yawn(self, mar):
        """
        Phát hiện ngáp dựa trên MAR (Mouth Aspect Ratio).
        Trả về True nếu phát hiện ngáp, False nếu không.
        """
        current_time = time.time()

        if mar > self.yawn_threshold and not self.mouth_open:
            self.yawn_counter += 1
            if self.yawn_counter >= self.yawn_consec_frames:
                # Kiểm tra khoảng cách với lần ngáp gần nhất
                if not self.yawn_times or (current_time - self.yawn_times[-1] >= 4):
                    self.mouth_open = True
                    self.yawn_counter = 0
                    self.yawn_total += 1
                    self.yawn_times.append(current_time)
                    logger.debug(f"Ngáp phát hiện, tổng số: {self.yawn_total}")
                    return True
                else:
                    self.yawn_counter = 0  
        elif mar <= self.yawn_threshold and self.mouth_open:
            self.mouth_open = False
            self.yawn_counter = 0
        else:
            self.yawn_counter = max(0, self.yawn_counter - 1)

        return False

    def check_yawn_frequency(self):
        """
        Kiểm tra tần suất ngáp trong 1 phút.
        Trả về True nếu vượt ngưỡng, False nếu không.
        """
        current_time = time.time()
        # Lọc các lần ngáp trong vòng 60 giây qua
        recent_yawns = [t for t in self.yawn_times if current_time - t <= 60]
        yawns_per_minute = len(recent_yawns)
        return yawns_per_minute >= self.yawn_per_minute_threshold

    def check_blink_frequency(self):
        """
        Kiểm tra tần suất nháy mắt trong 1 phút.
        Trả về True nếu vượt ngưỡng, False nếu không.
        """
        current_time = time.time()
        # Lọc các lần nháy mắt trong vòng 60 giây qua
        recent_blinks = [t for t in self.blink_times if current_time - t <= 60]
        blinks_per_minute = len(recent_blinks)
        return blinks_per_minute >= self.blink_per_minute_threshold
    
    def reset_counters_if_needed(self):
        """
        Xóa các lần nháy mắt, ngáp và thông báo đã cũ quá 60 giây.
        """
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            # Giữ lại chỉ các lần trong 60 giây gần nhất
            old_blink_count = len(self.blink_times)
            old_yawn_count = len(self.yawn_times)
            old_fatigue_alert_count = self.fatigue_alert_count
            
            # Cập nhật lại tổng số
            self.blink_times.clear()
            self.yawn_times.clear()
            self.blink_total = 0
            self.yawn_total = 0
            self.fatigue_alert_count = 0
            self.last_reset_time = current_time
            logger.info(f"[Reset counters] Blink {old_blink_count} → {self.blink_total}, Yawn {old_yawn_count} → {self.yawn_total}, Fatigue Alerts {old_fatigue_alert_count} → {self.fatigue_alert_count}")
            
    @staticmethod
    def find_largest_face(faces):
        """Tìm khuôn mặt lớn nhất trong danh sách các khuôn mặt được phát hiện."""
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
        self.reset_counters_if_needed()
        """Xử lý một khung hình từ camera và phát hiện buồn ngủ, ngáp, hoặc mệt mỏi."""
        if not self.camera or not self.camera.isOpened():
            logger.error("Camera chưa khởi tạo hoặc đã bị đóng, thử khởi tạo lại")
            try:
                self.start_camera()
            except Exception as e:
                logger.error(f"Không thể khởi tạo lại camera: {e}")
                return None, False

        ret, frame = self.camera.read()
        if not ret or frame is None:
            logger.error("Không thể lấy khung hình, thử khởi tạo lại camera")
            try:
                self.stop_camera()
                self.start_camera()
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.error("Vẫn không thể lấy khung hình sau khi khởi tạo lại")
                    return None, False
            except Exception as e:
                logger.error(f"Khởi tạo lại camera thất bại: {e}")
                return None, False

        frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        if not faces:
            self.no_face_counter += 1
            self.face_detected = False
            if self.no_face_counter >= self.no_face_alert_frames:
                frame = self.alert_system.render_distraction_alert(frame)
                return frame, True
            frame = self.alert_system.put_text_unicode(frame, "Không phát hiện khuôn mặt", (20, 30), self.config.ALERT_COLOR, font_size=24)
            return frame, False

        self.no_face_counter = 0
        self.face_detected = True
        drowsiness_detected = False
        head_tilt_detected = False
        fatigue_detected = False
        largest_face = self.find_largest_face(faces)

        if largest_face:
            shape = self.landmark_predictor(gray, largest_face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            x1, y1, x2, y2 = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.config.PRIMARY_COLOR, 2)
            self.draw_facial_ratios(frame, shape_np)
            left_eye, right_eye = shape_np[36:42], shape_np[42:48]
            mouth = shape_np[48:68]
            left_ear = self.analyzer.calculate_ear(left_eye)
            right_ear = self.analyzer.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = self.analyzer.calculate_mar(mouth)
            self.detect_blink(ear)
            self.detect_yawn(mar)
            blink_frequent = self.check_blink_frequency()
            yawn_frequent = self.check_yawn_frequency()
            fatigue_detected = blink_frequent or yawn_frequent
            roll_angle, pitch_angle = self.analyzer.calculate_head_pose(shape_np)
            head_tilted = roll_angle > self.head_tilt_threshold or pitch_angle > self.head_tilt_threshold
            if head_tilted:
                self.head_tilt_counter += 1
                if self.head_tilt_counter >= self.head_tilt_frames:
                    head_tilt_detected = True
            else:
                self.head_tilt_counter = max(0, self.head_tilt_counter - 1)
            if ear < self.ear_threshold:
                self.eye_counter += 1
                if self.eye_counter == 1:
                    self.drowsiness_start_time = time.time()
                if self.eye_counter >= self.ear_consec_frames:
                    drowsiness_detected = True
                    drowsiness_duration = time.time() - self.drowsiness_start_time
                    frame = self.alert_system.render_drowsiness_alert(frame, drowsiness_duration)
            else:
                self.eye_counter = 0
                self.drowsiness_start_time = None
            if head_tilt_detected:
                frame = self.alert_system.render_head_tilt_alert(frame)
            if fatigue_detected and self.fatigue_alert_count < 1:
                if self.fatigue_start_time is None:
                    self.fatigue_start_time = time.time()
                frame = self.alert_system.render_fatigue_alert(frame)
                if time.time() - self.fatigue_start_time >= self.notification_duration:
                    self.fatigue_alert_count += 1
                    self.fatigue_start_time = None

        return frame, drowsiness_detected or head_tilt_detected or fatigue_detected

    def draw_facial_ratios(self, frame, shape_np):
        """Vẽ các đặc điểm khuôn mặt lên khung hình."""
        for i in range(68):
            cv2.circle(frame, tuple(shape_np[i]), 1, self.config.PRIMARY_COLOR, -1)
        cv2.polylines(frame, [shape_np[36:42]], True, self.config.PRIMARY_COLOR, 1)
        cv2.polylines(frame, [shape_np[42:48]], True, self.config.PRIMARY_COLOR, 1)
        cv2.polylines(frame, [shape_np[48:60]], True, self.config.PRIMARY_COLOR, 1)
        cv2.polylines(frame, [shape_np[27:36]], True, self.config.PRIMARY_COLOR, 1)
        return frame

    def reset_calibration(self):
        """Reset trạng thái hiệu chỉnh."""
        self.calibration_ear_values = []
        logger.info("Đã reset trạng thái hiệu chỉnh")

    def process_calibration_frame(self):
        """Xử lý khung hình để hiệu chỉnh EAR."""
        if not self.camera or not self.camera.isOpened():
            logger.error("Camera chưa khởi tạo hoặc đã bị đóng")
            return None, 0.0
        ret, frame = self.camera.read()
        if not ret or frame is None:
            logger.error("Không thể lấy khung hình")
            return None, 0.0
        frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        ear = 0.0
        largest_face = self.find_largest_face(faces)
        if largest_face:
            shape = self.landmark_predictor(gray, largest_face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye, right_eye = shape_np[36:42], shape_np[42:48]
            left_ear = self.analyzer.calculate_ear(left_eye)
            right_ear = self.analyzer.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            self.calibration_ear_values.append(ear)
            self.draw_facial_ratios(frame, shape_np)
            frame = self.alert_system.put_text_unicode(frame, f"EAR hiện tại: {ear:.2f}", (20, 60), (0, 255, 0), font_size=24)
        else:
            frame = self.alert_system.put_text_unicode(frame, "Không phát hiện khuôn mặt", (20, 30), self.config.ALERT_COLOR, font_size=24)
        return frame, ear

    def finalize_calibration(self):
        """Hoàn tất hiệu chỉnh và lưu ngưỡng EAR mới."""
        if self.calibration_ear_values:
            avg_ear = np.mean(self.calibration_ear_values)
            new_threshold = avg_ear * 0.9
            logger.info(f"Hiệu chỉnh hoàn tất. Ngưỡng EAR mới: {new_threshold:.3f}")
            self.ear_threshold = new_threshold
            self.config.save_calibration(new_threshold)
            self.calibration_ear_values = []
            return True, new_threshold
        logger.error("Hiệu chỉnh thất bại: Không phát hiện khuôn mặt")
        self.calibration_ear_values = []
        return False, self.ear_threshold

    def __del__(self):
        """Giải phóng tài nguyên khi đối tượng bị hủy."""
        self.stop_camera()