import cv2
import dlib
import numpy as np
import time
import logging
import os
from collections import deque
from src.configs.config import Config
from src.core.model_manager import ModelManager
from src.core.facial_analyzer import FacialAnalyzer
from src.core.alert_system import AlertSystem

logger = logging.getLogger(__name__)


def manual_bgr_to_gray(bgr):
    b = bgr[:, :, 0].astype(np.float64)
    g = bgr[:, :, 1].astype(np.float64)
    r = bgr[:, :, 2].astype(np.float64)
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)


def manual_resize(img, new_w, new_h):
    old_h, old_w = img.shape[:2]
    if old_h == new_h and old_w == new_w:
        return img.copy()

    oy = np.arange(new_h, dtype=np.float64)
    ox = np.arange(new_w, dtype=np.float64)
    iy = (oy + 0.5) * old_h / new_h - 0.5
    ix = (ox + 0.5) * old_w / new_w - 0.5
    iy = np.clip(iy, 0, old_h - 1)
    ix = np.clip(ix, 0, old_w - 1)
    iy0 = np.floor(iy).astype(np.intp)
    iy1 = np.minimum(iy0 + 1, old_h - 1)
    ix0 = np.floor(ix).astype(np.intp)
    ix1 = np.minimum(ix0 + 1, old_w - 1)
    dy = (iy - iy0)[:, np.newaxis]
    dx = (ix - ix0)[np.newaxis, :]

    if img.ndim == 2:
        v00 = img[iy0[:, np.newaxis], ix0[np.newaxis, :]]
        v10 = img[iy1[:, np.newaxis], ix0[np.newaxis, :]]
        v01 = img[iy0[:, np.newaxis], ix1[np.newaxis, :]]
        v11 = img[iy1[:, np.newaxis], ix1[np.newaxis, :]]
        result = (v00 * (1 - dx) + v01 * dx) * (1 - dy) + (v10 * (1 - dx) + v11 * dx) * dy
        return np.clip(np.round(result), 0, 255).astype(img.dtype)
    else:
        c = img.shape[2]
        result = np.zeros((new_h, new_w, c), dtype=img.dtype)
        for k in range(c):
            v00 = img[iy0[:, np.newaxis], ix0[np.newaxis, :], k]
            v10 = img[iy1[:, np.newaxis], ix0[np.newaxis, :], k]
            v01 = img[iy0[:, np.newaxis], ix1[np.newaxis, :], k]
            v11 = img[iy1[:, np.newaxis], ix1[np.newaxis, :], k]
            result[:, :, k] = (v00 * (1 - dx) + v01 * dx) * (1 - dy) + (v10 * (1 - dx) + v11 * dx) * dy
        return np.clip(np.round(result), 0, 255).astype(img.dtype)


def non_max_suppression(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    for i in order:
        if len(keep) == 0:
            keep.append(i)
            continue
        xx1 = np.maximum(x1[i], x1[keep])
        yy1 = np.maximum(y1[i], y1[keep])
        xx2 = np.minimum(x2[i], x2[keep])
        yy2 = np.minimum(y2[i], y2[keep])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[i]
        if np.all(overlap <= iou_threshold):
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


class PipelineStage:
    def __init__(self, output_dir="pipeline_output"):
        self.output_dir = output_dir
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def save_stage(self, name, image, frame_id=None):
        if frame_id is not None:
            self.frame_count = frame_id
        filename = f"frame{self.frame_count:04d}_{name}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath


class DrowsinessDetector:
    def __init__(self, save_pipeline=False):
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
        self.face_cascade = self._init_face_cascade()
        self.face_net_dnn = self._init_face_detector_dnn()
        self.face_cnn_detector = self._init_face_detector_cnn()
        self.landmark_predictor = self.model_manager.predictor
        self.head_tilt_threshold = self.config.HEAD_TILT_THRESHOLD
        self.head_tilt_frames = self.config.HEAD_TILT_FRAMES
        self.head_tilt_counter = 0
        self.reference_roll = None
        self.reference_pitch = None
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
        self.save_pipeline = save_pipeline
        self.pipeline = PipelineStage() if save_pipeline else None
        self._last_canny_left = None
        self._last_canny_right = None

    def _init_face_cascade(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            logger.error(f"Haar cascade not found at {cascade_path}")
            return None
        logger.info("Initialized Haar cascade face detector")
        return cv2.CascadeClassifier(cascade_path)

    def _init_face_detector_dnn(self):
        proto = self.config.DNN_PROTOTXT
        model = self.config.DNN_CAFFEMODEL
        if not os.path.exists(proto) or not os.path.exists(model):
            logger.warning("DNN model files not found, downloading...")
            self.model_manager.download_dnn_models()
        try:
            net = cv2.dnn.readNetFromCaffe(proto, model)
            logger.info("Initialized OpenCV DNN face detector (SSD)")
            return net
        except Exception as e:
            logger.error(f"DNN face detector init failed: {e}")
            return None

    def detect_faces_dnn(self, frame):
        if self.face_net_dnn is None:
            return [], []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.face_net_dnn.setInput(blob)
        detections = self.face_net_dnn.forward()
        boxes = []
        scores = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.config.DNN_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
        if boxes:
            boxes, scores = non_max_suppression(boxes, scores, self.config.DNN_NMS_THRESHOLD)
        if boxes:
            boxes = [[int(x) for x in b] for b in boxes]
            scores = [float(s) for s in scores]
        return boxes, scores

    def _init_face_detector_cnn(self):
        model_path = self.config.CNN_FACE_MODEL
        if not os.path.exists(model_path):
            logger.warning("CNN face model not found, downloading...")
            self.model_manager.download_cnn_face_model()
        try:
            cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
            logger.info("Initialized dlib CNN face detector (MMOD)")
            return cnn_detector
        except Exception as e:
            logger.error(f"CNN face detector init failed: {e}")
            return None

    def detect_faces_cnn(self, gray):
        if self.face_cnn_detector is None:
            return []
        faces = self.face_cnn_detector(gray, 1)
        boxes = []
        for f in faces:
            r = f.rect
            boxes.append([int(r.left()), int(r.top()), int(r.right()), int(r.bottom())])
        return boxes

    def detect_faces_haar(self, gray):
        if self.face_cascade is None:
            return []
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
        return boxes

    def detect_faces_dlib(self, gray):
        return self.model_manager.detector(gray)

    def start_camera(self):
        logger.info("Initializing camera...")
        if self.camera and self.camera.isOpened():
            return
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for index in [self.config.CAMERA_ID, 1]:
            for backend in backends:
                try:
                    self.camera = cv2.VideoCapture(index, backend)
                    if self.camera.isOpened():
                        break
                    self.camera.release()
                    self.camera = None
                except Exception:
                    self.camera = None
            if self.camera and self.camera.isOpened():
                break
        if not self.camera or not self.camera.isOpened():
            raise IOError("Cannot open camera")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        logger.info("Camera initialized successfully")

    def stop_camera(self):
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.analyzer.reset_display()
            self.camera = None
            logger.info("Camera stopped")

    def detect_blink(self, ear):
        self.ear_history.append(ear)
        if len(self.ear_history) < self.blink_consec_frames:
            return False
        recent_ear_avg = float(np.mean(list(self.ear_history)[-10:])) if len(self.ear_history) >= 10 else float(np.mean(self.ear_history))
        dynamic_threshold = min(self.ear_threshold, recent_ear_avg * 0.8)
        if not self.eye_closed and ear < dynamic_threshold:
            self.eye_closed = True
            return False
        elif self.eye_closed and ear >= dynamic_threshold:
            self.eye_closed = False
            current_time = time.time()
            self.blink_total += 1
            self.blink_times.append(current_time)
            return True
        return False

    def detect_yawn(self, mar):
        current_time = time.time()
        if mar > self.yawn_threshold and not self.mouth_open:
            self.yawn_counter += 1
            if self.yawn_counter >= self.yawn_consec_frames:
                if not self.yawn_times or (current_time - self.yawn_times[-1] >= 4):
                    self.mouth_open = True
                    self.yawn_counter = 0
                    self.yawn_total += 1
                    self.yawn_times.append(current_time)
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
        current_time = time.time()
        recent_yawns = [t for t in self.yawn_times if current_time - t <= 60]
        return len(recent_yawns) >= self.yawn_per_minute_threshold

    def check_blink_frequency(self):
        current_time = time.time()
        recent_blinks = [t for t in self.blink_times if current_time - t <= 60]
        return len(recent_blinks) >= self.blink_per_minute_threshold

    def reset_counters_if_needed(self):
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self.blink_times.clear()
            self.yawn_times.clear()
            self.blink_total = 0
            self.yawn_total = 0
            self.fatigue_alert_count = 0
            self.last_reset_time = current_time

    def process_frame(self):
        self.reset_counters_if_needed()
        if not self.camera or not self.camera.isOpened():
            try:
                self.start_camera()
            except Exception as e:
                logger.error(f"Failed to reinitialize camera: {e}")
                return None, False, self._empty_metrics()

        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None, False, self._empty_metrics()

        frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        stage_images = {}

        gray_eq = self.analyzer.apply_clahe(gray)
        stage_images["01_grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        stage_images["02_clahe"] = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

        face_boxes = []
        face_scores = []

        try:
            dnn_boxes, dnn_scores = self.detect_faces_dnn(frame)
            if dnn_boxes:
                face_boxes = dnn_boxes
                face_scores = dnn_scores
        except Exception as e:
            logger.warning(f"DNN face detection failed: {e}")

        if not face_boxes:
            try:
                faces_dlib = self.detect_faces_dlib(gray_eq)
                if faces_dlib:
                    face_boxes = [[f.left(), f.top(), f.right(), f.bottom()] for f in faces_dlib]
                    face_scores = [0.9] * len(faces_dlib)
            except Exception as e:
                logger.warning(f"dlib face detection failed: {e}")

        if not face_boxes:
            try:
                faces_haar = self.detect_faces_haar(gray_eq)
                if faces_haar:
                    face_boxes = faces_haar
                    face_scores = [0.5] * len(faces_haar)
            except Exception as e:
                logger.warning(f"Haar face detection failed: {e}")

        if not face_boxes:
            self.no_face_counter += 1
            self.face_detected = False
            stage_images["03_face_detection"] = frame.copy()
            if self.no_face_counter >= self.no_face_alert_frames:
                frame = self.alert_system.render_distraction_alert(frame)
                if self.save_pipeline:
                    for name, img in stage_images.items():
                        self.pipeline.save_stage(name, img)
                return frame, True, self._empty_metrics()
            stage_images["03_face_detection"] = frame.copy()
            frame = self.alert_system.put_text_unicode(frame, "Không phát hiện khuôn mặt", (20, 30), self.config.ALERT_COLOR, font_size=24)
            if self.save_pipeline:
                for name, img in stage_images.items():
                    self.pipeline.save_stage(name, img)
            return frame, False, self._empty_metrics()

        self.no_face_counter = 0
        self.face_detected = True

        face_box = face_boxes[0]
        face_vis = frame.copy()
        for fb in face_boxes:
            cv2.rectangle(face_vis, (fb[0], fb[1]), (fb[2], fb[3]), (0, 255, 0), 2)
        stage_images["03_face_detection"] = face_vis

        drowsiness_detected = False
        head_tilt_detected = False
        fatigue_detected = False
        ear = 0.0
        mar = 0.0
        roll_angle = 0.0
        pitch_angle = 0.0
        pitch_ratio = 0.0

        dlib_face = dlib.rectangle(int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3]))

        if dlib_face:
            shape = self.landmark_predictor(gray_eq, dlib_face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape_np[36:42]
            right_eye = shape_np[42:48]
            mouth = shape_np[48:68]

            left_ear = self.analyzer.calculate_ear(left_eye)
            right_ear = self.analyzer.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = self.analyzer.calculate_mar(mouth)
            roll_angle, pitch_angle, pitch_ratio = self.analyzer.calculate_head_pose(shape_np)

            if self.reference_roll is None:
                self.reference_roll = roll_angle
                self.reference_pitch = pitch_angle

            edges_left = self.analyzer.apply_canny_on_eye(gray_eq, left_eye, 50, 150)
            edges_right = self.analyzer.apply_canny_on_eye(gray_eq, right_eye, 50, 150)
            self._last_canny_left = edges_left
            self._last_canny_right = edges_right

            iris_area_left = self.analyzer.detect_iris_by_contour(edges_left)
            iris_area_right = self.analyzer.detect_iris_by_contour(edges_right)

            roi_vis = frame.copy()
            cv2.polylines(roi_vis, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(roi_vis, [right_eye], True, (0, 255, 0), 1)
            cv2.polylines(roi_vis, [mouth], True, (0, 255, 0), 1)
            stage_images["04_landmarks_roi"] = roi_vis

            if edges_left.size > 0 and edges_right.size > 0:
                h_l, w_l = edges_left.shape
                h_r, w_r = edges_right.shape
                max_h = max(h_l, h_r)
                combined_w = w_l + w_r
                canny_vis = np.zeros((max_h, combined_w), dtype=np.uint8)
                canny_vis[:h_l, :w_l] = edges_left
                canny_vis[:h_r, w_l:w_l + w_r] = edges_right
                stage_images["05_canny_edges"] = cv2.cvtColor(canny_vis, cv2.COLOR_GRAY2BGR)

            self.detect_blink(ear)
            self.detect_yawn(mar)
            blink_frequent = self.check_blink_frequency()
            yawn_frequent = self.check_yawn_frequency()
            fatigue_detected = blink_frequent or yawn_frequent

            delta_roll = abs(roll_angle - self.reference_roll) if self.reference_roll is not None else 0
            delta_pitch = abs(pitch_angle - self.reference_pitch) if self.reference_pitch is not None else 0
            head_tilted = delta_roll > self.head_tilt_threshold or delta_pitch > self.head_tilt_threshold
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

            self.draw_facial_ratios(frame, shape_np)

        stage_images["06_result"] = frame

        if self.save_pipeline:
            fid = int(time.time() * 1000) % 100000
            for name, img in stage_images.items():
                self.pipeline.save_stage(name, img, frame_id=fid)

        metrics = {
            'ear': ear,
            'mar': mar,
            'roll_angle': roll_angle,
            'pitch_angle': pitch_angle,
            'pitch_ratio': pitch_ratio,
            'blink_count': self.blink_total,
            'yawn_count': self.yawn_total,
            'face_detected': self.face_detected,
            'head_tilt_detected': head_tilt_detected,
            'fatigue_detected': fatigue_detected,
            'drowsiness_detected': drowsiness_detected,
            'eye_counter': self.eye_counter,
            'head_tilt_counter': self.head_tilt_counter,
        }
        return frame, drowsiness_detected or head_tilt_detected or fatigue_detected, metrics

    def draw_facial_ratios(self, frame, shape_np):
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        for i in range(68):
            cv2.circle(frame, tuple(shape_np[i]), 1, self.config.PRIMARY_COLOR, -1)
        cv2.polylines(frame, [shape_np[36:42]], True, self.config.PRIMARY_COLOR, 1)
        cv2.polylines(frame, [shape_np[42:48]], True, self.config.PRIMARY_COLOR, 1)
        cv2.polylines(frame, [shape_np[48:60]], True, self.config.PRIMARY_COLOR, 1)
        cv2.polylines(frame, [shape_np[27:36]], True, self.config.PRIMARY_COLOR, 1)
        return frame

    def reset_calibration(self):
        self.calibration_ear_values = []

    def reset_head_reference(self):
        self.reference_roll = None
        self.reference_pitch = None

    def process_calibration_frame(self):
        if not self.camera or not self.camera.isOpened():
            return None, 0.0
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None, 0.0
        frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = self.analyzer.apply_clahe(gray)
        faces = self.detect_faces_dlib(gray_eq)
        ear = 0.0
        if faces:
            shape = self.landmark_predictor(gray_eq, faces[0])
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = shape_np[36:42]
            right_eye = shape_np[42:48]
            left_ear = self.analyzer.calculate_ear(left_eye)
            right_ear = self.analyzer.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            self.calibration_ear_values.append(ear)
            self.draw_facial_ratios(frame, shape_np)
        return frame, ear

    def finalize_calibration(self):
        if self.calibration_ear_values:
            avg_ear = float(np.mean(self.calibration_ear_values))
            new_threshold = avg_ear * 0.9
            self.ear_threshold = new_threshold
            self.config.save_calibration(new_threshold)
            self.calibration_ear_values = []
            return True, new_threshold
        self.calibration_ear_values = []
        return False, self.ear_threshold

    @staticmethod
    def _empty_metrics():
        return {
            'ear': 0.0, 'mar': 0.0, 'roll_angle': 0.0, 'pitch_angle': 0.0,
            'pitch_ratio': 0.0,
            'blink_count': 0, 'yawn_count': 0, 'face_detected': False,
            'head_tilt_detected': False, 'fatigue_detected': False,
            'drowsiness_detected': False, 'eye_counter': 0, 'head_tilt_counter': 0,
        }

    def __del__(self):
        self.stop_camera()
