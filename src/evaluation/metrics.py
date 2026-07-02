import numpy as np
import time
import logging
import cv2
import os
from src.core.detector import manual_bgr_to_gray, manual_resize

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, ear_threshold=0.22, mar_threshold=0.3, head_tilt_threshold=45.0,
                 ear_consec_frames=15, head_tilt_frames=20):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.head_tilt_threshold = head_tilt_threshold
        self.ear_consec_frames = ear_consec_frames
        self.head_tilt_frames = head_tilt_frames
        self.reset()

    def reset(self):
        self.ground_truth = []
        self.predictions = []
        self.ear_values = []
        self.mar_values = []
        self.roll_values = []
        self.pitch_values = []
        self.timestamps = []
        self.eye_counter = 0
        self.head_tilt_counter = 0

    def add_sample(self, ear, mar, roll_angle, pitch_angle, is_drowsy_ground_truth=None):
        self.ear_values.append(ear)
        self.mar_values.append(mar)
        self.roll_values.append(roll_angle)
        self.pitch_values.append(pitch_angle)
        self.timestamps.append(time.time())

        if ear < self.ear_threshold:
            self.eye_counter += 1
        else:
            self.eye_counter = 0
        drowsy_pred = self.eye_counter >= self.ear_consec_frames

        if abs(roll_angle) > self.head_tilt_threshold or abs(pitch_angle) > self.head_tilt_threshold:
            self.head_tilt_counter += 1
        else:
            self.head_tilt_counter = max(0, self.head_tilt_counter - 1)
        head_tilt_pred = self.head_tilt_counter >= self.head_tilt_frames

        combined_pred = drowsy_pred or head_tilt_pred
        self.predictions.append(combined_pred or (mar > self.mar_threshold))

        if is_drowsy_ground_truth is not None:
            self.ground_truth.append(is_drowsy_ground_truth)

    def compute_metrics(self):
        if not self.ground_truth or not self.predictions:
            logger.warning("Không có dữ liệu ground truth hoặc prediction để tính metrics")
            return {}

        gt = np.array(self.ground_truth)
        pred = np.array(self.predictions[:len(gt)])

        tp = int(np.sum((pred == 1) & (gt == 1)))
        tn = int(np.sum((pred == 0) & (gt == 0)))
        fp = int(np.sum((pred == 1) & (gt == 0)))
        fn = int(np.sum((pred == 0) & (gt == 1)))

        accuracy = (tp + tn) / len(gt) if len(gt) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics = {
            'total_samples': len(gt),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
        }

        logger.info(f"Accuracy={accuracy:.3f}, Precision={precision:.3f}, "
                    f"Recall={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}")
        return metrics

    def ear_sensitivity_analysis(self, ear_range=None):
        if ear_range is None:
            ear_range = np.arange(0.16, 0.30, 0.02)
        results = []
        for threshold in ear_range:
            self.ear_threshold = float(threshold)
            self.eye_counter = 0
            preds = []
            for ear in self.ear_values:
                if ear < self.ear_threshold:
                    self.eye_counter += 1
                else:
                    self.eye_counter = 0
                preds.append(self.eye_counter >= self.ear_consec_frames)
            if self.ground_truth:
                gt = np.array(self.ground_truth[:len(preds)])
                pred = np.array(preds[:len(gt)])
                tp = int(np.sum((pred == 1) & (gt == 1)))
                fp = int(np.sum((pred == 1) & (gt == 0)))
                fn = int(np.sum((pred == 0) & (gt == 1)))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                results.append({
                    'ear_threshold': float(threshold),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tp': tp, 'fp': fp, 'fn': fn
                })
        return results

    def summary_stats(self):
        if not self.ear_values:
            return {}
        return {
            'ear_mean': float(np.mean(self.ear_values)),
            'ear_std': float(np.std(self.ear_values)),
            'ear_min': float(np.min(self.ear_values)),
            'ear_max': float(np.max(self.ear_values)),
            'mar_mean': float(np.mean(self.mar_values)),
            'mar_std': float(np.std(self.mar_values)),
            'total_frames': len(self.ear_values),
            'duration_seconds': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0,
        }


def evaluate_on_video(video_path, config, output_dir='evaluation_results'):
    """Đánh giá pipeline trên một video có sẵn (real data)."""
    from src.core.detector import DrowsinessDetector
    from src.core.facial_analyzer import FacialAnalyzer
    from src.core.model_manager import ModelManager
    import dlib

    os.makedirs(output_dir, exist_ok=True)

    model_manager = ModelManager()
    predictor = model_manager.predictor
    analyzer = FacialAnalyzer()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Không thể mở video: {video_path}")
        return

    collector = MetricsCollector(
        ear_threshold=config.EAR_THRESHOLD,
        mar_threshold=config.YAWN_THRESHOLD,
        head_tilt_threshold=config.HEAD_TILT_THRESHOLD,
        ear_consec_frames=config.EAR_CONSEC_FRAMES,
        head_tilt_frames=config.HEAD_TILT_FRAMES
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = manual_resize(frame, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        gray = manual_bgr_to_gray(frame)
        gray_eq = analyzer.apply_clahe(gray)

        faces = cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        ear = mar = roll = pitch = 0.0
        drowsy = False

        if len(faces) > 0:
            x, y, w, h = faces[0]
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            dlib_faces = model_manager.detector(gray_eq)
            if dlib_faces:
                shape = predictor(gray_eq, dlib_faces[0])
                shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                left_eye = shape_np[36:42]
                right_eye = shape_np[42:48]
                mouth = shape_np[48:68]
                left_ear = analyzer.calculate_ear(left_eye)
                right_ear = analyzer.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = analyzer.calculate_mar(mouth)
                roll, pitch, _ = analyzer.calculate_head_pose(shape_np)

                if ear < config.EAR_THRESHOLD:
                    drowsy = True

        collector.add_sample(ear, mar, roll, pitch, is_drowsy_ground_truth=drowsy)
        frame_count += 1

        if frame_count % 100 == 0:
            logger.info(f"Đã xử lý {frame_count} frames...")

    cap.release()

    metrics = collector.compute_metrics()
    ear_analysis = collector.ear_sensitivity_analysis()
    summary = collector.summary_stats()

    logger.info(f"\n=== KẾT QUẢ ĐÁNH GIÁ VIDEO ===")
    logger.info(f"Video: {video_path}")
    logger.info(f"Tổng frames: {frame_count}")
    logger.info(f"F1-score: {metrics.get('f1_score', 0):.3f}")
    logger.info(f"Precision: {metrics.get('precision', 0):.3f}")
    logger.info(f"Recall: {metrics.get('recall', 0):.3f}")

    return metrics, ear_analysis, summary
