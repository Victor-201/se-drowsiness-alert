import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import logging
import time
from src.core.detector import DrowsinessDetector, manual_resize
from src.configs.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('drowsiness_detector.log')
    ]
)
logger = logging.getLogger(__name__)


def draw_metrics_overlay(frame, metrics, config):
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    overlay_data = [
        (f"EAR: {metrics['ear']:.2f}", (20, 30), (0, 255, 0) if metrics['ear'] >= config.EAR_THRESHOLD else (0, 0, 255)),
        (f"MAR: {metrics['mar']:.2f}", (20, 55), (0, 255, 0) if metrics['mar'] <= config.YAWN_THRESHOLD else (0, 0, 255)),
        (f"Roll: {metrics['roll_angle']:.1f} | Pitch: {metrics['pitch_angle']:.1f}", (20, 80), (255, 255, 255)),
        (f"Blink: {metrics['blink_count']}  Yawn: {metrics['yawn_count']}", (20, 105), (255, 255, 255)),
        (f"Face: {'Yes' if metrics['face_detected'] else 'No'}", (20, 130),
         (0, 255, 0) if metrics['face_detected'] else (0, 0, 255)),
    ]
    for text, pos, color in overlay_data:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def show_pipeline_windows(frame, metrics, detector):
    small = manual_resize(frame, 320, 240)
    cv2.imshow("Camera", small)

    if detector._last_canny_left is not None and detector._last_canny_right is not None:
        if detector._last_canny_left.size > 0 and detector._last_canny_right.size > 0:
            h_l, w_l = detector._last_canny_left.shape
            h_r, w_r = detector._last_canny_right.shape
            max_h = max(h_l, h_r)
            combined = np.zeros((max_h, w_l + w_r), dtype=np.uint8)
            combined[:h_l, :w_l] = detector._last_canny_left
            combined[:h_r, w_l:w_l + w_r] = detector._last_canny_right
            canny_big = manual_resize(combined, 320, 120)
            cv2.imshow("Canny Edges", canny_big)


def run_detection(save_pipeline=False):
    detector = DrowsinessDetector(save_pipeline=save_pipeline)
    config = Config()

    try:
        detector.start_camera()
    except Exception as e:
        logger.error(f"Không thể khởi tạo camera: {e}")
        return

    logger.info("Nhấn 'q' để thoát, 'p' để bật/tắt lưu pipeline")

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 640, 480)

    while True:
        frame, alert, metrics = detector.process_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        draw_metrics_overlay(frame, metrics, config)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            detector.save_pipeline = not detector.save_pipeline
            logger.info(f"Lưu pipeline: {'BẬT' if detector.save_pipeline else 'TẮT'}")

    detector.stop_camera()
    cv2.destroyAllWindows()


def run_calibration():
    detector = DrowsinessDetector()
    try:
        detector.start_camera()
    except Exception as e:
        logger.error(f"Không thể khởi tạo camera: {e}")
        return

    logger.info("Nhìn thẳng vào camera trong 5 giây để hiệu chỉnh...")
    logger.info("Nhấn 'q' để bỏ qua hiệu chỉnh")

    detector.reset_calibration()
    start_time = time.time()
    duration = 5

    while time.time() - start_time < duration:
        frame, ear = detector.process_calibration_frame()
        if frame is None:
            continue

        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"Hiệu chinh: {remaining:.0f}s", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Hieu chinh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    success, new_threshold = detector.finalize_calibration()
    if success:
        logger.info(f"Hiệu chỉnh hoàn tất. Ngưỡng EAR mới: {new_threshold:.3f}")
    else:
        logger.warning("Hiệu chỉnh thất bại, dùng ngưỡng mặc định")

    detector.stop_camera()
    cv2.destroyAllWindows()


def run_kivy():
    """Chạy giao diện Kivy Desktop (đầy đủ chức năng)"""
    from src.ui.app import DrowsinessDetectorApp
    DrowsinessDetectorApp().run()


if __name__ == '__main__':
    import sys
    if '--calibrate' in sys.argv:
        run_calibration()
    elif '--opencv' in sys.argv:
        run_detection(save_pipeline='--save-pipeline' in sys.argv)
    else:
        run_kivy()
