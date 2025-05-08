import os
import pickle
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    # Cấu hình ngưỡng và thông số
    EAR_THRESHOLD = 0.22
    EAR_CONSEC_FRAMES = 15
    HEAD_TILT_THRESHOLD = 15
    HEAD_TILT_FRAMES = 20
    BLINK_CONSEC_FRAMES = 3
    BLINK_PER_MINUTE_THRESHOLD = 25
    YAWN_THRESHOLD = 0.3
    YAWN_CONSEC_FRAMES = 5
    YAWN_PER_MINUTE_THRESHOLD = 3
    NO_FACE_ALERT_FRAMES = 20
    NOTIFICATION_DURATION = 3.0

    ALERT_COOLDOWN = 3
    ALERT_STOP_DELAY = 1.0
    CAMERA_ID = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    SOUND_ENABLED = True
    PRIMARY_COLOR = (0, 255, 0)
    SECONDARY_COLOR = (255, 165, 0)
    ALERT_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DAT = os.path.join(DATA_DIR, "shape_predictor_68_face_landmarks.dat")
    MODEL_DAT_BZ2 = os.path.join(DATA_DIR, "shape_predictor_68_face_landmarks.dat.bz2")
    CALIB_FILE = os.path.join(DATA_DIR, "calibration.pkl")
    SETTINGS_FILE = os.path.join(DATA_DIR, "settings.pkl")
    ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
    IMAGE_DIR = os.path.join(ASSETS_DIR, "images")
    SOUND_DIR = os.path.join(ASSETS_DIR, "sounds")
    FONT_DIR = os.path.join(ASSETS_DIR, "fonts")
    SOUND_ALERT_DIR = os.path.join(SOUND_DIR, "alerts")
    SOUND_NOTIFICATION_DIR = os.path.join(SOUND_DIR, "notifications")
    FONT_PATH = os.path.join(FONT_DIR, "arial.ttf")
    ALERT_SOUND_FILE = os.path.join(SOUND_ALERT_DIR, "alert.wav")
    FATIGUE_SOUND_FILE = os.path.join(SOUND_NOTIFICATION_DIR, "canh_bao_buon_ngu.mp3")
    FACIAL_LANDMARKS_INDEXES = {
        "right_eye": (36, 42),
        "left_eye": (42, 48),
        "mouth": (48, 68)
    }

    def save_calibration(self, ear_threshold):
        # Lưu ngưỡng EAR đã hiệu chỉnh
        try:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            with open(self.CALIB_FILE, 'wb') as f:
                pickle.dump({'ear_threshold': ear_threshold}, f)
            logger.info("Lưu hiệu chỉnh thành công")
        except Exception as e:
            logger.error(f"Lưu hiệu chỉnh thất bại: {e}")

    def load_calibration(self):
        # Tải ngưỡng EAR đã lưu
        try:
            if os.path.exists(self.CALIB_FILE):
                with open(self.CALIB_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.EAR_THRESHOLD = data.get('ear_threshold', self.EAR_THRESHOLD)
                    logger.info("Tải hiệu chỉnh thành công")
                    return True
        except Exception as e:
            logger.error(f"Tải hiệu chỉnh thất bại: {e}")
        return False

    def save_settings(self, settings_data):
        try:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            with open(self.SETTINGS_FILE, 'wb') as f:
                pickle.dump(settings_data, f)
            logger.info("Lưu cài đặt thành công")
            return True
        except Exception as e:
            logger.error(f"Lưu cài đặt thất bại: {e}")
            return False

    def load_settings(self):
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, 'rb') as f:
                    settings_data = pickle.load(f)
                    logger.info("Tải cài đặt thành công")
                    return settings_data
        except Exception as e:
            logger.error(f"Tải cài đặt thất bại: {e}")
        return None