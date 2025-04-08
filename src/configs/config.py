import os
import pickle
from playsound import playsound
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    EAR_THRESHOLD = 0.2
    MAR_THRESHOLD = 0.5
    EAR_CONSEC_FRAMES = 15  
    BLINK_CONSEC_FRAMES = 3
    NO_FACE_ALERT_FRAMES = 60

    ALERT_COOLDOWN = 3.0
    ALERT_STOP_DELAY = 1.0
    
    CAMERA_ID = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    PRIMARY_COLOR = (0, 255, 0)
    ALERT_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_FILE = os.path.join(DATA_DIR,"shape_predictor_68_face_landmarks.dat")
    CALIB_FILE = os.path.join(DATA_DIR,"calibration.pkl")
    ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
    ALERT_SOUND_FILE = os.path.join(ASSETS_DIR, "alert.wav")

    # Chỉ số landmarks khuôn mặt
    FACIAL_LANDMARKS_INDEXES = {
        "right_eye": (36, 42),
        "left_eye": (42, 48),
        "mouth": (48, 68)
    }
    def save_calibration(self, ear_threshold):
        with open(self.CALIB_FILE, 'wb') as f:
            pickle.dump({'ear_threshold': ear_threshold}, f)

    def load_calibration(self):
        if os.path.exists(self.CALIB_FILE):
            with open(self.CALIB_FILE, 'rb') as f:
                data = pickle.load(f)
                self.EAR_THRESHOLD = data.get('ear_threshold', self.EAR_THRESHOLD)
                return True
        return False
