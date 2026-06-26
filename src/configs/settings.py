import cv2
import logging
import os
import pickle
import concurrent.futures
from src.configs.config import Config

logger = logging.getLogger(__name__)


class Settings:
    def __init__(self):
        self._camera_index = 0
        self._alert_volume = 50
        self._alert_sound_file = None
        self.config = Config()
        self.settings_file = os.path.join(self.config.DATA_DIR, "settings.pkl")
        self.load()

    @property
    def camera_index(self):
        return self._camera_index

    @camera_index.setter
    def camera_index(self, value):
        self._camera_index = value

    @property
    def alert_volume(self):
        return self._alert_volume

    @alert_volume.setter
    def alert_volume(self, value):
        self._alert_volume = max(0, min(100, value))

    @property
    def alert_sound_file(self):
        return self._alert_sound_file

    @alert_sound_file.setter
    def alert_sound_file(self, value):
        self._alert_sound_file = value

    def get_available_sounds(self):
        sounds = ['Mặc định']
        if os.path.exists(self.config.SOUND_ALERT_DIR):
            for f in os.listdir(self.config.SOUND_ALERT_DIR):
                if f.lower().endswith(('.wav', '.mp3', '.ogg')):
                    sounds.append(f)
        return sounds

    def _try_camera(self, index):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                return True
            cap.release()
        except Exception:
            pass
        return False

    def get_available_cameras(self):
        cameras = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for i in range(10):
                future = executor.submit(self._try_camera, i)
                futures[future] = i
            for future in concurrent.futures.as_completed(futures, timeout=3.0):
                idx = futures[future]
                try:
                    if future.result():
                        cameras.append(idx)
                except Exception:
                    pass
        return cameras

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'wb') as f:
                pickle.dump({
                    'camera_index': self._camera_index,
                    'alert_volume': self._alert_volume,
                    'alert_sound_file': self._alert_sound_file,
                }, f)
        except Exception as e:
            logger.error(f"Lưu cài đặt thất bại: {e}")

    def load(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'rb') as f:
                    data = pickle.load(f)
                    self._camera_index = data.get('camera_index', 0)
                    self._alert_volume = data.get('alert_volume', 50)
                    self._alert_sound_file = data.get('alert_sound_file', None)
        except Exception as e:
            logger.error(f"Tải cài đặt thất bại: {e}")
