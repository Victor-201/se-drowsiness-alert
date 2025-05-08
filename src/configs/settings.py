# settings.py
import cv2
import logging
import os
import pickle
from src.configs.config import Config

logger = logging.getLogger(__name__)

class Settings:
    def __init__(self):
        # Khởi tạo cài đặt mặc định
        self._camera_index = 0
        self._alert_volume = 50
        self._alert_sound_file = None  
        self.config = Config()
        self.sound_alert_dir = self.config.SOUND_ALERT_DIR
        self.settings_file = self.config.SETTINGS_FILE
        self.load()  # Tải cài đặt từ file

    @property
    def camera_index(self):
        # Lấy chỉ số camera hiện tại
        return self._camera_index

    @camera_index.setter
    def camera_index(self, value):
        # Đặt chỉ số camera
        self._camera_index = value

    @property
    def alert_volume(self):
        # Lấy âm lượng cảnh báo
        return self._alert_volume

    @alert_volume.setter
    def alert_volume(self, value):
        # Đặt âm lượng cảnh báo (giới hạn từ 0 đến 100)
        self._alert_volume = max(0, min(100, value))

    @property
    def alert_sound_file(self):
        # Lấy file âm thanh cảnh báo
        return self._alert_sound_file

    @alert_sound_file.setter
    def alert_sound_file(self, value):
        # Đặt file âm thanh cảnh báo
        self._alert_sound_file = value

    def get_available_cameras(self):
        # Lấy danh sách các camera khả dụng
        index = 0
        cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            cameras.append(index)
            cap.release()
            index += 1
        return cameras

    def get_available_sounds(self):
        # Lấy danh sách các file âm thanh khả dụng trong thư mục âm thanh
        os.makedirs(self.sound_alert_dir, exist_ok=True)
        sound_files = ['Mặc định']  # Tùy chọn âm thanh mặc định
        for file in os.listdir(self.sound_alert_dir):
            if file.lower().endswith(('.wav', '.mp3')):
                sound_files.append(file)
        return sound_files

    def save(self):
        # Lưu cài đặt vào file
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.settings_file, 'wb') as f:
                pickle.dump({
                    'camera_index': self._camera_index,
                    'alert_volume': self._alert_volume,
                    'alert_sound_file': self._alert_sound_file
                }, f)
            logger.info("Đã lưu cài đặt thành công")
        except Exception as e:
            logger.error(f"Lỗi khi lưu cài đặt: {e}")

    def load(self):
        # Tải cài đặt từ file
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'rb') as f:
                    data = pickle.load(f)
                    self._camera_index = data.get('camera_index', 0)
                    self._alert_volume = data.get('alert_volume', 50)
                    self._alert_sound_file = data.get('alert_sound_file', None)
                logger.info("Đã tải cài đặt thành công")
        except Exception as e:
            logger.error(f"Lỗi khi tải cài đặt: {e}")