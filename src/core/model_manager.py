import os
import urllib.request
import bz2
import dlib
import logging
from src.configs.config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        # Khởi tạo với cấu hình và các tham chiếu model
        self.config = Config()
        self._detector = None
        self._predictor = None

    def download_model(self):
        # Tải và giải nén model nhận diện khuôn mặt nếu chưa có
        model_file = self.config.MODEL_DAT
        model_bz2 = model_file + ".bz2"
        os.makedirs(self.config.DATA_DIR, exist_ok=True)

        if not os.path.exists(model_file):
            logger.info(f"Tải model về {model_file}")
            try:
                urllib.request.urlretrieve(self.config.MODEL_DAT_BZ2, model_bz2)
                with open(model_file, 'wb') as new_file, bz2.BZ2File(model_bz2, 'rb') as file:
                    new_file.write(file.read())  # Giải nén file
                os.remove(model_bz2)  # Xóa file nén
                logger.info(f"Model được tải và giải nén tới {model_file}")
            except Exception as e:
                logger.error(f"Tải hoặc giải nén model thất bại: {e}")
                raise
        return model_file

    @property
    def detector(self):
        # Khởi tạo và trả về detector khuôn mặt
        if self._detector is None:
            logger.info("Khởi tạo detector khuôn mặt")
            self._detector = dlib.get_frontal_face_detector()
        return self._detector

    @property
    def predictor(self):
        # Khởi tạo và trả về predictor đặc điểm khuôn mặt
        if self._predictor is None:
            logger.info("Khởi tạo predictor đặc điểm khuôn mặt")
            try:
                self._predictor = dlib.shape_predictor(self.download_model())
            except Exception as e:
                logger.error(f"Khởi tạo predictor thất bại: {e}")
                raise
        return self._predictor