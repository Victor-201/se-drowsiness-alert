import os
import urllib.request
import bz2
import dlib
import logging
from src.configs.config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.config = Config()
        self._detector = None
        self._predictor = None

    def download_model(self):
        model_file = self.config.MODEL_DAT
        model_bz2 = model_file + ".bz2"
        os.makedirs(self.config.DATA_DIR, exist_ok=True)

        if not os.path.exists(model_file):
            logger.info(f"Downloading model to {model_file}")
            try:
                urllib.request.urlretrieve(self.config.MODEL_DAT_URL, model_bz2)
                with open(model_file, 'wb') as new_file, bz2.BZ2File(model_bz2, 'rb') as file:
                    new_file.write(file.read())
                os.remove(model_bz2)
                logger.info(f"Model downloaded and extracted to {model_file}")
            except Exception as e:
                logger.error(f"Model download or extraction failed: {e}")
                raise
        return model_file

    @property
    def detector(self):
        if self._detector is None:
            logger.info("Initializing face detector")
            self._detector = dlib.get_frontal_face_detector()
        return self._detector

    @property
    def predictor(self):
        if self._predictor is None:
            logger.info("Initializing facial landmark predictor")
            try:
                self._predictor = dlib.shape_predictor(self.download_model())
            except Exception as e:
                logger.error(f"Predictor initialization failed: {e}")
                raise
        return self._predictor
