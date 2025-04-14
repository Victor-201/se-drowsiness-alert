import urllib.request
import bz2
import os
import dlib
from ..configs.config import Config

class ModelManager:
    def __init__(self):
        self.config = Config()
        self._detector = None
        self._predictor = None

    def download_model(self):
        model_file = self.config.MODEL_FILE
        model_bz2 = model_file + ".bz2"
        if not os.path.exists(model_file):
            print("Đang tải mô hình...")
            urllib.request.urlretrieve(self.config.MODEL_URL, model_bz2)
            with open(model_file, 'wb') as new_file, bz2.BZ2File(model_bz2, 'rb') as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)
            os.remove(model_bz2)
            print("Mô hình đã sẵn sàng.")
        return model_file

    @property
    def detector(self):
        if self._detector is None:
            self._detector = dlib.get_frontal_face_detector()
        return self._detector

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = dlib.shape_predictor(self.download_model())
        return self._predictor