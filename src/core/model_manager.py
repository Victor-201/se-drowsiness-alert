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

    def download_dnn_models(self):
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        proto_path = self.config.DNN_PROTOTXT
        model_path = self.config.DNN_CAFFEMODEL
        if not os.path.exists(proto_path):
            logger.info(f"Downloading DNN prototxt to {proto_path}")
            try:
                urllib.request.urlretrieve(proto_url, proto_path)
                logger.info("DNN prototxt downloaded")
            except Exception as e:
                logger.error(f"Failed to download DNN prototxt: {e}")
                raise
        if not os.path.exists(model_path):
            logger.info(f"Downloading DNN caffemodel to {model_path}")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("DNN caffemodel downloaded")
            except Exception as e:
                logger.error(f"Failed to download DNN caffemodel: {e}")
                raise

    def download_cnn_face_model(self):
        import bz2
        model_path = self.config.CNN_FACE_MODEL
        if os.path.exists(model_path):
            return
        url = self.config.CNN_FACE_MODEL_URL
        bz2_path = model_path + ".bz2"
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        logger.info(f"Downloading CNN face model from {url}")
        try:
            urllib.request.urlretrieve(url, bz2_path)
            with open(model_path, 'wb') as new_file, bz2.BZ2File(bz2_path, 'rb') as f:
                new_file.write(f.read())
            os.remove(bz2_path)
            logger.info(f"CNN face model extracted to {model_path}")
        except Exception as e:
            logger.error(f"CNN face model download failed: {e}")
            raise

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
