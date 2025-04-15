# src/core/model_manager.py
import urllib.request
import bz2
import os
import dlib
import logging
from src.configs.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ModelManager:
    def __init__(self):
        """
        Initialize the ModelManager with configuration and model references.
        """
        self.config = Config()
        self._detector = None
        self._predictor = None

    def download_model(self):
        """
        Download and decompress the facial landmark model if not already present.
        Returns the path to the model file.
        """
        model_file = self.config.MODEL_FILE
        model_bz2 = model_file + ".bz2"

        # Ensure the data directory exists
        os.makedirs(self.config.DATA_DIR, exist_ok=True)

        if not os.path.exists(model_file):
            logging.info(f"Model not found at {model_file}. Downloading...")
            try:
                # Download the compressed model
                urllib.request.urlretrieve(self.config.MODEL_URL, model_bz2)
                logging.info("Model downloaded successfully")

                # Decompress the model
                with open(model_file, 'wb') as new_file, bz2.BZ2File(model_bz2, 'rb') as file:
                    for data in iter(lambda: file.read(100 * 1024), b''):
                        new_file.write(data)
                logging.info(f"Model decompressed to {model_file}")

                # Remove the compressed file
                os.remove(model_bz2)
                logging.info("Cleaned up compressed file")
            except Exception as e:
                logging.error(f"Failed to download or decompress model: {e}")
                raise
        else:
            logging.info(f"Model already exists at {model_file}")

        return model_file

    @property
    def detector(self):
        """
        Lazily initialize and return the dlib face detector.
        """
        if self._detector is None:
            logging.info("Initializing face detector")
            self._detector = dlib.get_frontal_face_detector()
        return self._detector

    @property
    def predictor(self):
        """
        Lazily initialize and return the dlib shape predictor.
        """
        if self._predictor is None:
            logging.info("Initializing shape predictor")
            try:
                self._predictor = dlib.shape_predictor(self.download_model())
            except Exception as e:
                logging.error(f"Failed to initialize shape predictor: {e}")
                raise
        return self._predictor