class DrowsinessDetectorException(Exception):
    """Base exception class for Drowsiness Detector application"""
    pass

class CameraError(DrowsinessDetectorException):
    """Raised when there are issues with camera operations"""
    pass

class ModelError(DrowsinessDetectorException):
    """Raised when there are issues with ML model operations"""
    pass
