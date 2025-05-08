import logging
from src.ui.app import DrowsinessDetectorApp

def setup_logging():
    # Cấu hình logging để ghi log ra console và file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('drowsiness_detector.log')
        ]
    )

if __name__ == '__main__':
    # Chạy ứng dụng
    setup_logging()
    try:
        app = DrowsinessDetectorApp()
        app.run()
    except Exception as e:
        logging.error(f"Lỗi xảy ra trong ứng dụng: {e}")