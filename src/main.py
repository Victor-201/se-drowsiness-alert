from src.ui.app import DrowsinessDetectorApp
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

if __name__ == '__main__':
    setup_logging()
    app = DrowsinessDetectorApp()
    app.run()