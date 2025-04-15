import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader
import cv2
from src.core.detector import DrowsinessDetector
from src.configs.config import Config
import os

class DrowsinessDetectorApp(App):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.detector = DrowsinessDetector()
        self.image = Image()
        self.status_label = Label(text='Status: Stopped', size_hint=(1, 0.1))
        
        # Khởi tạo âm thanh cảnh báo
        self.alert_sound = None
        sound_path = self.config.ALERT_SOUND_FILE
        if os.path.exists(sound_path):
            self.alert_sound = SoundLoader.load(sound_path)
            if not self.alert_sound:
                logging.warning(f"Không thể tải file âm thanh: {sound_path}")
        else:
            logging.warning(f"File âm thanh không tồn tại: {sound_path}")
        
        self.is_monitoring = False
        self.alert_active = False
        self.alert_stop_timer = None
        self.alert_stop_delay = self.config.ALERT_STOP_DELAY

    def build(self):
        layout = BoxLayout(orientation='vertical')
        controls = BoxLayout(size_hint=(1, 0.1), spacing=10)
        controls.add_widget(Button(text='Start', on_press=self.start_monitoring))
        controls.add_widget(Button(text='Stop', on_press=self.stop_monitoring))
        
        layout.add_widget(self.status_label)
        layout.add_widget(self.image)
        layout.add_widget(controls)
        
        self.detector.start_camera()
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return layout

    def start_monitoring(self):
        self.is_monitoring = True
        self.status_label.text = 'Status: Monitoring'

    def stop_monitoring(self):
        self.is_monitoring = False
        self.status_label.text = 'Status: Stopped'
        self.image.texture = None
        self.alert_sound.stop()

    def update(self):
        if not self.is_monitoring:
            return
        
        frame, drowsiness_detected = self.detector.process_frame()
        if frame is not None:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
            
            if drowsiness_detected and not self.alert_active:
                self.status_label.text = 'ALERT: Drowsiness Detected!'
                self.start_alert()
            elif not drowsiness_detected and self.alert_active and not self.alert_stop_timer:
                self.alert_stop_timer = Clock.schedule_once(self.stop_alert, self.alert_stop_delay)
            elif not self.alert_active:
                self.status_label.text = 'Status: Monitoring'

    def start_alert(self):
        if self.alert_active or not self.alert_sound:
            return
        
        logging.info("Bắt đầu phát âm thanh cảnh báo")
        self.alert_active = True
        self.alert_sound.stop()
        self.alert_sound.loop = True
        self.alert_sound.play()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None

    def stop_alert(self):
        logging.info("Dừng âm thanh cảnh báo")
        self.alert_active = False
        if self.alert_sound:
            self.alert_sound.stop()
        self.alert_stop_timer = None
        self.status_label.text = 'Status: Monitoring'

    def on_stop(self):
        logging.info("Dọn dẹp tài nguyên")
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
        if self.alert_sound:
            self.alert_sound.stop()
        self.detector.stop_camera()