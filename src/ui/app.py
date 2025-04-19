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
import logging

class DrowsinessDetectorApp(App):
    def __init__(self):
        # Khởi tạo ứng dụng Kivy
        super().__init__()
        self.config = Config()
        self.detector = DrowsinessDetector()
        self.image = Image(size_hint=(1, 0.8))  # Widget hiển thị khung hình
        self.status_label = Label(text='Status: Stopped', size_hint=(1, 0.1))  # Nhãn trạng thái
        self.camera_initialized = False
        
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
        # Xây dựng giao diện người dùng
        layout = BoxLayout(orientation='vertical')
        controls = BoxLayout(size_hint=(1, 0.1), spacing=10)

        # Thêm các nút điều khiển
        controls.add_widget(Button(text='Start', on_press=self.start_monitoring))
        controls.add_widget(Button(text='Stop', on_press=self.stop_monitoring))
        controls.add_widget(Button(text='Calibrate', on_press=self.calibrate))

        layout.add_widget(self.status_label)
        layout.add_widget(self.image)
        layout.add_widget(controls)

        # Thử khởi tạo camera
        try:
            self.detector.start_camera()
            self.camera_initialized = True
            logging.info("Khởi tạo camera thành công")
        except Exception as e:
            logging.error(f"Khởi tạo camera thất bại: {e}")
            self.status_label.text = 'Lỗi: Không khởi tạo được camera'

        # Lên lịch cập nhật khung hình 30 FPS
        Clock.schedule_interval(self._update_wrapper, 1.0 / 30.0)
        return layout

    def _update_wrapper(self, dt):
        # Đảm bảo chỉ gọi update khi đang giám sát và camera đã sẵn sàng
        if self.is_monitoring and self.camera_initialized:
            self.update()

    def start_monitoring(self, instance):
        # Bắt đầu giám sát trạng thái buồn ngủ
        if not self.camera_initialized:
            self.status_label.text = 'Lỗi: Camera chưa khởi tạo'
            logging.error("Không thể bắt đầu giám sát: Camera chưa khởi tạo")
            return
        self.is_monitoring = True
        self.status_label.text = 'Trạng thái: Đang giám sát'
        logging.info("Bắt đầu giám sát")

    def stop_monitoring(self, instance):
        # Dừng giám sát và reset giao diện
        self.is_monitoring = False
        self.alert_active = False
        self.status_label.text = 'Trạng thái: Đã dừng'
        self.image.texture = None
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None
            self.alert_sound.stop()
        logging.info("Dừng giám sát")

    def calibrate(self, instance):
        # Thực hiện hiệu chỉnh EAR
        if not self.camera_initialized:
            self.status_label.text = 'Lỗi: Camera chưa khởi tạo'
            logging.error("Không thể hiệu chỉnh: Camera chưa khởi tạo")
            return
        self.status_label.text = 'Trạng thái: Đang hiệu chỉnh...'
        logging.info("Bắt đầu hiệu chỉnh")
        try:
            success = self.detector.calibrate(duration=5)
            self.status_label.text = 'Trạng thái: Hiệu chỉnh hoàn tất' if success else 'Trạng thái: Hiệu chỉnh thất bại'
            logging.info("Hiệu chỉnh hoàn tất" if success else "Hiệu chỉnh thất bại")
        except Exception as e:
            self.status_label.text = 'Trạng thái: Lỗi hiệu chỉnh'
            logging.error(f"Lỗi hiệu chỉnh: {e}")

    def update(self):
        # Cập nhật giao diện với khung hình mới nhất
        try:
            frame, drowsiness_detected = self.detector.process_frame()
            if frame is not None:
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt='bgr', bufferfmt='ubyte')
                self.image.texture = texture

                if drowsiness_detected and not self.alert_active:
                    self.status_label.text = 'CẢNH BÁO: Phát hiện buồn ngủ!'
                    self.alert_active = True
                    logging.info("Phát hiện buồn ngủ")
                    self.start_alert()
                elif not drowsiness_detected and self.alert_active and not self.alert_stop_timer:
                    self.alert_stop_timer = Clock.schedule_once(self.stop_alert, self.alert_stop_delay)  # Dừng cảnh báo sau 4 giây
                elif not self.alert_active:
                    self.status_label.text = 'Trạng thái: Đang giám sát'
        except Exception as e:
            logging.error(f"Lỗi xử lý khung hình: {e}")
            self.status_label.text = 'Lỗi: Xử lý khung hình thất bại'


    def start_alert(self):
        logging.info("Bắt đầu phát âm thanh cảnh báo")
        self.alert_active = True
        self.alert_sound.stop()
        self.alert_sound.loop = True
        self.alert_sound.play()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None

    def stop_alert(self, dt):
        # Dừng cảnh báo và reset trạng thái
        self.alert_active = False
        if self.alert_sound:
            self.alert_sound.stop()
        self.alert_stop_timer = None
        self.status_label.text = 'Trạng thái: Đang giám sát'
        logging.info("Dừng cảnh báo")
        

    def on_stop(self):
        # Dọn dẹp tài nguyên khi đóng ứng dụng
        logging.info("Dọn dẹp tài nguyên")
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
        if self.alert_sound:
            self.alert_sound.stop()
        self.detector.stop_camera()