import os
import cv2
import logging
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.core.audio import SoundLoader
from kivy.uix.screenmanager import ScreenManager
from src.core.detector import DrowsinessDetector
from src.configs.config import Config
from src.configs.settings import Settings
from src.ui.screens.main_screen import MainScreen
from src.ui.screens.settings_screen import SettingsScreen

logger = logging.getLogger(__name__)

class DrowsinessDetectorApp(App):
    def __init__(self):
        super().__init__()
        # Khởi tạo các thành phần của ứng dụng
        self.config = Config()
        self.detector = DrowsinessDetector()
        self.image = Image(size_hint=(1, 1))
        self.status_label = Label(text='Trạng thái: Đã dừng', size_hint=(1, 0.1))
        self.settings = Settings()
        self.sound_alert_dir = self.config.SOUND_ALERT_DIR
        self.image_dir = self.config.IMAGE_DIR
        self.alert_sound_file = self.config.ALERT_SOUND_FILE
        self.fatigue_sound_file = self.config.FATIGUE_SOUND_FILE
        self.ear_threshold = self.config.EAR_THRESHOLD
        self.camera_width = self.config.CAMERA_WIDTH
        self.camera_height = self.config.CAMERA_HEIGHT
        self.camera_fps = self.config.CAMERA_FPS
        self.initialize_app()
        self.last_metrics = {
            'ear': None,
            'mar': None,
            'roll_angle': None,
            'pitch_angle': None,
            'blink_count': 0,
            'yawn_count': 0
        }

    def initialize_app(self):
        # Khởi tạo trạng thái ban đầu của ứng dụng
        self.camera_initialized = False
        self.alert_sound = None
        self.fatigue_sound = None
        self.setup_alert_sound()
        self.setup_fatigue_sound()
        self.setup_state_variables()
        logger.info(f"Camera khả dụng: {self.settings.get_available_cameras()}")

    def setup_alert_sound(self):
        # Thiết lập âm thanh cảnh báo
        default_sound = os.path.join(self.sound_alert_dir, "alert.wav")
        sound_path = (
            os.path.join(self.sound_alert_dir, self.settings.alert_sound_file)
            if self.settings.alert_sound_file
            else default_sound
        )
        if os.path.exists(sound_path):
            self.alert_sound = SoundLoader.load(sound_path)
            if not self.alert_sound:
                logging.warning(f"Không thể tải file âm thanh: {sound_path}")
        else:
            logging.warning(f"File âm thanh không tồn tại: {sound_path}")

    def setup_fatigue_sound(self):
        # Thiết lập âm thanh thông báo mệt mỏi
        sound_path = self.fatigue_sound_file
        if os.path.exists(sound_path):
            self.fatigue_sound = SoundLoader.load(sound_path)
            if not self.fatigue_sound:
                logging.warning(f"Không thể tải file âm thanh thông báo: {sound_path}")
        else:
            logging.warning(f"File âm thanh thông báo không tồn tại: {sound_path}")

    def setup_state_variables(self):
        # Thiết lập các biến trạng thái
        self.is_monitoring = False
        self.alert_active = False
        self.alert_stop_timer = None
        self.alert_stop_delay = self.config.ALERT_STOP_DELAY
        self.calibration_event = None
        self.calibration_start_time = None
        self.background_color = [0, 0, 0, 1]
        self.screen_manager = None

    def build(self):
        # Xây dựng giao diện ứng dụng
        self.screen_manager = ScreenManager()
        main_screen = MainScreen(app_instance=self, name='main')
        self.screen_manager.add_widget(main_screen)
        settings_screen = SettingsScreen(name='settings', app_instance=self)
        self.screen_manager.add_widget(settings_screen)
        try:
            self.detector.start_camera()
            self.camera_initialized = True
            logging.info("Khởi tạo camera thành công")
        except Exception as e:
            logging.error(f"Khởi tạo camera thất bại: {e}")
            self.status_label.text = 'Lỗi: Không khởi tạo được camera'
        Clock.schedule_interval(self._update_wrapper, 1.0 / 30.0)
        return self.screen_manager

    def switch_to_settings(self, instance):
        # Chuyển sang màn hình cài đặt
        self.stop_monitoring(instance)
        self.screen_manager.current = 'settings'
        logging.info("Chuyển sang màn hình cài đặt")
        main_screen = self.screen_manager.get_screen('main')
        main_screen.update_metrics(
            self.last_metrics['ear'],
            self.last_metrics['mar'],
            self.last_metrics['roll_angle'],
            self.last_metrics['pitch_angle'],
            self.last_metrics['blink_count'],
            self.last_metrics['yawn_count']
        )

    def switch_to_main(self):
        # Chuyển về màn hình chính
        self.screen_manager.current = 'main'
        logging.info("Chuyển về màn hình chính")
        main_screen = self.screen_manager.get_screen('main')
        main_screen.update_metrics(
            self.last_metrics['ear'],
            self.last_metrics['mar'],
            self.last_metrics['roll_angle'],
            self.last_metrics['pitch_angle'],
            self.last_metrics['blink_count'],
            self.last_metrics['yawn_count']
        )

    def update_background_color(self):
        # Cập nhật màu nền của màn hình
        if self.screen_manager.current == 'main':
            main_screen = self.screen_manager.get_screen('main')
            with main_screen.canvas.before:
                Color(*self.background_color)
                main_screen.background_rect = Rectangle(pos=main_screen.pos, size=main_screen.size)

    def exit_app(self, instance):
        # Thoát ứng dụng
        logging.info("Thoát ứng dụng")
        self.stop_monitoring(instance)
        App.get_running_app().stop()

    def _update_wrapper(self, dt):
        # Cập nhật trạng thái ứng dụng theo chu kỳ
        if self.is_monitoring and self.camera_initialized:
            self.update()
        elif self.calibration_event and self.camera_initialized:
            self.update_calibration(dt)
        else:
            main_screen = self.screen_manager.get_screen('main')
            main_screen.update_metrics(
                self.last_metrics['ear'],
                self.last_metrics['mar'],
                self.last_metrics['roll_angle'],
                self.last_metrics['pitch_angle'],
                self.last_metrics['blink_count'],
                self.last_metrics['yawn_count']
            )

    def start_monitoring(self, instance):
        # Bắt đầu giám sát
        if not self.camera_initialized:
            self.status_label.text = 'Lỗi: Camera chưa khởi tạo'
            logging.error("Không thể bắt đầu giám sát: Camera chưa khởi tạo")
            return
        self.is_monitoring = True
        self.status_label.text = 'Trạng thái: Đang giám sát'
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        logging.info("Bắt đầu giám sát")

    def stop_monitoring(self, instance):
        # Dừng giám sát
        self.is_monitoring = False
        self.alert_active = False
        self.status_label.text = 'Trạng thái: Đã dừng'
        self.image.texture = None
        if self.alert_sound:
            self.alert_sound.stop()
        if self.fatigue_sound:
            self.fatigue_sound.stop()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None
        if self.calibration_event:
            self.calibration_event.cancel()
            self.calibration_event = None
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        main_screen = self.screen_manager.get_screen('main')
        main_screen.update_metrics(
            self.last_metrics['ear'],
            self.last_metrics['mar'],
            self.last_metrics['roll_angle'],
            self.last_metrics['pitch_angle'],
            self.last_metrics['blink_count'],
            self.last_metrics['yawn_count']
        )
        logging.info("Dừng giám sát")

    def calibrate(self, instance):
        # Hiệu chỉnh ứng dụng
        if not self.camera_initialized:
            self.status_label.text = 'Lỗi: Camera chưa khởi tạo'
            logging.error("Không thể hiệu chỉnh: Camera chưa khởi tạo")
            return
        self.stop_monitoring(instance)
        self.status_label.text = 'Trạng thái: Đang hiệu chỉnh...'
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        logging.info("Bắt đầu hiệu chỉnh")
        self.detector.reset_calibration()
        self.calibration_start_time = Clock.get_time()
        self.calibration_event = Clock.schedule_interval(self.update_calibration, 1.0 / 30.0)

    def update_calibration(self, dt):
        # Cập nhật quá trình hiệu chỉnh
        duration = 5
        elapsed = Clock.get_time() - self.calibration_start_time
        if elapsed >= duration:
            success, new_threshold = self.detector.finalize_calibration()
            self.calibration_event.cancel()
            self.calibration_event = None
            self.status_label.text = f'Trạng thái: Hiệu chỉnh hoàn tất ( Ngưỡng mắt mới: {new_threshold:.3f})' if success else 'Trạng thái: Hiệu chỉnh thất bại'
            logging.info(
                f"Hiệu chỉnh {'hoàn tất' if success else 'thất bại'}. Ngưỡng mắt mới: {new_threshold:.3f}" if success else "Hiệu chỉnh thất bại")
            self.image.texture = None
            self.background_color = [0, 0, 0, 1]
            self.update_background_color()
            main_screen = self.screen_manager.get_screen('main')
            main_screen.update_metrics(
                self.last_metrics['ear'],
                self.last_metrics['mar'],
                self.last_metrics['roll_angle'],
                self.last_metrics['pitch_angle'],
                self.last_metrics['blink_count'],
                self.last_metrics['yawn_count']
            )
            return
        frame, ear = self.detector.process_calibration_frame()
        if frame is not None:
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
            remaining = int(duration - elapsed)
            self.status_label.text = f'Đang hiệu chỉnh... {remaining}s (Ngưỡng mắt: {ear:.2f})'
            main_screen = self.screen_manager.get_screen('main')
            main_screen.update_metrics(
                ear if ear != 0.0 else self.last_metrics['ear'],
                self.last_metrics['mar'],
                self.last_metrics['roll_angle'],
                self.last_metrics['pitch_angle'],
                self.last_metrics['blink_count'],
                self.last_metrics['yawn_count']
            )

    def update(self):
        # Cập nhật trạng thái giám sát
        try:
            frame, alert_detected = self.detector.process_frame()
            ear = self.last_metrics['ear']
            mar = self.last_metrics['mar']
            roll_angle = self.last_metrics['roll_angle']
            pitch_angle = self.last_metrics['pitch_angle']
            blink_count = self.detector.blink_total
            yawn_count = self.detector.yawn_total

            if frame is None:
                logger.warning("Received None frame from process_frame")
                self.status_label.text = 'Lỗi: Không lấy được khung hình'
                main_screen = self.screen_manager.get_screen('main')
                main_screen.update_metrics(
                    self.last_metrics['ear'],
                    self.last_metrics['mar'],
                    self.last_metrics['roll_angle'],
                    self.last_metrics['pitch_angle'],
                    self.last_metrics['blink_count'],
                    self.last_metrics['yawn_count']
                )
                return

            if not isinstance(frame, np.ndarray):
                logger.error(f"Frame is not a NumPy array, got type {type(frame)}: {frame}")
                self.status_label.text = 'Lỗi: Khung hình không hợp lệ'
                main_screen = self.screen_manager.get_screen('main')
                main_screen.update_metrics(
                    self.last_metrics['ear'],
                    self.last_metrics['mar'],
                    self.last_metrics['roll_angle'],
                    self.last_metrics['pitch_angle'],
                    self.last_metrics['blink_count'],
                    self.last_metrics['yawn_count']
                )
                return

            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

            if self.detector.face_detected:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                largest_face = self.detector.find_largest_face(self.detector.face_detector(gray))
                if largest_face:
                    shape = self.detector.landmark_predictor(gray, largest_face)
                    shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                    left_eye, right_eye = shape_np[36:42], shape_np[42:48]
                    mouth = shape_np[48:68]
                    ear = float((self.detector.analyzer.calculate_ear(
                        left_eye) + self.detector.analyzer.calculate_ear(right_eye)) / 2.0)
                    mar = float(self.detector.analyzer.calculate_mar(mouth))
                    roll_angle, pitch_angle = self.detector.analyzer.calculate_head_pose(shape_np)
                    roll_angle = float(roll_angle)
                    pitch_angle = float(pitch_angle)
                    self.last_metrics['ear'] = ear
                    self.last_metrics['mar'] = mar
                    self.last_metrics['roll_angle'] = roll_angle
                    self.last_metrics['pitch_angle'] = pitch_angle
                    self.last_metrics['blink_count'] = blink_count
                    self.last_metrics['yawn_count'] = yawn_count

            main_screen = self.screen_manager.get_screen('main')
            main_screen.update_metrics(ear, mar, roll_angle, pitch_angle, blink_count, yawn_count)

            if alert_detected and not self.alert_active:
                if self.detector.check_blink_frequency() or self.detector.check_yawn_frequency():
                    status_text = 'CẢNH BÁO: Dấu hiệu mệt mỏi!'
                    self.status_label.text = status_text
                    self.alert_active = True
                    self.background_color = [1, 0, 0, 1]
                    self.update_background_color()
                    logging.info(status_text)
                    self.start_fatigue_alert()
                elif self.detector.eye_counter >= self.detector.ear_consec_frames:
                    status_text = 'CẢNH BÁO: Phát hiện buồn ngủ!'
                    self.status_label.text = status_text
                    self.alert_active = True
                    self.background_color = [1, 0, 0, 1]
                    self.update_background_color()
                    logging.info(status_text)
                    self.start_alert()
                elif self.detector.head_tilt_counter >= self.detector.head_tilt_frames:
                    status_text = 'CẢNH BÁO: Tư thế đầu bất thường!'
                    self.status_label.text = status_text
                    self.alert_active = True
                    self.background_color = [1, 0, 0, 1]
                    self.update_background_color()
                    logging.info(status_text)
                    self.start_alert()

            elif not alert_detected and self.alert_active and not self.alert_stop_timer:
                self.alert_stop_timer = Clock.schedule_once(self.stop_alert, self.alert_stop_delay)
            elif not self.alert_active:
                self.status_label.text = 'Trạng thái: Đang giám sát'
                self.background_color = [0, 0, 0, 1]
                self.update_background_color()

        except Exception as e:
            logger.error(f"Lỗi xử lý khung hình: {e}")
            self.status_label.text = 'Lỗi: Xử lý khung hình thất bại'
            self.background_color = [0, 0, 0, 1]
            self.update_background_color()
            main_screen = self.screen_manager.get_screen('main')
            main_screen.update_metrics(
                self.last_metrics['ear'],
                self.last_metrics['mar'],
                self.last_metrics['roll_angle'],
                self.last_metrics['pitch_angle'],
                self.last_metrics['blink_count'],
                self.last_metrics['yawn_count']
            )

    def start_alert(self):
        # Bắt đầu phát âm thanh cảnh báo
        logging.info("Bắt đầu phát âm thanh cảnh báo")
        self.alert_active = True
        if self.fatigue_sound and self.fatigue_sound.state == 'play':
            self.fatigue_sound.stop() 
            logging.info("Dừng âm thanh mệt mỏi để ưu tiên âm thanh cảnh báo")
        if self.alert_sound:
            self.alert_sound.stop()
            self.alert_sound.loop = True
            self.alert_sound.volume = self.settings.alert_volume / 100.0
            self.alert_sound.play()
        self.background_color = [1, 0, 0, 1]
        self.update_background_color()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None

    def start_fatigue_alert(self):
        # Bắt đầu phát âm thanh thông báo mệt mỏi
        logging.info("Bắt đầu phát âm thanh thông báo mệt mỏi")
        if self.alert_sound and self.alert_sound.state == 'play':
            logging.info("Ưu tiên âm thanh cảnh báo alert, bỏ qua âm thanh mệt mỏi")
            return  
        self.alert_active = True
        if self.fatigue_sound:
            self.fatigue_sound.stop()
            self.fatigue_sound.loop = False
            self.fatigue_sound.volume = self.settings.alert_volume / 100.0
            self.fatigue_sound.play()
        self.background_color = [1, 0, 0, 1]
        self.update_background_color()
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None

    def stop_alert(self, dt):
        # Dừng âm thanh cảnh báo
        self.alert_active = False
        if self.alert_sound:
            self.alert_sound.stop()
        if self.fatigue_sound:
            self.fatigue_sound.stop()
        self.alert_stop_timer = None
        self.status_label.text = 'Trạng thái: Đang giám sát'
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        main_screen = self.screen_manager.get_screen('main')
        main_screen.update_metrics(
            self.last_metrics['ear'],
            self.last_metrics['mar'],
            self.last_metrics['roll_angle'],
            self.last_metrics['pitch_angle'],
            self.last_metrics['blink_count'],
            self.last_metrics['yawn_count']
        )

    def on_stop(self):
        # Dọn dẹp tài nguyên khi ứng dụng dừng
        logging.info("Dọn dẹp tài nguyên")
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
        if self.alert_sound:
            self.alert_sound.stop()
        if self.fatigue_sound:
            self.fatigue_sound.stop()
        if self.calibration_event:
            self.calibration_event.cancel()
        self.detector.stop_camera()
        self.background_color = [0, 0, 0, 1]
        self.update_background_color()
        main_screen = self.screen_manager.get_screen('main')
        main_screen.update_metrics(
            self.last_metrics['ear'],
            self.last_metrics['mar'],
            self.last_metrics['roll_angle'],
            self.last_metrics['pitch_angle'],
            self.last_metrics['blink_count'],
            self.last_metrics['yawn_count']
        )