import os
import logging
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.core.audio import SoundLoader
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
import cv2

class IconButton(Button):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        # Khởi tạo nút với biểu tượng
        self.source = source
        with self.canvas:
            self.icon = Image(source=self.source, allow_stretch=True, keep_ratio=True)
        self.bind(pos=self.update_icon, size=self.update_icon)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)

    def update_icon(self, *args):
        # Cập nhật vị trí và kích thước biểu tượng
        self.icon.pos = self.pos
        self.icon.size = self.size

class SettingsScreen(Screen):
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        # Khởi tạo màn hình cài đặt
        self.app = app_instance
        self.build()

    def build(self):
        # Xây dựng giao diện màn hình cài đặt
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # Cài đặt nền
        with layout.canvas.before:
            Color(0.1, 0.1, 0.1, 1)
            self.background_rect = Rectangle(pos=layout.pos, size=layout.size)
        layout.bind(pos=self.update_background_rect, size=self.update_background_rect)

        # Tạo tiêu đề và nút quay lại
        header_layout = FloatLayout(size_hint=(1, 0.12))

        back_button = IconButton(
            source=f'{self.app.image_dir}/goback_button.png',
            size_hint=(None, None),
            size=(42, 42),
            pos_hint={'x': 0, 'top': 1},
            on_press=lambda instance: self.app.switch_to_main()
        )
        header_layout.add_widget(back_button)

        title_label = Label(
            text='CÀI ĐẶT HỆ THỐNG',
            size_hint=(None, None),
            size=(400, 64),
            pos_hint={'center_x': 0.5, 'top': 1},
            font_size='24sp',
            halign='center',
            valign='middle',
            color=(1, 1, 1, 1)
        )
        header_layout.add_widget(title_label)

        layout.add_widget(header_layout)

        # Tùy chọn camera
        camera_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10)
        camera_layout.add_widget(Label(text='Chọn Camera:', size_hint=(0.4, 1), font_size='18sp', color=(1, 1, 1, 1)))

        self.camera_spinner = Spinner(
            text=f'Camera {self.app.settings.camera_index}',
            values=[f'Camera {cam}' for cam in self.app.settings.get_available_cameras()],
            size_hint=(0.6, 1),
            background_color=(0.2, 0.6, 1, 0.8),
            color=(1, 1, 1, 1),
            font_size='16sp'
        )
        camera_layout.add_widget(self.camera_spinner)
        layout.add_widget(camera_layout)

        # Điều chỉnh âm lượng
        volume_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10)
        volume_layout.add_widget(Label(text='Âm lượng cảnh báo:', size_hint=(0.4, 1), font_size='18sp', color=(1, 1, 1, 1)))

        self.volume_slider = Slider(
            min=0, max=100, value=self.app.settings.alert_volume,
            size_hint=(0.6, 1), cursor_size=(42, 42),
        )
        volume_layout.add_widget(self.volume_slider)
        layout.add_widget(volume_layout)

        # Tùy chọn âm thanh
        sound_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10)
        sound_layout.add_widget(Label(text='Âm thanh cảnh báo:', size_hint=(0.4, 1), font_size='18sp', color=(1, 1, 1, 1)))

        self.sound_spinner = Spinner(
            text=self.app.settings.alert_sound_file or 'Mặc định',
            values=self.app.settings.get_available_sounds(),
            size_hint=(0.4, 1),
            background_color=(0.2, 0.6, 1, 0.8),
            color=(1, 1, 1, 1),
            font_size='16sp'
        )
        sound_layout.add_widget(self.sound_spinner)

        preview_button = IconButton(
            source=f'{self.app.image_dir}/preview_sound_button.png',
            size_hint=(None, None),
            size=(40, 40),
            on_press=self.preview_sound
        )
        sound_layout.add_widget(preview_button)
        layout.add_widget(sound_layout)

        # Thêm khoảng trống
        layout.add_widget(Label(size_hint=(1, 0.05)))

        # Nút lưu cài đặt
        save_button = Button(
            text='LƯU CÀI ĐẶT',
            size_hint=(1, 0.15),
            background_color=(0.2, 0.6, 1, 0.8),
            font_size='22sp',
            on_press=self.save_settings
        )
        layout.add_widget(save_button)

        self.add_widget(layout)

    def update_background_rect(self, instance, value):
        # Cập nhật hình chữ nhật nền
        self.background_rect.pos = instance.pos
        self.background_rect.size = instance.size

    def preview_sound(self, instance):
        # Phát thử âm thanh được chọn
        selected_sound = self.sound_spinner.text
        sound_path = (self.app.alert_sound_file if selected_sound == 'Mặc định'
                      else os.path.join(self.app.sound_alert_dir, selected_sound))

        if os.path.exists(sound_path):
            preview = SoundLoader.load(sound_path)
            if preview:
                preview.volume = self.volume_slider.value / 100.0
                preview.play()
                Clock.schedule_once(lambda dt: preview.stop(), 3)
                logging.info(f"Phát thử âm thanh: {sound_path}")
            else:
                logging.warning(f"Không thể tải âm thanh: {sound_path}")
                self.app.status_label.text = "Lỗi: Không thể phát âm thanh"
        else:
            logging.warning(f"File âm thanh không tồn tại: {sound_path}")
            self.app.status_label.text = "Lỗi: File âm thanh không tồn tại"

    def save_settings(self, instance):
        # Lưu cài đặt
        try:
            selected_camera = self.camera_spinner.text
            new_camera_index = int(selected_camera.replace('Camera ', ''))

            # Kiểm tra camera mới
            test_cap = cv2.VideoCapture(new_camera_index)
            if not test_cap.isOpened():
                raise ValueError(f"Không thể truy cập Camera {new_camera_index}")
            test_cap.release()

            camera_changed = new_camera_index != self.app.settings.camera_index
            self.app.settings.camera_index = new_camera_index
            self.app.settings.alert_volume = int(self.volume_slider.value)

            selected_sound = self.sound_spinner.text
            self.app.settings.alert_sound_file = None if selected_sound == 'Mặc định' else selected_sound

            self.app.settings.save()
            logging.info(f"Đã lưu: Camera {new_camera_index}, Âm lượng {self.app.settings.alert_volume}%, Âm thanh {selected_sound}")

            # Cập nhật camera nếu thay đổi
            if camera_changed and self.app.camera_initialized:
                self.app.detector.stop_camera()
                self.app.config.CAMERA_ID = new_camera_index
                self.app.detector.start_camera()
                logging.info(f"Chuyển camera: {new_camera_index}")

            self._update_alert_sound()
            self.app.switch_to_main()
            self.app.status_label.text = "Cài đặt đã lưu thành công"

        except ValueError as e:
            logging.error(f"Lỗi camera: {e}")
            self.app.status_label.text = f"Lỗi: {str(e)}"
        except Exception as e:
            logging.error(f"Lỗi khi lưu: {e}")
            self.app.status_label.text = "Lỗi: Không thể lưu cài đặt"

    def _update_alert_sound(self):
        # Cập nhật âm thanh cảnh báo
        selected_sound = self.sound_spinner.text
        sound_path = (self.app.alert_sound_file if selected_sound == 'Mặc định'
                      else os.path.join(self.app.sound_alert_dir, selected_sound))

        if os.path.exists(sound_path):
            self.app.alert_sound = SoundLoader.load(sound_path)
            if self.app.alert_sound:
                self.app.alert_sound.volume = self.app.settings.alert_volume / 100.0
                logging.info(f"Đặt âm thanh cảnh báo: {sound_path}")
            else:
                logging.warning(f"Không thể load âm thanh: {sound_path}")
        else:
            logging.warning(f"File âm thanh không tồn tại: {sound_path}")
