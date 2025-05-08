import numpy as np
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, PushMatrix, Rotate, PopMatrix
from kivy.properties import NumericProperty, ListProperty, BooleanProperty
from kivy.uix.image import Image
from kivy.clock import Clock
import time

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

class StatusBar(Widget):
    value = NumericProperty(0.0)
    max_value = NumericProperty(1.0)
    threshold = NumericProperty(0.0)
    angle = NumericProperty(0.0)
    reverse_threshold = BooleanProperty(False)
    bar_color = ListProperty([0, 1, 0, 1])
    bar_length = NumericProperty(150)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Khởi tạo thanh trạng thái
        self.bind(value=self.update_bar, threshold=self.update_bar, angle=self.update_bar,
                  reverse_threshold=self.update_bar, size=self.update_bar, pos=self.update_bar,
                  bar_length=self.update_bar)
        self.bar_height = 20
        self.blink_state = 1.0
        self.update_bar()

    def update_bar(self, *args):
        # Cập nhật thanh trạng thái
        self.canvas.clear()
        with self.canvas:
            bar_x = self.x + (self.width - self.bar_length) / 2
            bar_y = self.y + (self.height - self.bar_height) / 2
            if self.reverse_threshold:
                is_safe = self.value >= self.threshold
            else:
                is_safe = self.value < self.threshold
            if not is_safe:
                self.blink_state = 0.5 + 0.5 * abs(np.sin(time.time() * 5))
                self.bar_color = [1, 0, 0, self.blink_state]
            else:
                self.blink_state = 1.0
                self.bar_color = [0, 1, 0, 1]
            filled_length = min(int(self.bar_length * (self.value / self.max_value)),
                                self.bar_length) if self.max_value > 0 else 0
            Color(0.2, 0.2, 0.2, 1)
            Rectangle(pos=(bar_x, bar_y), size=(self.bar_length, self.bar_height))
            Color(*self.bar_color)
            PushMatrix()
            Rotate(angle=self.angle, origin=(bar_x + self.bar_length / 2, bar_y + self.bar_height / 2))
            Rectangle(pos=(bar_x, bar_y), size=(filled_length, self.bar_height))
            PopMatrix()

class MainScreen(Screen):
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        # Khởi tạo màn hình chính
        self.app = app_instance
        self.metrics_widgets = {
            'ear': {
                'label': Label(
                    text='Cỡ mắt: --',
                    size_hint=(1, 0.1),
                    font_size='20sp',
                    halign='left',
                    valign='middle',
                    text_size=(None, None),
                    padding_x=10
                ),
                'bar': StatusBar(value=0.0, max_value=0.4, threshold=self.app.ear_threshold, reverse_threshold=True,
                                 size_hint=(1, 0.1), bar_length=150)
            },
            'mar': {
                'label': Label(
                    text='Cỡ miệng: --',
                    size_hint=(1, 0.1),
                    font_size='20sp',
                    halign='left',
                    valign='middle',
                    text_size=(None, None),
                    padding_x=10
                ),
                'bar': StatusBar(value=0.0, max_value=1.0, threshold=0.5, size_hint=(1, 0.1), bar_length=150)
            },
            'roll_angle': {
                'label': Label(
                    text='Góc nghiêng: --',
                    size_hint=(1, 0.1),
                    font_size='20sp',
                    halign='left',
                    valign='middle',
                    text_size=(None, None),
                    padding_x=10
                ),
                'bar': StatusBar(value=0.0, max_value=45.0, threshold=15.0, size_hint=(1, 0.1), bar_length=150)
            },
            'pitch_angle': {
                'label': Label(
                    text='Góc cúi: --',
                    size_hint=(1, 0.1),
                    font_size='20sp',
                    halign='left',
                    valign='middle',
                    text_size=(None, None),
                    padding_x=10
                ),
                'bar': StatusBar(value=0.0, max_value=45.0, threshold=15.0, size_hint=(1, 0.1), bar_length=150)
            },
            'blink_count': {
                'label': Label(
                    text='Nháy mắt: --',
                    size_hint=(1, 0.1),
                    font_size='20sp',
                    halign='left',
                    valign='middle',
                    text_size=(None, None),
                    padding_x=10
                ),
                'bar': StatusBar(value=0.0, max_value=50.0, threshold=30.0, size_hint=(1, 0.1), bar_length=70)
            },
            'yawn_count': {
                'label': Label(
                    text='Ngáp: --',
                    size_hint=(1, 0.1),
                    font_size='20sp',
                    halign='left',
                    valign='middle',
                    text_size=(None, None),
                    padding_x=10
                ),
                'bar': StatusBar(value=0.0, max_value=50.0, threshold=30.0, size_hint=(1, 0.1), bar_length=70)
            }
        }
        self.scanning = False
        self.scan_y = 0
        self.scan_direction = -1 
        self.is_calibrating = False
        self.build()

    def build(self):
        # Xây dựng giao diện màn hình chính
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        with main_layout.canvas.before:
            Color(*self.app.background_color)
            self.background_rect = Rectangle(pos=main_layout.pos, size=main_layout.size)
        main_layout.bind(pos=self.update_background_rect, size=self.update_background_rect)
        
        # Header với các nút
        header = FloatLayout(size_hint=(1, 0.1))

        # Nút Thoát ở rìa trái
        exit_button = Button(
            text='Thoát',
            size_hint=(None, None),
            size=(100, 42),
            pos_hint={'x': 0, 'top': 1},
            background_color=(1, 0, 0, 1),
            on_press=self.app.exit_app
        )
        header.add_widget(exit_button)

        # Nhóm 3 nút ở giữa
        middle_buttons_layout = BoxLayout(
            size_hint=(None, None),
            size=(300, 42),
            pos_hint={'center_x': 0.5, 'top': 1},
            spacing=10
        )
        middle_buttons = [
            ('Bắt đầu', (0, 1, 0, 1), self.app.start_monitoring),
            ('Dừng', (1, 0.5, 0, 1), self.app.stop_monitoring),
            ('Hiệu chỉnh', (0, 0, 1, 1), self.start_calibration),
        ]
        for text, color, callback in middle_buttons:
            button = Button(
                text=text,
                size_hint=(0.33, 1),
                background_color=color,
                on_press=callback
            )
            middle_buttons_layout.add_widget(button)
        header.add_widget(middle_buttons_layout)
        
        # Nút Cài đặt với icon ở rìa phải
        settings_button = IconButton(
            source=f'{self.app.image_dir}/settings_button.png',
            size_hint=(None, None),
            size=(42, 42),
            pos_hint={'right': 1, 'top': 1},
            on_press=lambda instance: self.app.switch_to_settings(instance)
        )
        header.add_widget(settings_button)
        
        # Status label dưới header
        self.app.status_label.size_hint = (1, 0.1)
        self.app.status_label.font_size = '24sp'
        self.app.status_label.bold = True
        self.app.status_label.halign = 'center'
        self.app.status_label.valign = 'middle'
        self.app.status_label.text_size = (None, None)
        
        # Content layout
        content_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.7), spacing=10)
        metrics_layout = BoxLayout(orientation='vertical', size_hint=(0.3, 1), spacing=5)
        
        # Thêm các widget cho EAR, MAR, roll_angle, pitch_angle
        for key in ['ear', 'mar', 'roll_angle', 'pitch_angle']:
            metrics_layout.add_widget(self.metrics_widgets[key]['label'])
            metrics_layout.add_widget(self.metrics_widgets[key]['bar'])
        
        # Tạo BoxLayout ngang cho blink_count và yawn_count
        blink_yawn_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)
        
        # Thêm nhãn và thanh trạng thái cho blink_count
        blink_layout = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=5)
        blink_layout.add_widget(self.metrics_widgets['blink_count']['label'])
        blink_layout.add_widget(self.metrics_widgets['blink_count']['bar'])
        blink_yawn_layout.add_widget(blink_layout)
        
        # Thêm nhãn và thanh trạng thái cho yawn_count
        yawn_layout = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=5)
        yawn_layout.add_widget(self.metrics_widgets['yawn_count']['label'])
        yawn_layout.add_widget(self.metrics_widgets['yawn_count']['bar'])
        blink_yawn_layout.add_widget(yawn_layout)
        
        # Thêm blink_yawn_layout vào metrics_layout
        metrics_layout.add_widget(blink_yawn_layout)
        
        # Camera layout với hiệu ứng quét
        self.camera_layout = BoxLayout(size_hint=(0.7, 1))
        self.camera_layout.add_widget(self.app.image)
        content_layout.add_widget(metrics_layout)
        content_layout.add_widget(self.camera_layout)
        
        # Thêm các thành phần vào main_layout
        main_layout.add_widget(header)
        main_layout.add_widget(self.app.status_label)
        main_layout.add_widget(content_layout)
        self.add_widget(main_layout)

    def start_calibration(self, instance):
        # Bắt đầu quá trình hiệu chỉnh và hiệu ứng quét
        self.scanning = True
        self.is_calibrating = True
        self.scan_y = self.camera_layout.height  # Bắt đầu từ trên cùng
        self.scan_direction = -1
        Clock.schedule_interval(self.update_scan, 1 / 60)  # 60 FPS
        self.app.status_label.text = "Trạng thái: Đang hiệu chỉnh..."
        self.app.calibrate(instance)

    def stop_calibration(self):
        # Dừng hiệu ứng quét
        self.scanning = False
        self.is_calibrating = False
        Clock.unschedule(self.update_scan)
        self.camera_layout.canvas.after.clear()
        self.app.status_label.text = f'Trạng thái: Hiệu chỉnh hoàn tất'

    def update_scan(self, dt):
        # Cập nhật vị trí thanh quét và kiểm tra trạng thái hiệu chỉnh
        if not self.scanning:
            return
        if "Hiệu chỉnh hoàn tất" in self.app.status_label.text:
            self.stop_calibration()
            return
        scan_speed = self.camera_layout.height / 2  # Hoàn thành quét trong 2 giây
        self.scan_y += self.scan_direction * scan_speed * dt
        if self.scan_y <= 0:
            self.scan_y = 0
            self.scan_direction = 1
        elif self.scan_y >= self.camera_layout.height:
            self.scan_y = self.camera_layout.height
            self.scan_direction = -1
        self.camera_layout.canvas.after.clear()
        with self.camera_layout.canvas.after:
            Color(0, 1, 0, 0.5)  # Màu xanh lá mờ
            Rectangle(
                pos=(self.camera_layout.x, self.camera_layout.y + self.scan_y - 5),
                size=(self.camera_layout.width, 10)
            )

    def update_background_rect(self, instance, value):
        # Cập nhật hình chữ nhật nền
        self.background_rect.pos = instance.pos
        self.background_rect.size = instance.size

    def update_metrics(self, ear, mar, roll_angle, pitch_angle, blink_count, yawn_count=None):
        # Cập nhật các chỉ số hiển thị
        def safe_float(value, default=None):
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        is_alert = self.app.alert_active
        ear_value = safe_float(ear, None)
        self.metrics_widgets['ear']['label'].text = f'Cỡ mắt: {ear:.2f}' if ear is not None else 'Cỡ mắt: --'
        self.metrics_widgets['ear']['label'].color = [1, 0, 0,
                                                      1] if is_alert and ear is not None and ear_value < self.app.ear_threshold else [
            1, 1, 1, 1]
        if ear is not None:
            self.metrics_widgets['ear']['bar'].value = ear_value
        mar_value = safe_float(mar, None)
        self.metrics_widgets['mar']['label'].text = f'Cỡ miệng: {mar:.2f}' if mar is not None else 'Cỡ miệng: --'
        self.metrics_widgets['mar']['label'].color = [1, 0, 0,
                                                      1] if is_alert and mar is not None and mar_value > 0.5 else [1, 1,
                                                                                                                   1, 1]
        if mar is not None:
            self.metrics_widgets['mar']['bar'].value = mar_value
        roll_value = safe_float(roll_angle, None)
        self.metrics_widgets['roll_angle'][
            'label'].text = f'Góc nghiêng: {roll_angle:.1f}°' if roll_angle is not None else 'Góc nghiêng: --'
        self.metrics_widgets['roll_angle']['label'].color = [1, 0, 0,
                                                             1] if is_alert and roll_angle is not None and roll_value > 15.0 else [
            1, 1, 1, 1]
        if roll_angle is not None:
            self.metrics_widgets['roll_angle']['bar'].value = roll_value
        pitch_value = safe_float(pitch_angle, None)
        self.metrics_widgets['pitch_angle'][
            'label'].text = f'Góc cúi: {pitch_angle:.1f}°' if pitch_angle is not None else 'Góc cúi: --'
        self.metrics_widgets['pitch_angle']['label'].color = [1, 0, 0,
                                                              1] if is_alert and pitch_angle is not None and pitch_value > 15.0 else [
            1, 1, 1, 1]
        if pitch_angle is not None:
            self.metrics_widgets['pitch_angle']['bar'].value = pitch_value
        blink_value = safe_float(blink_count, 0)
        self.metrics_widgets['blink_count'][
            'label'].text = f'Nháy mắt: {int(blink_count)}' if blink_count is not None else 'Nháy mắt: 0'
        self.metrics_widgets['blink_count']['label'].color = [1, 0, 0,
                                                              1] if is_alert and self.app.detector.check_blink_frequency() else [
            1, 1, 1, 1]
        self.metrics_widgets['blink_count']['bar'].value = blink_value
        yawn_value = safe_float(yawn_count, 0)
        self.metrics_widgets['yawn_count'][
            'label'].text = f'Ngáp: {int(yawn_count)}/min' if yawn_count is not None else 'Ngáp: 0/min'
        self.metrics_widgets['yawn_count']['label'].color = [1, 0, 0,
                                                             1] if is_alert and self.app.detector.check_yawn_frequency() else [
            1, 1, 1, 1]
        self.metrics_widgets['yawn_count']['bar'].value = yawn_value