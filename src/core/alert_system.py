import cv2
import numpy as np
import time
import threading
import logging
from PIL import Image, ImageDraw, ImageFont
from src.configs.config import Config

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        self.config = Config()
        self.alert_cooldown = self.config.ALERT_COOLDOWN
        self.last_alert_time = 0
        self.alert_start_time = None
        self._lock = threading.Lock()
        self._font_cache = {}
        self.notification_text = None
        self.notification_start_time = None
        self.notification_duration = self.config.NOTIFICATION_DURATION

    def put_text_unicode(self, frame, text, position, color, font_size):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        if font_size not in self._font_cache:
            try:
                logger.info(f"Đang tải phông chữ từ: {self.config.FONT_PATH}")
                self._font_cache[font_size] = ImageFont.truetype(self.config.FONT_PATH, font_size)
            except Exception as e:
                logger.error(f"Không thể tải phông chữ {self.config.FONT_PATH}: {e}")
                self._font_cache[font_size] = ImageFont.load_default()
        font = self._font_cache[font_size]
        draw.text(position, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def center_text(self, frame, text, font_size, color, region_height=None):
        if font_size not in self._font_cache:
            try:
                logger.info(f"Đang tải phông chữ từ: {self.config.FONT_PATH}")
                self._font_cache[font_size] = ImageFont.truetype(self.config.FONT_PATH, font_size)
            except Exception as e:
                logger.error(f"Không thể tải phông chữ {self.config.FONT_PATH}: {e}")
                self._font_cache[font_size] = ImageFont.load_default()
        font = self._font_cache[font_size]
        temp_image = Image.new('RGB', (frame.shape[1], frame.shape[0]))
        draw = ImageDraw.Draw(temp_image)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        if region_height is None:
            text_x = (frame.shape[1] - text_width) // 2
            text_y = (frame.shape[0] - text_height) // 2
        else:
            text_x = (frame.shape[1] - text_width) // 2
            text_y = (region_height - text_height) // 2
        return self.put_text_unicode(frame, text, (text_x, text_y), color, font_size)

    def render_notification(self, frame, text):
        """Render a phone-like notification at the top 1/5 of the frame."""
        height = frame.shape[0]
        notification_height = height // 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], notification_height), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        frame = self.center_text(frame, text, font_size=24, color=self.config.TEXT_COLOR, region_height=notification_height)
        return frame

    def render_fatigue_alert(self, frame):
        """Manage and render notifications for fatigue detection with 3-second duration."""
        current_time = time.time()
        
        if not self.notification_text and not self.notification_start_time:
            self.notification_text = "Bạn đang có dấu hiệu buồn ngủ! Hãy nghỉ ngơi."
            self.notification_start_time = time.time()
            logger.info("Thông báo buồn ngủ được kích hoạt")
        
        # Chỉ render thông báo nếu nó đang active
        if self.notification_text and self.notification_start_time:
            if current_time - self.notification_start_time <= self.notification_duration:
                frame = self.render_notification(frame, self.notification_text)
            else:
                # Xóa thông báo sau 3 giây
                self.notification_text = None
                self.notification_start_time = None
                logger.debug("Xóa thông báo buồn ngủ")
        
        return frame

    def render_drowsiness_alert(self, frame, duration=None):
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 8)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.ALERT_COLOR, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        alert_text = "CẢNH BÁO NGỦ GẬT!"
        frame = self.center_text(frame, alert_text, font_size=40, color=self.config.ALERT_COLOR)
        return frame

    def render_distraction_alert(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.SECONDARY_COLOR, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        alert_text = "KHÔNG PHÁT HIỆN TÀI XẾ!"
        frame = self.center_text(frame, alert_text, font_size=30, color=self.config.ALERT_COLOR)
        return frame

    def render_head_tilt_alert(self, frame):
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 6)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 165, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        alert_text = "CẢNH BÁO TƯ THẾ ĐẦU!"
        frame = self.center_text(frame, alert_text, font_size=35, color=(255, 255, 255))
        return frame