import cv2
import numpy as np
import time
import threading
import winsound
import logging
from src.configs.config import Config
import os

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        # Khởi tạo hệ thống cảnh báo
        self.config = Config()
        self.alert_cooldown = self.config.ALERT_COOLDOWN
        self.last_alert_time = 0
        self.alert_start_time = None
        self._lock = threading.Lock()

    def render_drowsiness_alert(self, frame, duration=None):
        # Hiển thị cảnh báo buồn ngủ
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 8)  # Tạo hiệu ứng nhấp nháy
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.ALERT_COLOR, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        alert_text = "CẢNH BÁO BUỒN NGỦ!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x, text_y = (frame.shape[1] - text_size[0]) // 2, (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.config.ALERT_COLOR, 3)
        
        if duration:
            cv2.putText(frame, f"Thời gian ngủ: {duration:.1f}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        self.config.ALERT_COLOR, 2)
            

    def render_distraction_alert(self, frame):
        # Hiển thị cảnh báo mất tập trung
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.SECONDARY_COLOR, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        alert_text = "KHÔNG PHÁT HIỆN TÀI XẾ!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, alert_text, (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.config.ALERT_COLOR, 2)
        return frame

    def render_head_tilt_alert(self, frame):
        # Hiển thị cảnh báo nghiêng đầu
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 6)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 165, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        alert_text = "CẢNH BÁO TƯ THẾ ĐẦU!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, alert_text, (text_x, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return frame

    def render_fatigue_alert(self, frame):
        # Hiển thị cảnh báo mệt mỏi mắt
        overlay = frame.copy()
        alpha = 0.4 + 0.3 * abs(np.sin(time.time() * 5))
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.ALERT_COLOR, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        alert_text = "CẢNH BÁO MỆT MỎI MẮT!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x, text_y = (frame.shape[1] - text_size[0]) // 2, (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.config.ALERT_COLOR, 2)
        return frame

    def render_status_bar(self, frame, ear, ear_threshold):
        # Hiển thị thanh trạng thái EAR
        bar_length, bar_height = 150, 20
        filled_length = min(int(bar_length * (ear / 0.4)), bar_length)
        bar_x, bar_y = 20, 50
        bar_color = self.config.ALERT_COLOR if ear < ear_threshold else self.config.PRIMARY_COLOR
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_length, bar_y + bar_height), bar_color, -1)
        return frame

    def render_metrics(self, frame, metrics, status_text):
        # Hiển thị thông số và trạng thái
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (20, 80 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)
        return frame