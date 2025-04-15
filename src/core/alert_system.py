# src/core/alert_system.py
import cv2
import numpy as np
import time
import threading
from playsound import playsound
from src.configs.config import Config


class AlertSystem:
    def __init__(self):
        self.config = Config()
        self.alert_cooldown = self.config.ALERT_COOLDOWN
        self.last_alert_time = 0

    def _play_sound(self):
        if self.config.SOUND_ENABLED and self.config.ALERT_SOUND_FILE:
            try:
                playsound(self.config.ALERT_SOUND_FILE, block=False)
            except Exception as e:
                print(f"Error playing sound: {e}")

    def play_alert_sound(self):
        if time.time() - self.last_alert_time > self.alert_cooldown:
            threading.Thread(target=self._play_sound, daemon=True).start()
            self.last_alert_time = time.time()

    def render_drowsiness_alert(self, frame, duration=None):
        self.play_alert_sound()
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 8)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.ALERT_COLOR, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        alert_text = "DROWSINESS ALERT!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x, text_y = (frame.shape[1] - text_size[0]) // 2, (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.config.ALERT_COLOR, 3)

        if duration:
            cv2.putText(frame, f"Sleep time: {duration:.1f}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        self.config.ALERT_COLOR, 2)
        return frame

    def render_distraction_alert(self, frame):
        self.play_alert_sound()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.SECONDARY_COLOR, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        alert_text = "DRIVER NOT DETECTED!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, alert_text, (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.config.ALERT_COLOR, 2)
        return frame

    def render_fatigue_alert(self, frame):
        self.play_alert_sound()
        overlay = frame.copy()
        alpha = 0.4 + 0.3 * abs(np.sin(time.time() * 5))
        alert_text = "EYE FATIGUE ALERT!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x, text_y = (frame.shape[1] - text_size[0]) // 2, (frame.shape[0] + text_size[1]) // 2
        cv2.putText(overlay, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.config.ALERT_COLOR, 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def render_status_bar(self, frame, ear, ear_threshold):
        bar_length, bar_height = 150, 20
        filled_length = min(int(bar_length * (ear / 0.4)), bar_length)
        bar_x, bar_y = 20, 50
        bar_color = self.config.ALERT_COLOR if ear < ear_threshold else self.config.PRIMARY_COLOR
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_length, bar_y + bar_height), bar_color, -1)
        return frame

    def render_metrics(self, frame, metrics, status_text):
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (20, 80 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.TEXT_COLOR, 1)
        return frame