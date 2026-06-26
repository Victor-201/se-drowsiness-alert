import cv2
import numpy as np
import time
import logging
from PIL import Image, ImageDraw, ImageFont
from src.configs.config import Config

logger = logging.getLogger(__name__)


def _add_weighted(src1, alpha, src2, beta, gamma, dst):
    dst[:] = np.clip(src1.astype(np.float64) * alpha + src2.astype(np.float64) * beta + gamma, 0, 255).astype(np.uint8)
    return dst


def _bgr_to_rgb(bgr):
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _rgb_to_bgr(rgb):
    return np.ascontiguousarray(rgb[:, :, ::-1])


class AlertSystem:
    def __init__(self):
        self.config = Config()
        self._font_cache = {}

    def put_text_unicode(self, frame, text, position, color, font_size):
        frame_rgb = _bgr_to_rgb(frame)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        if font_size not in self._font_cache:
            try:
                self._font_cache[font_size] = ImageFont.truetype(self.config.FONT_PATH, font_size)
            except Exception:
                self._font_cache[font_size] = ImageFont.load_default()
        font = self._font_cache[font_size]
        draw.text(position, text, font=font, fill=color[::-1])
        return _rgb_to_bgr(np.array(pil_image))

    def center_text(self, frame, text, font_size, color, region_height=None):
        if font_size not in self._font_cache:
            try:
                self._font_cache[font_size] = ImageFont.truetype(self.config.FONT_PATH, font_size)
            except Exception:
                self._font_cache[font_size] = ImageFont.load_default()
        font = self._font_cache[font_size]
        temp_image = Image.new('RGB', (frame.shape[1], frame.shape[0]))
        draw = ImageDraw.Draw(temp_image)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (frame.shape[1] - text_width) // 2
        if region_height is None:
            text_y = (frame.shape[0] - text_height) // 2
        else:
            text_y = (region_height - text_height) // 2
        return self.put_text_unicode(frame, text, (text_x, text_y), color, font_size)

    def render_drowsiness_alert(self, frame, duration=None):
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 8)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.ALERT_COLOR, -1)
        _add_weighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return self.center_text(frame, "CẢNH BÁO NGỦ GẬT!", font_size=40, color=self.config.ALERT_COLOR)

    def render_distraction_alert(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.config.SECONDARY_COLOR, -1)
        _add_weighted(overlay, 0.3, frame, 0.7, 0, frame)
        return self.center_text(frame, "KHÔNG PHÁT HIỆN TÀI XẾ!", font_size=30, color=self.config.ALERT_COLOR)

    def render_head_tilt_alert(self, frame):
        overlay = frame.copy()
        alpha = 0.4 + 0.2 * np.sin(time.time() * 6)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 165, 255), -1)
        _add_weighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return self.center_text(frame, "CẢNH BÁO TƯ THẾ ĐẦU!", font_size=35, color=(255, 255, 255))

    def render_fatigue_alert(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        _add_weighted(overlay, 0.7, frame, 0.3, 0, frame)
        return self.center_text(frame, "Bạn đang có dấu hiệu buồn ngủ!", font_size=24, color=(255, 255, 255), region_height=50)
