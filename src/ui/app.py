# src/ui/app.py
import logging
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from src.core.detector import DrowsinessDetector
from src.configs.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DrowsinessDetectorApp(App):
    def __init__(self):
        """
        Initialize the Kivy application for drowsiness detection.
        """
        super().__init__()
        self.config = Config()
        self.detector = DrowsinessDetector()
        self.image = Image(size_hint=(1, 0.8))
        self.status_label = Label(text='Status: Stopped', size_hint=(1, 0.1))

        self.is_monitoring = False
        self.alert_active = False
        self.alert_stop_timer = None
        self.alert_stop_delay = self.config.ALERT_STOP_DELAY
        self.camera_initialized = False

    def build(self):
        """
        Build the Kivy GUI layout with image display, status label, and control buttons.
        """
        layout = BoxLayout(orientation='vertical')
        controls = BoxLayout(size_hint=(1, 0.1), spacing=10)

        # Add buttons for start, stop, and calibrate
        controls.add_widget(Button(text='Start', on_press=self.start_monitoring))
        controls.add_widget(Button(text='Stop', on_press=self.stop_monitoring))
        controls.add_widget(Button(text='Calibrate', on_press=self.calibrate))

        layout.add_widget(self.status_label)
        layout.add_widget(self.image)
        layout.add_widget(controls)

        # Attempt to initialize the camera
        try:
            self.detector.start_camera()
            self.camera_initialized = True
        except Exception as e:
            logging.error(f"Failed to start camera: {e}")
            self.status_label.text = 'Error: Camera initialization failed'
            self.camera_initialized = False

        # Schedule frame updates at 30 FPS
        Clock.schedule_interval(self._update_wrapper, 1.0 / 30.0)
        return layout

    def _update_wrapper(self, dt):
        """
        Wrapper to ensure the update method is called with only the dt argument.
        """
        if self.is_monitoring and self.camera_initialized:
            self.update(dt)

    def start_monitoring(self, instance):
        """
        Start monitoring for drowsiness if the camera is initialized.
        """
        if not self.camera_initialized:
            self.status_label.text = 'Error: Camera not initialized'
            logging.error("Cannot start monitoring: Camera not initialized")
            return
        self.is_monitoring = True
        self.status_label.text = 'Status: Monitoring'
        logging.info("Monitoring started")

    def stop_monitoring(self, instance):
        """
        Stop monitoring and reset the GUI.
        """
        self.is_monitoring = False
        self.alert_active = False
        self.status_label.text = 'Status: Stopped'
        self.image.texture = None
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
            self.alert_stop_timer = None
        logging.info("Monitoring stopped")

    def calibrate(self, instance):
        """
        Trigger EAR calibration for the specified duration.
        """
        if not self.camera_initialized:
            self.status_label.text = 'Error: Camera not initialized'
            logging.error("Cannot calibrate: Camera not initialized")
            return
        self.status_label.text = 'Status: Calibrating...'
        logging.info("Starting calibration")
        try:
            success = self.detector.calibrate(duration=5)
            if success:
                self.status_label.text = 'Status: Calibration Complete'
                logging.info("Calibration completed successfully")
            else:
                self.status_label.text = 'Status: Calibration Failed'
                logging.error("Calibration failed: No face detected")
        except Exception as e:
            self.status_label.text = 'Status: Calibration Error'
            logging.error(f"Calibration error: {e}")

    def update(self, dt):
        """
        Update the GUI with the latest frame and status.
        Args:
            dt (float): Time elapsed since the last update.
        """
        try:
            frame, drowsiness_detected = self.detector.process_frame()
            if frame is not None:
                # Flip frame vertically for correct orientation and convert to texture
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.image.texture = texture

                # Update status based on detection
                if drowsiness_detected and not self.alert_active:
                    self.status_label.text = 'ALERT: Drowsiness Detected!'
                    self.alert_active = True
                    logging.info("Drowsiness detected")
                elif not drowsiness_detected and self.alert_active and not self.alert_stop_timer:
                    self.alert_stop_timer = Clock.schedule_once(self.stop_alert, self.alert_stop_delay)
                elif not self.alert_active:
                    self.status_label.text = 'Status: Monitoring'
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            self.status_label.text = 'Error: Frame processing failed'

    def stop_alert(self, dt):
        """
        Stop the alert and reset the status.
        """
        self.alert_active = False
        self.alert_stop_timer = None
        self.status_label.text = 'Status: Monitoring'
        logging.info("Alert stopped")

    def on_stop(self):
        """
        Clean up resources when the application closes.
        """
        logging.info("Cleaning up resources")
        if self.alert_stop_timer:
            self.alert_stop_timer.cancel()
        self.detector.stop_camera()
        self.camera_initialized = False