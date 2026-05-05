"""
Gesture Trigger Recognition - MediaPipe Gesture Recognizer
Detects when user starts and stops signing using Open Palm / Closed Fist.
"""

from __future__ import annotations

from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class TriggerGestureDetector:
    """Detects start/end trigger gestures from webcam frames.

    Mapping:
      - Open_Palm  -> start_capture
      - Closed_Fist -> stop_capture
    """

    def __init__(self, model_path: str, num_hands: int = 1):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.last_trigger: Optional[str] = None

    def _result_callback(self, result, output_image, timestamp_ms):
        if not result.gestures:
            self.last_trigger = None
            return

        gesture = result.gestures[0][0].category_name  # e.g., Open_Palm, Closed_Fist

        if gesture == "Open_Palm":
            self.last_trigger = "start_capture"
        elif gesture == "Closed_Fist":
            self.last_trigger = "stop_capture"
        else:
            self.last_trigger = None

    def process_frame(self, frame_bgr, timestamp_ms: int) -> Optional[str]:
        """Process a single BGR frame and return trigger label or None."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
        self.recognizer.recognize_async(
            mp_image, timestamp_ms, self._result_callback
        )
        return self.last_trigger
