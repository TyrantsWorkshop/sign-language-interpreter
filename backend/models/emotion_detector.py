"""
Emotion Detection Model - FER (facial-emotion-recognition) wrapper
Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
from fer import FER


class EmotionDetector:
    """Real-time emotion detection using FER (MTCNN or OpenCV face detection)."""

    def __init__(self, use_mtcnn: bool = True):
        # FER returns emotions: angry, disgust, fear, happy, sad, surprise, neutral
        self.detector = FER(mtcnn=use_mtcnn)

    def detect_emotion(self, frame) -> Dict[str, float]:
        """
        Detect emotion in frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            dict: {emotion, confidence, all_probabilities}
        """
        if frame is None:
            return {'emotion': 'Unknown', 'confidence': 0.0, 'all_probabilities': {}}

        # FER expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_emotions(rgb)

        if not results:
            return {'emotion': 'Unknown', 'confidence': 0.0, 'all_probabilities': {}}

        # Choose the face with highest confidence emotion
        best = None
        for res in results:
            emotions = res.get('emotions', {})
            if not emotions:
                continue
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]
            if best is None or confidence > best['confidence']:
                best = {
                    'emotion': top_emotion.capitalize(),
                    'confidence': confidence,
                    'all_probabilities': {k.capitalize(): v for k, v in emotions.items()}
                }

        if best is None:
            return {'emotion': 'Unknown', 'confidence': 0.0, 'all_probabilities': {}}

        return best

    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Get BGR color for emotion visualization"""
        emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 255, 255),  # Yellow
            'Fear': (255, 0, 0),       # Blue
            'Happy': (0, 255, 0),      # Green
            'Neutral': (128, 128, 128),# Gray
            'Sad': (255, 0, 255),      # Magenta
            'Surprise': (0, 165, 255)  # Orange
        }
        return emotion_colors.get(emotion, (255, 255, 255))
