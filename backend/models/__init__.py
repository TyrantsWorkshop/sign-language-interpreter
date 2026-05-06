"""
Sign Language Interpreter Models Module
"""

from .sign_language_model import (
    ViTSignLanguage,
    ImprovedViTWithConvolutions,
    MultiScaleViT,
    create_model
)

from .gesture_trigger import (
    GestureRecognitionNet,
    TriggerGestureDetector
)

try:
    from .emotion_detector import (
        EmotionCNN,
        EmotionDetector,
        LightweightEmotionDetector
    )
except Exception:  # pragma: no cover - optional dependency
    EmotionCNN = None
    EmotionDetector = None
    LightweightEmotionDetector = None

__all__ = [
    'ViTSignLanguage',
    'ImprovedViTWithConvolutions',
    'MultiScaleViT',
    'create_model',
    'GestureRecognitionNet',
    'TriggerGestureDetector',
    'EmotionCNN',
    'EmotionDetector',
    'LightweightEmotionDetector'
]
