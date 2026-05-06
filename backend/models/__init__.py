"""
Sign Language Interpreter Models Module
"""

from .sign_language_model import (
    ViTSignLanguage,
    ImprovedViTWithConvolutions,
    MultiScaleViT,
    create_model
)

try:
    from .gesture_trigger import (
        TriggerGestureDetector
    )
except Exception:  # pragma: no cover - optional dependency
    TriggerGestureDetector = None

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
    'TriggerGestureDetector',
    'EmotionCNN',
    'EmotionDetector',
    'LightweightEmotionDetector'
]
