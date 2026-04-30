"""
Emotion Detection Model - Lightweight CNN for Real-time Facial Emotion Recognition
Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, Tuple


class EmotionCNN(nn.Module):
    """Lightweight CNN for real-time facial emotion detection (48x48 grayscale input)"""
    
    def __init__(self, num_emotions: int = 7):
        super(EmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 48x48 -> 24x24
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2: 24x24 -> 12x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3: 12x12 -> 6x6
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # Classifier: (256 * 6 * 6) -> emotions
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1, 48, 48) - grayscale images
        
        Returns:
            logits: (batch_size, num_emotions)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EmotionDetector:
    """Real-time emotion detection from face regions"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Initialize model
        self.model = EmotionCNN(num_emotions=len(self.emotions))
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using random initialization")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Face cascade detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48, 48)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def detect_emotion(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect emotion in frame
        
        Args:
            frame: BGR image (H, W, 3)
        
        Returns:
            dict: {emotion: confidence, ...} for all detected faces
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        emotions_list = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess face
            face_roi = cv2.resize(face_roi, (48, 48))
            face_tensor = self.transform(face_roi).unsqueeze(0).to(self.device)
            
            # Predict emotion
            with torch.no_grad():
                logits = self.model(face_tensor)
                probabilities = F.softmax(logits, dim=1)[0]
            
            # Get emotion with highest confidence
            top_emotion_idx = probabilities.argmax().item()
            top_emotion = self.emotions[top_emotion_idx]
            confidence = probabilities[top_emotion_idx].item()
            
            emotions_list.append({
                'emotion': top_emotion,
                'confidence': confidence,
                'all_probabilities': {self.emotions[i]: probabilities[i].item() 
                                     for i in range(len(self.emotions))}
            })
        
        if not emotions_list:
            return {'emotion': 'Unknown', 'confidence': 0.0, 'all_probabilities': {}}
        
        # Return highest confidence emotion
        best_emotion = max(emotions_list, key=lambda x: x['confidence'])
        return {
            'emotion': best_emotion['emotion'],
            'confidence': best_emotion['confidence'],
            'all_probabilities': best_emotion['all_probabilities']
        }
    
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


class LightweightEmotionDetector:
    """Ultra-lightweight emotion detector for edge devices (RPi, Jetson Nano)"""
    
    def __init__(self, quantized_model_path: str = None):
        self.device = 'cpu'  # Force CPU for edge devices
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Simplified model for edge
        self.model = self._create_lightweight_model()
        
        if quantized_model_path:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def _create_lightweight_model(self) -> nn.Module:
        """Create lightweight model"""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.emotions))
        )
    
    def detect_emotion(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect emotion (optimized for speed)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {'emotion': 'Unknown', 'confidence': 0.0}
        
        # Use first face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_tensor = torch.FloatTensor(face_roi).unsqueeze(0).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            logits = self.model(face_tensor)
            probs = F.softmax(logits, dim=1)[0]
        
        emotion_idx = probs.argmax().item()
        emotion = self.emotions[emotion_idx]
        confidence = probs[emotion_idx].item()
        
        return {
            'emotion': emotion,
            'confidence': confidence
        }
