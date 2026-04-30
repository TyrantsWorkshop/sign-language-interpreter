"""
Gesture Trigger Recognition - LSTM for detecting START/END gestures
Detects when user starts and stops signing
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, Dict


class GestureRecognitionNet(nn.Module):
    """LSTM-based network for recognizing trigger gestures"""
    
    def __init__(self, input_size: int = 42, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 3):
        """
        Args:
            input_size: Hand keypoint features (21 joints × 2 for x,y)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: 3 classes (START, RECORDING, END)
        """
        super(GestureRecognitionNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last frame's hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out


class TriggerGestureDetector:
    """Detects start/end trigger gestures using hand landmarks"""
    
    def __init__(self, model_path: str = None, window_size: int = 15, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = GestureRecognitionNet(input_size=42, hidden_size=128, 
                                          num_layers=2, num_classes=3)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load gesture model from {model_path}: {e}")
        
        self.model.eval()
        self.model.to(self.device)
        
        self.window = deque(maxlen=window_size)
        self.gesture_classes = ['START', 'RECORDING', 'END']
        self.confidence_threshold = 0.7
        self.smoothing_window = deque(maxlen=5)
    
    def extract_hand_features(self, left_hand: np.ndarray, 
                            right_hand: np.ndarray) -> np.ndarray:
        """
        Extract hand features for gesture recognition
        
        Args:
            left_hand: (21, 3) hand landmarks or None
            right_hand: (21, 3) hand landmarks or None
        
        Returns:
            features: (42,) concatenated x,y coordinates
        """
        if left_hand is None:
            left_hand = np.zeros((21, 3))
        if right_hand is None:
            right_hand = np.zeros((21, 3))
        
        # Combine hand features: (42,)
        features = np.concatenate([
            left_hand[:, :2].flatten(),
            right_hand[:, :2].flatten()
        ])
        
        return features
    
    def detect_gesture(self, left_hand: np.ndarray, 
                      right_hand: np.ndarray) -> Tuple[str, float]:
        """
        Detect trigger gesture with temporal smoothing
        
        Args:
            left_hand: Left hand landmarks
            right_hand: Right hand landmarks
        
        Returns:
            (gesture_class, confidence)
        """
        features = self.extract_hand_features(left_hand, right_hand)
        self.window.append(features)
        
        if len(self.window) < self.window.maxlen:
            return 'RECORDING', 0.0
        
        window_tensor = torch.tensor(
            np.array(self.window), 
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(window_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        gesture_idx = probs.argmax().item()
        confidence = probs[gesture_idx].item()
        gesture = self.gesture_classes[gesture_idx]
        
        self.smoothing_window.append((gesture, confidence))
        
        if confidence < self.confidence_threshold:
            return 'RECORDING', confidence
        
        return gesture, confidence
