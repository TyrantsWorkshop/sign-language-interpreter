"""
Gesture Trigger Recognition - LSTM-based gesture detection for start/end triggers
Detects: START, RECORDING, END gestures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Optional


class GestureRecognitionNet(nn.Module):
    """LSTM-based network for recognizing trigger gestures from hand keypoints"""
    
    def __init__(self, input_size: int = 42, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 3):
        """
        Args:
            input_size: Hand features (21 joints * 2 coordinates = 42)
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
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
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
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits


class GestureRecognitionNet2D(nn.Module):
    """CNN-based gesture recognition for 2D hand keypoints"""
    
    def __init__(self, input_channels: int = 42, num_classes: int = 3):
        super(GestureRecognitionNet2D, self).__init__()
        
        # Treat keypoints as 1D sequence
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 64),  # Adaptive pooling to 8
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Transpose for conv1d: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = F.adaptive_avg_pool1d(x, 8)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


class TriggerGestureDetector:
    """
    Detects start/end trigger gestures using hand landmarks
    Uses sliding window approach for real-time detection
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 window_size: int = 15,
                 model_type: str = 'lstm',
                 device: str = 'cuda'):
        """
        Args:
            model_path: Path to pre-trained model
            window_size: Number of frames in sliding window
            model_type: 'lstm' or 'cnn'
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.gesture_classes = ['START', 'RECORDING', 'END']
        self.confidence_threshold = 0.7
        
        # Initialize model
        if model_type == 'lstm':
            self.model = GestureRecognitionNet(
                input_size=42,
                hidden_size=128,
                num_layers=2,
                num_classes=3
            )
        elif model_type == 'cnn':
            self.model = GestureRecognitionNet2D(
                input_channels=42,
                num_classes=3
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load pre-trained weights if available
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded gesture model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Sliding window buffer
        self.window = deque(maxlen=window_size)
        
        # Smoothing for temporal consistency
        self.last_gesture = 'RECORDING'
        self.gesture_history = deque(maxlen=5)
    
    def extract_hand_features(self, left_hand: Optional[np.ndarray], 
                             right_hand: Optional[np.ndarray]) -> np.ndarray:
        """
        Extract hand features for gesture recognition
        
        Args:
            left_hand: (21, 3) or None
            right_hand: (21, 3) or None
        
        Returns:
            features: (42,) containing x, y coordinates of both hands
        """
        if left_hand is None:
            left_hand = np.zeros((21, 3))
        if right_hand is None:
            right_hand = np.zeros((21, 3))
        
        # Use only x, y coordinates (drop z/depth)
        features = np.concatenate([
            left_hand[:, :2].flatten(),
            right_hand[:, :2].flatten()
        ])
        
        return features
    
    def detect_gesture(self, left_hand: Optional[np.ndarray], 
                      right_hand: Optional[np.ndarray]) -> Tuple[str, float]:
        """
        Detect trigger gesture from hand keypoints
        
        Args:
            left_hand: Left hand keypoints (21, 3)
            right_hand: Right hand keypoints (21, 3)
        
        Returns:
            (gesture_class, confidence)
        """
        # Extract features
        features = self.extract_hand_features(left_hand, right_hand)
        self.window.append(features)
        
        # Need full window to make prediction
        if len(self.window) < self.window_size:
            return 'RECORDING', 0.0
        
        # Prepare input tensor
        window_array = np.array(list(self.window))
        window_tensor = torch.tensor(
            window_array, 
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Predict gesture
        with torch.no_grad():
            logits = self.model(window_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        gesture_idx = probs.argmax().item()
        confidence = probs[gesture_idx].item()
        gesture = self.gesture_classes[gesture_idx]
        
        # Temporal smoothing: require high confidence or consensus
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > 2:
            # Check if most recent 3 are same
            recent = list(self.gesture_history)[-3:]
            if len(set(recent)) == 1 and recent[0] != 'RECORDING':
                gesture = recent[0]
                confidence = min(1.0, confidence * 1.1)  # Boost confidence
        
        # Only trigger START/END with high confidence
        if gesture != 'RECORDING' and confidence < self.confidence_threshold:
            return 'RECORDING', confidence
        
        self.last_gesture = gesture
        return gesture, confidence
    
    def reset(self):
        """Reset gesture detector state"""
        self.window.clear()
        self.gesture_history.clear()
        self.last_gesture = 'RECORDING'


class HandGestureAnalyzer:
    """Analyzes hand shapes and positions for custom gesture recognition"""
    
    @staticmethod
    def get_hand_pose(hand_keypoints: np.ndarray) -> str:
        """
        Analyze hand keypoint configuration to determine hand shape/pose
        
        Args:
            hand_keypoints: (21, 3) keypoints from MediaPipe hand detection
        
        Returns:
            pose_name: 'open_palm', 'fist', 'peace', 'thumbs_up', etc.
        """
        if hand_keypoints is None or hand_keypoints.shape[0] < 21:
            return 'unknown'
        
        # Extract key distances
        palm_center = hand_keypoints[0]
        finger_tips = hand_keypoints[[4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky tips
        
        # Calculate average finger extension
        distances = np.linalg.norm(finger_tips - palm_center, axis=1)
        avg_distance = np.mean(distances)
        
        # Simple heuristics
        if avg_distance > 0.3:
            return 'open_palm'
        elif avg_distance < 0.1:
            return 'fist'
        else:
            return 'intermediate'
    
    @staticmethod
    def get_hand_orientation(hand_keypoints: np.ndarray) -> str:
        """
        Determine hand orientation (left vs right facing direction)
        
        Args:
            hand_keypoints: (21, 3) keypoints
        
        Returns:
            orientation: 'up', 'down', 'left', 'right', 'neutral'
        """
        if hand_keypoints is None or hand_keypoints.shape[0] < 21:
            return 'unknown'
        
        wrist = hand_keypoints[0]
        middle_tip = hand_keypoints[12]
        
        direction = middle_tip - wrist
        
        # Determine primary direction
        abs_direction = np.abs(direction[:2])
        
        if abs_direction[1] > abs_direction[0]:
            return 'down' if direction[1] > 0 else 'up'
        else:
            return 'right' if direction[0] > 0 else 'left'


def create_gesture_detector(model_path: Optional[str] = None,
                           model_type: str = 'lstm',
                           window_size: int = 15) -> TriggerGestureDetector:
    """
    Factory function to create gesture detector
    
    Args:
        model_path: Path to pre-trained model
        model_type: 'lstm' or 'cnn'
        window_size: Sliding window size
    
    Returns:
        detector: TriggerGestureDetector instance
    """
    return TriggerGestureDetector(
        model_path=model_path,
        window_size=window_size,
        model_type=model_type
    )
