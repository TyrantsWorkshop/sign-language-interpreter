"""
Sign Language Model - Vision Transformer with CNN
Architecture: Hybrid approach combining 1D CNN + Transformer for spatial-temporal modeling
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ViTSignLanguage(nn.Module):
    """Pure Vision Transformer for Sign Language Recognition"""
    
    def __init__(self, 
                 input_dim: int = 1536,
                 num_frames: int = 30,
                 num_classes: int = 100,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        
        super(ViTSignLanguage, self).__init__()
        
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.embedding_dim = embedding_dim
        
        # Project keypoints to embedding dimension
        self.keypoint_projection = nn.Linear(input_dim, embedding_dim)
        
        # Temporal positional encoding
        self.temporal_embedding = nn.Embedding(num_frames, embedding_dim)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_frames, input_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        B, T, D = x.shape
        
        # Project keypoints
        x = self.keypoint_projection(x)  # (B, T, embedding_dim)
        
        # Add temporal positional encoding
        pos_indices = torch.arange(T, device=x.device)
        temporal_pos = self.temporal_embedding(pos_indices)  # (T, embedding_dim)
        x = x + temporal_pos.unsqueeze(0)  # Broadcast to batch
        
        # Transformer encoding
        x = self.transformer(x)  # (B, T, embedding_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embedding_dim)
        
        # Classification
        logits = self.fc_head(x)  # (B, num_classes)
        
        return logits


class ImprovedViTWithConvolutions(nn.Module):
    """
    Hybrid model: Conv1D + ViT for better spatial-temporal learning
    Recommended for production use
    """
    
    def __init__(self,
                 input_dim: int = 1536,
                 num_frames: int = 30,
                 num_classes: int = 100,
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4):
        
        super(ImprovedViTWithConvolutions, self).__init__()
        
        self.num_frames = num_frames
        self.embedding_dim = embedding_dim
        
        # 1D Convolution layers for temporal feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(512, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames, embedding_dim))
        
        # Transformer for spatial-temporal modeling
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_frames, input_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        B, T, D = x.shape
        
        # Transpose for conv1d: (B, input_dim, T)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)  # (B, embedding_dim, T)
        
        # Transpose back: (B, T, embedding_dim)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer
        x = self.transformer(x)  # (B, T, embedding_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embedding_dim)
        
        # Classification
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits


class MultiScaleViT(nn.Module):
    """Multi-scale Vision Transformer for robust recognition across different speeds"""
    
    def __init__(self,
                 input_dim: int = 1536,
                 num_frames: int = 30,
                 num_classes: int = 100,
                 embedding_dim: int = 256):
        
        super(MultiScaleViT, self).__init__()
        
        # Different temporal scales
        self.models = nn.ModuleList([
            ImprovedViTWithConvolutions(input_dim, num_frames, 256, embedding_dim),
            ImprovedViTWithConvolutions(input_dim, num_frames, 256, embedding_dim),
            ImprovedViTWithConvolutions(input_dim, num_frames, 256, embedding_dim),
        ])
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale forward pass"""
        # Extract features at different temporal scales
        f1 = self.models[0](x)  # Full resolution
        f2 = self.models[1](x[:, ::2, :])  # 2x downsampled
        f2 = F.pad(f2, (0, f1.shape[1] - f2.shape[1]))  # Pad back
        f3 = self.models[2](x[:, ::3, :])  # 3x downsampled
        f3 = F.pad(f3, (0, f1.shape[1] - f3.shape[1]))  # Pad back
        
        # Concatenate features
        fused = torch.cat([f1, f2, f3], dim=-1)
        
        # Classify
        logits = self.fusion(fused)
        
        return logits


def create_model(model_type: str = 'vit-conv', 
                 input_dim: int = 1536,
                 num_frames: int = 30,
                 num_classes: int = 100,
                 embedding_dim: int = 256) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'vit', 'vit-conv', or 'multi-scale'
        input_dim: Keypoint feature dimension
        num_frames: Number of frames per sequence
        num_classes: Number of sign classes
        embedding_dim: Embedding dimension
    
    Returns:
        model: Initialized model
    """
    if model_type == 'vit':
        return ViTSignLanguage(input_dim, num_frames, num_classes, embedding_dim)
    elif model_type == 'vit-conv':
        return ImprovedViTWithConvolutions(input_dim, num_frames, num_classes, embedding_dim)
    elif model_type == 'multi-scale':
        return MultiScaleViT(input_dim, num_frames, num_classes, embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
