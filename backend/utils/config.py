"""
Configuration Management for Sign Language Interpreter
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Keys
    OPENAI_API_KEY: str = "your_key_here"
    ELEVENLABS_API_KEY: str = "your_key_here"
    
    # Model Paths
    MODEL_PATH: str = "./saved_models/sign_language_model.pt"
    EMOTION_MODEL_PATH: str = "./saved_models/emotion_model.pt"
    GESTURE_MODEL_PATH: str = "./saved_models/gesture_model.pt"
    LABEL_MAP_PATH: str = "./saved_models/label_map.pkl"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    WORKERS: int = 1
    
    # Device Configuration
    DEVICE: str = "cuda"
    CUDA_VISIBLE_DEVICES: str = "0"
    
    # Training Configuration
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 50
    GRADIENT_CLIP: float = 1.0
    
    # Frontend Configuration
    FRONTEND_URL: str = "http://localhost:3000"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # TTS Configuration
    TTS_PROVIDER: str = "elevenlabs"
    TTS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"
    TTS_MODEL: str = "eleven_monolingual_v1"
    
    # LLM Configuration
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_MAX_TOKENS: int = 100
    LLM_TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
