# Backend API Documentation

## Overview
The backend is a FastAPI application providing real-time sign language recognition, emotion detection, and gesture recognition.

## Architecture

```
FastAPI Server
├── WebSocket Endpoint (/ws/sign-language)
│   └── Real-time frame processing
├── REST Endpoints
│   ├── /api/health
│   ├── /api/models/info
│   └── /api/recognize-video
└── Model Inference
    ├── Sign Language Model (ViT)
    ├── Emotion Detector (CNN)
    └── Gesture Detector (LSTM)
```

## Endpoints

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Model Information
```http
GET /api/models/info
```

**Response:**
```json
{
  "models": {
    "sign_language": "ImprovedViTWithConvolutions",
    "emotion_detection": "EmotionCNN",
    "gesture_recognition": "GestureRecognitionNet"
  },
  "num_sign_classes": 100,
  "device": "cuda",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Video Recognition
```http
POST /api/recognize-video
Content-Type: multipart/form-data

file: [video file]
```

**Response:**
```json
{
  "recognized_sequence": "HELLO WORLD",
  "llm_response": "The person signed: Hello, how are you doing?",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### WebSocket Connection
```ws
WS /ws/sign-language
```

**Client → Server:**
- Binary frames (JPEG encoded video frames)

**Server → Client:**
```json
{
  "status": "recognizing",
  "detected_sign": "HELLO",
  "confidence": 0.94,
  "emotion": "Happy",
  "gesture": "RECORDING",
  "gesture_confidence": 0.87,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error

## Error Handling

Errors are returned as:
```json
{
  "detail": "Error description"
}
```

## Rate Limiting

None by default. Configure in production as needed.

## Authentication

No authentication by default. Add JWT/OAuth in production.
