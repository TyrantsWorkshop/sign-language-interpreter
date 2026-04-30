# Architecture & Design

## System Overview

```
┌─────────────────┐
│  User/Webcam   │
└────────┬────────┘
         │ Video Stream
         ▼
┌─────────────────────────────────────┐
│     Frontend (React 18)             │
│  • VideoFeed Component              │
│  • Real-time UI Updates             │
│  • WebSocket Client                 │
└────────┬────────────────────────────┘
         │ HTTP/WebSocket
         ▼
┌─────────────────────────────────────────────────┐
│        Backend (FastAPI)                        │
│  ┌────────────────────────────────────────────┐ │
│  │  Frame Processing Pipeline                 │ │
│  │  1. MediaPipe Holistic Extraction          │ │
│  │  2. Keypoint Normalization                 │ │
│  │  3. Sequence Buffering                     │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │  Model Inference                           │ │
│  │  ┌─────────────────┐                       │ │
│  │  │ Gesture Detector│ ──► START/END/REC    │ │
│  │  └─────────────────┘                       │ │
│  │  ┌─────────────────┐                       │ │
│  │  │ Emotion Detector│ ──► 7 Emotions       │ │
│  │  └─────────────────┘                       │ │
│  │  ┌─────────────────┐                       │ │
│  │  │  Sign Model     │ ──► 100+ Signs       │ │
│  │  └─────────────────┘                       │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │  Post-Processing                           │ │
│  │  • LLM Contextualization (GPT-3.5)        │ │
│  │  • TTS Synthesis (ElevenLabs)             │ │
│  └────────────────────────────────────────────┘ │
└────────┬────────────────────────────────────────┘
         │ JSON Response
         ▼
┌─────────────────────┐
│  Frontend Display   │
│  • Recognized Text  │
│  • Audio Playback   │
│  • Emotion Badge    │
└─────────────────────┘
```

## Data Flow

### Real-time Recognition Flow

```
Webcam Frame
     ↓
[MediaPipe Extraction]
  • Hand keypoints (21*3*2 = 126 features)
  • Pose keypoints (17*3 = 51 features)
  • Face keypoints (468*3 = 1404 features)
  • Total: 1536 features per frame
     ↓
[Gesture Detection]
  • Extract hand keypoints → LSTM → START/END/REC
  • If START: Initialize buffer
  • If END: Process sequence
     ↓
[Emotion Detection]
  • Extract face region
  • CNN inference → 7 emotions
  • Return emotion + confidence
     ↓
[Sign Buffering]
  • Accumulate frames (max 30)
  • When buffer full or END signal:
     ↓
[Sign Recognition]
  • ViT inference → Sign predictions
  • Get top-K signs
  • Confidence > threshold?
     ↓
[LLM Contextualization]
  • "HELLO WORLD" → ChatGPT
  • Returns: "The person signed hello and greeted us."
     ↓
[TTS Synthesis]
  • Text → ElevenLabs API
  • Returns: Audio bytes
     ↓
[Response to Frontend]
  JSON with: signs, emotion, audio, llm_response
```

## Model Architecture Details

### Vision Transformer for Sign Language

```
Input: (B, T, 1536)  where T=30 frames
   ↓
[Linear Projection]
Input Projection: (B, T, 1536) → (B, T, 256)
   ↓
[Positional Encoding]
Add learnable temporal positions: (T, 256)
   ↓
[Conv1D Feature Extraction]  [OPTIONAL]
3 Conv blocks: (B, 1536, 30) → (B, 256, 30)
BatchNorm + GELU + Dropout
   ↓
[Vision Transformer]
6 Transformer encoder layers:
  • Multi-head attention (8 heads)
  • Feed-forward networks (d_ff=1024)
  • LayerNorm + Residual connections
  • Dropout: 0.1
   ↓
[Global Average Pooling]
(B, T, 256) → (B, 256)
   ↓
[Classification Head]
Linear (256 → 128)
  ↓
ReLU + Dropout(0.3)
  ↓
Linear (128 → num_classes)
   ↓
Output: (B, num_classes)  [Logits]
```

### Emotion Detection CNN

```
Input: (B, 1, 48, 48)  [Grayscale face]
   ↓
[Block 1]
Conv2d(1, 64, 3, padding=1)
BatchNorm2d(64)
ReLU
MaxPool2d(2, 2)           → (B, 64, 24, 24)
Dropout(0.25)
   ↓
[Block 2]
Conv2d(64, 128, 3, padding=1)
BatchNorm2d(128)
ReLU
MaxPool2d(2, 2)           → (B, 128, 12, 12)
Dropout(0.25)
   ↓
[Block 3]
Conv2d(128, 256, 3, padding=1)
BatchNorm2d(256)
ReLU
MaxPool2d(2, 2)           → (B, 256, 6, 6)
Dropout(0.25)
   ↓
[Flatten]
(B, 256, 6, 6) → (B, 9216)
   ↓
[FC Layers]
Linear(9216, 512) → ReLU → Dropout(0.5)
Linear(512, 256) → ReLU → Dropout(0.3)
Linear(256, 7)  [7 emotions]
   ↓
Output: (B, 7)  [Emotion logits]
```

### Gesture Recognition LSTM

```
Input: (B, T, 42)  where T=15, 42 = 21*2 per hand
   ↓
[LSTM Layer 1]
LSTM input_size=42, hidden_size=128
Dropout=0.3
   ↓
[LSTM Layer 2]
LSTM input_size=128, hidden_size=128
Dropout=0.3
   ↓
[Take Last Hidden State]
h_n: (B, 128)
   ↓
[FC Head]
Linear(128, 64) → ReLU → Dropout(0.3)
Linear(64, 3)  [START, RECORDING, END]
   ↓
Output: (B, 3)  [Gesture logits]
```

## Integration Points

### MediaPipe Integration
- Input: BGR video frames
- Output: 1536-dimensional keypoint vectors
- Extraction: Hand (21 joints), Pose (17 joints), Face (468 points)
- Fallback: Zero padding if body part not detected

### OpenAI GPT Integration
- Input: Recognized sign sequence (e.g., "HELLO GOOD MORNING")
- System prompt: Sign language interpretation context
- Output: Natural language response
- Async support via aiohttp

### ElevenLabs TTS Integration
- Input: Text string from LLM
- API endpoint: https://api.elevenlabs.io/v1/text-to-speech
- Output: MP3 audio bytes
- Voice ID: Configurable (default 21m00Tcm4TlvDq8ikWAM)

## Performance Optimizations

1. **Mixed Precision Training**
   - FP16 for forward pass
   - FP32 for loss calculation
   - Reduces memory by ~50%

2. **Batch Processing**
   - Accumulate frames before inference
   - Reduces model switching overhead

3. **Async Operations**
   - Non-blocking WebSocket updates
   - Parallel API calls (LLM + TTS)

4. **Model Caching**
   - Single model instance
   - No redundant loading

5. **Frame Skipping**
   - Skip frames if processing lags
   - Maintain 30 FPS target

## Scalability Considerations

1. **Horizontal Scaling**
   - Load balancer → Multiple backend instances
   - Shared model cache (Redis)
   - Session affinity for WebSocket

2. **Model Serving**
   - ONNX Runtime for inference
   - TensorRT for further optimization
   - Model quantization for edge devices

3. **Database**
   - Store recognized sequences
   - Track user statistics
   - Cache LLM responses
