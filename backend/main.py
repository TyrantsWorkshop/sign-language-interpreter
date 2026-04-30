"""
Main FastAPI Application for Sign Language Interpreter
Real-time sign language recognition with WebSocket and REST endpoints
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
from io import BytesIO

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import ImprovedViTWithConvolutions, EmotionDetector, TriggerGestureDetector
from utils.config import settings
from utils.llm_integration import LLMProcessor
from utils.tts_integration import TTSProcessor

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Sign Language Interpreter",
    description="Real-time sign language to text and speech conversion",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device(settings.DEVICE if torch.cuda.is_available() else 'cpu')
sign_model = None
label_map = None
emotion_detector = None
gesture_detector = None
mp_holistic = None
llm_processor = None
tts_processor = None


def load_models():
    """Load all ML models"""
    global sign_model, label_map, emotion_detector, gesture_detector, mp_holistic, llm_processor, tts_processor
    
    try:
        # Load label map
        if Path(settings.LABEL_MAP_PATH).exists():
            with open(settings.LABEL_MAP_PATH, 'rb') as f:
                label_map = pickle.load(f)
        else:
            label_map = {}
            logger.warning(f"Label map not found at {settings.LABEL_MAP_PATH}")
        
        num_classes = len(label_map) if label_map else 100
        
        # Load sign language model
        if Path(settings.MODEL_PATH).exists():
            sign_model = ImprovedViTWithConvolutions(
                input_dim=1536,
                num_frames=30,
                num_classes=num_classes
            ).to(device)
            sign_model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
            sign_model.eval()
            logger.info("✓ Sign language model loaded")
        else:
            sign_model = ImprovedViTWithConvolutions(
                input_dim=1536,
                num_frames=30,
                num_classes=num_classes
            ).to(device)
            sign_model.eval()
            logger.warning(f"Sign language model not found at {settings.MODEL_PATH}, using random weights")
        
        # Load emotion detector
        emotion_detector = EmotionDetector(
            model_path=settings.EMOTION_MODEL_PATH if Path(settings.EMOTION_MODEL_PATH).exists() else None,
            device=settings.DEVICE
        )
        logger.info("✓ Emotion detector loaded")
        
        # Load gesture detector
        gesture_detector = TriggerGestureDetector(
            model_path=settings.GESTURE_MODEL_PATH if Path(settings.GESTURE_MODEL_PATH).exists() else None,
            device=settings.DEVICE
        )
        logger.info("✓ Gesture detector loaded")
        
        # MediaPipe holistic
        mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.info("✓ MediaPipe holistic loaded")
        
        # LLM processor
        llm_processor = LLMProcessor(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE
        )
        logger.info("✓ LLM processor initialized")
        
        # TTS processor
        tts_processor = TTSProcessor(
            provider=settings.TTS_PROVIDER,
            api_key=settings.ELEVENLABS_API_KEY,
            voice_id=settings.TTS_VOICE_ID
        )
        logger.info("✓ TTS processor initialized")
        
        logger.info("✓ All models loaded successfully")
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


class SignLanguageProcessor:
    """Process frames for sign language recognition"""
    
    def __init__(self):
        self.frame_buffer = []
        self.keypoint_buffer = []
        self.recording = False
        self.recognized_sequence = []
        self.current_emotion = "Neutral"
    
    def extract_keypoints(self, frame: np.ndarray) -> Dict:
        """Extract keypoints and emotion from frame"""
        results = mp_holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Extract hand landmarks
        left_hand = None
        right_hand = None
        
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] 
                                 for lm in results.left_hand_landmarks.landmark])
        
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] 
                                  for lm in results.right_hand_landmarks.landmark])
        
        # Extract pose
        pose = None
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] 
                           for lm in results.pose_landmarks.landmark])
        
        # Extract face
        face = None
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] 
                           for lm in results.face_landmarks.landmark])
        
        # Concatenate keypoints
        left_hand = left_hand if left_hand is not None else np.zeros((21, 3))
        right_hand = right_hand if right_hand is not None else np.zeros((21, 3))
        pose = pose if pose is not None else np.zeros((17, 3))
        face = face if face is not None else np.zeros((468, 3))
        
        keypoints = np.concatenate([
            left_hand.flatten(),
            right_hand.flatten(),
            pose.flatten(),
            face.flatten()
        ])
        
        # Detect emotion
        emotion_dict = emotion_detector.detect_emotion(frame)
        emotion = emotion_dict.get('emotion', 'Neutral')
        emotion_confidence = emotion_dict.get('confidence', 0.0)
        
        return {
            'keypoints': keypoints,
            'left_hand': left_hand,
            'right_hand': right_hand,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence
        }
    
    def recognize_sign(self) -> Dict:
        """Recognize sign from buffered keypoints"""
        if len(self.keypoint_buffer) < 30:
            return {'sign': None, 'confidence': 0.0, 'all_predictions': {}}
        
        # Prepare input
        sequence = np.array(self.keypoint_buffer[-30:])
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = sign_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()
        
        # Get label
        idx_to_label = {v: k for k, v in label_map.items()} if label_map else {}
        sign_label = idx_to_label.get(top_idx, f'SIGN_{top_idx}')
        
        return {
            'sign': sign_label,
            'confidence': float(confidence),
            'all_predictions': {
                idx_to_label.get(i, f'SIGN_{i}'): float(probs[i].item())
                for i in range(min(5, len(probs)))
            }
        }


processor = SignLanguageProcessor()


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()
    logger.info("Application startup complete")


@app.websocket("/ws/sign-language")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time sign language detection"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            
            # Decode frame
            np_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Extract keypoints
            result = processor.extract_keypoints(frame)
            keypoints = result['keypoints']
            emotion = result['emotion']
            processor.current_emotion = emotion
            
            # Check for trigger gesture
            gesture, gesture_conf = gesture_detector.detect_gesture(
                result['left_hand'], result['right_hand']
            )
            
            # Handle gesture states
            if gesture == 'START' and gesture_conf > 0.8:
                processor.recording = True
                processor.keypoint_buffer = []
                processor.recognized_sequence = []
                
                await websocket.send_json({
                    'status': 'recording_started',
                    'emotion': emotion,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info("Recording started")
            
            elif gesture == 'END' and gesture_conf > 0.8 and processor.recording:
                processor.recording = False
                
                # Process recognized sequence
                sequence_text = ' '.join(processor.recognized_sequence)
                
                # Get LLM response
                try:
                    llm_response = await llm_processor.contextualize_signs(sequence_text)
                except:
                    llm_response = f"Recognized: {sequence_text}"
                
                # Synthesize audio
                try:
                    audio_bytes = await tts_processor.synthesize(llm_response)
                    audio_base64 = None
                    if audio_bytes:
                        import base64
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                except:
                    audio_base64 = None
                
                await websocket.send_json({
                    'status': 'recording_ended',
                    'recognized_sequence': sequence_text,
                    'llm_response': llm_response,
                    'emotion': emotion,
                    'audio': audio_base64,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"Recording ended: {sequence_text}")
            
            elif processor.recording:
                processor.keypoint_buffer.append(keypoints)
                
                # Periodic recognition every 30 frames
                if len(processor.keypoint_buffer) % 30 == 0:
                    result = processor.recognize_sign()
                    if result['confidence'] > 0.5:
                        processor.recognized_sequence.append(result['sign'])
                    
                    await websocket.send_json({
                        'status': 'recognizing',
                        'detected_sign': result['sign'],
                        'confidence': result['confidence'],
                        'emotion': emotion,
                        'timestamp': datetime.now().isoformat()
                    })
            
            else:
                await websocket.send_json({
                    'status': 'idle',
                    'gesture': gesture,
                    'gesture_confidence': float(gesture_conf),
                    'emotion': emotion,
                    'timestamp': datetime.now().isoformat()
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.post("/api/recognize-video")
async def recognize_video(file: UploadFile = File(...)):
    """Process uploaded video file"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Save temporarily
        temp_path = f"temp_{datetime.now().timestamp()}.mp4"
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        recognized_signs = []
        processor.keypoint_buffer = []
        
        while cap.isOpened():
            ret, frame = cap.read()  
            if not ret:
                break
            
            result = processor.extract_keypoints(frame)
            processor.keypoint_buffer.append(result['keypoints'])
            
            if len(processor.keypoint_buffer) % 30 == 0:
                sign_result = processor.recognize_sign()
                if sign_result['confidence'] > 0.5:
                    recognized_signs.append(sign_result['sign'])
        
        cap.release()
        Path(temp_path).unlink()
        
        sequence_text = ' '.join(recognized_signs)
        
        try:
            llm_response = await llm_processor.contextualize_signs(sequence_text)
        except:
            llm_response = f"Recognized: {sequence_text}"
        
        return JSONResponse({
            'recognized_sequence': sequence_text,
            'llm_response': llm_response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        'status': 'healthy',
        'device': str(device),
        'models_loaded': True,
        'timestamp': datetime.now().isoformat()
    })


@app.get("/api/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return JSONResponse({
        'models': {
            'sign_language': 'ImprovedViTWithConvolutions',
            'emotion_detection': 'EmotionCNN',
            'gesture_recognition': 'GestureRecognitionNet'
        },
        'num_sign_classes': len(label_map) if label_map else 0,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
