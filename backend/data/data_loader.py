"""
Data Loader for Sign Language Datasets
Supports How2Sign (video) and Kaggle (CSV keypoints) datasets
"""

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import pickle
import json
from tqdm import tqdm


class SignLanguageDataProcessor:
    """Process sign language datasets and extract keypoints"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Extract hand, pose, and face keypoints using MediaPipe"""
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Extract keypoints with fallback to zeros
        hand_left = np.zeros((21, 3))
        hand_right = np.zeros((21, 3))
        pose = np.zeros((17, 3))
        face = np.zeros((468, 3))
        
        if results.left_hand_landmarks:
            hand_left = np.array([[lm.x, lm.y, lm.z] 
                                 for lm in results.left_hand_landmarks.landmark])
        
        if results.right_hand_landmarks:
            hand_right = np.array([[lm.x, lm.y, lm.z] 
                                  for lm in results.right_hand_landmarks.landmark])
        
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] 
                           for lm in results.pose_landmarks.landmark])
        
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] 
                           for lm in results.face_landmarks.landmark])
        
        # Concatenate all keypoints (21*3 + 21*3 + 17*3 + 468*3 = 1536 features)
        keypoints = np.concatenate([
            hand_left.flatten(),
            hand_right.flatten(),
            pose.flatten(),
            face.flatten()
        ])
        
        return keypoints
    
    def process_video(self, video_path: str, 
                     frame_limit: int = 30) -> Tuple[np.ndarray, str]:
        """Process video file and extract keypoint sequences"""
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            keypoints = self.extract_keypoints(frame)
            frames_data.append(keypoints)
            frame_count += 1
        
        cap.release()
        
        # Pad or trim to fixed length (30 frames)
        if len(frames_data) < frame_limit:
            # Pad with zeros
            if len(frames_data) > 0:
                frames_data.extend([np.zeros_like(frames_data[0])] 
                                 * (frame_limit - len(frames_data)))
            else:
                frames_data = [np.zeros(1536)] * frame_limit
        else:
            frames_data = frames_data[:frame_limit]
        
        sequence = np.array(frames_data)  # Shape: (30, 1536)
        
        # Extract label from filename
        label = Path(video_path).parent.name
        
        return sequence, label
    
    def process_csv_keypoints(self, csv_path: str, 
                             label: str) -> Tuple[np.ndarray, str]:
        """Process Kaggle CSV files with pre-extracted keypoints"""
        df = pd.read_csv(csv_path)
        
        # CSV typically has columns: x0, y0, z0, x1, y1, z1, ...
        keypoint_cols = [col for col in df.columns if col[0] in ['x', 'y', 'z']]
        sequence = df[keypoint_cols].values  # Shape: (frames, features)
        
        # Normalize to 30 frames
        if len(sequence) < 30:
            pad_size = 30 - len(sequence)
            sequence = np.vstack([sequence, np.zeros((pad_size, sequence.shape[1]))])
        else:
            sequence = sequence[:30]
        
        return sequence, label
    
    def create_dataset(self, dataset_type: str = 'video') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create full dataset from How2Sign or Kaggle"""
        X, y = [], []
        label_to_idx = {}
        idx = 0
        
        if dataset_type == 'video':
            video_files = list(self.data_path.rglob('*.mp4'))
            for video_file in tqdm(video_files, desc="Processing videos"):
                label = video_file.parent.name
                if label not in label_to_idx:
                    label_to_idx[label] = idx
                    idx += 1
                
                sequence, _ = self.process_video(str(video_file))
                X.append(sequence)
                y.append(label_to_idx[label])
        
        elif dataset_type == 'csv':
            csv_files = list(self.data_path.glob('*.csv'))
            for csv_file in tqdm(csv_files, desc="Processing CSVs"):
                label = csv_file.stem
                if label not in label_to_idx:
                    label_to_idx[label] = idx
                    idx += 1
                
                sequence, _ = self.process_csv_keypoints(str(csv_file), label)
                X.append(sequence)
                y.append(label_to_idx[label])
        
        return np.array(X), np.array(y), label_to_idx


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process sign language datasets")
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, default='./processed_data')
    parser.add_argument('--dataset-type', type=str, choices=['video', 'csv'], default='video')
    
    args = parser.parse_args()
    
    processor = SignLanguageDataProcessor(args.dataset_path)
    X, y, label_map = processor.create_dataset(dataset_type=args.dataset_type)
    
    # Save processed data
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'X_keypoints.npy', X)
    np.save(output_path / 'y_labels.npy', y)
    
    with open(output_path / 'label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    
    print(f"Processed dataset saved to {output_path}")
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Number of classes: {len(label_map)}")
