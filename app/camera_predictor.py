import os
# Suppress TensorFlow, MediaPipe, and other warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
import mediapipe as mp
import joblib
from pathlib import Path
import logging
import warnings

# Suppress scikit-learn UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraPredictor:
    def __init__(self):
        """Initializes the predictor by loading the complete model pipeline."""
        pipeline_path = Path(__file__).parent.parent / "models" / "pipeline_xgb.pkl"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Model pipeline file not found at {pipeline_path}")
        
        pipeline = joblib.load(pipeline_path)
        self.clf = pipeline["clf"]
        self.pca = pipeline["pca"]
        self.scaler_frames = pipeline["scaler_frames"]
        self.scaler_feats = pipeline["scaler_feats"]
        self.le = pipeline["label_enc"]
        self.standard_dim = self.scaler_frames.mean_.shape[0]
        
        logger.info(f"✅ Model loaded with {len(self.le.classes_)} classes and standard_dim={self.standard_dim}")

        # MediaPipe setup (Must be IDENTICAL to the training script)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False, 
            model_complexity=1,
            smooth_landmarks=True, 
            refine_face_landmarks=True,  # This is the critical setting for 1040 features
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def extract_landmarks_from_video(self, video_path: str):
        """
        Extracts landmarks from a video, ensuring a consistent 1040 feature dimension
        by correctly handling 478 refined face landmarks.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            
            lm = []
            # Right hand (42 features)
            if results.right_hand_landmarks:
                for p in results.right_hand_landmarks.landmark: lm.extend([p.x, p.y])
            else: lm.extend([0.0] * 42)
            # Left hand (42 features)
            if results.left_hand_landmarks:
                for p in results.left_hand_landmarks.landmark: lm.extend([p.x, p.y])
            else: lm.extend([0.0] * 42)
            
            # Refined Face landmarks (478 points * 2 = 956 features)
            if results.face_landmarks:
                face_landmarks = results.face_landmarks.landmark
                for i in range(478): # Iterate up to 478 for refined landmarks
                    if i < len(face_landmarks):
                        p = face_landmarks[i]
                        lm.extend([p.x, p.y])
                    else:
                        lm.extend([0.0, 0.0]) # Pad if landmarks are missed
            else:
                lm.extend([0.0] * 956)
                
            frames.append(np.array(lm, dtype=np.float32))
            
        cap.release()

        if not frames:
            raise RuntimeError("No landmarks were detected in the video.")
        
        return np.stack(frames, axis=0)

    def create_aggregated_feature_vector(self, sequence):
        """Transforms a landmark sequence into a single feature vector using the trained pipeline."""
        seq_scaled = self.scaler_frames.transform(sequence)
        seq_pca = self.pca.transform(seq_scaled)
        
        mean=seq_pca.mean(axis=0); std=seq_pca.std(axis=0); mn=seq_pca.min(axis=0); mx=seq_pca.max(axis=0)
        
        if seq_pca.shape[0] > 1:
            diffs = np.diff(seq_pca, axis=0)
            v_mean=diffs.mean(axis=0); v_std=diffs.std(axis=0)
            energy = np.sum(np.abs(diffs)) / seq_pca.shape[0]
        else:
            v_mean=np.zeros(self.pca.n_components_); v_std=np.zeros(self.pca.n_components_); energy=0.0
            
        try:
            right = sequence[:, :42].reshape(-1, 21, 2).mean(axis=1)
            left = sequence[:, 42:84].reshape(-1, 21, 2).mean(axis=1)
            dists = np.linalg.norm(right - left, axis=1)
            dist_mean = np.mean(dists); dist_std = np.std(dists) if dists.size > 1 else 0.0
        except Exception:
            dist_mean, dist_std = 0.0, 0.0
            
        return np.concatenate([mean, std, mn, mx, v_mean, v_std, [dist_mean, dist_std], [energy]])

    def predict_from_video(self, video_path: str, threshold: float = 0.3):
        """Performs end-to-end prediction directly from a video file."""
        
        # 1. Extract landmarks with the correct 1040-feature logic
        landmarks = self.extract_landmarks_from_video(video_path)
        
        # 2. Verify the extracted dimension matches the model's expectation
        if landmarks.shape[1] != self.standard_dim:
            raise ValueError(f"Landmark dimension mismatch. Expected {self.standard_dim}, but extracted {landmarks.shape[1]}")

        # 3. Create the aggregated feature vector
        feature_vector = self.create_aggregated_feature_vector(landmarks)
        
        # 4. Scale the final features and make the prediction
        final_features = self.scaler_feats.transform(feature_vector.reshape(1, -1))
        proba = self.clf.predict_proba(final_features)[0]
        
        conf = np.max(proba)
        pred_index = np.argmax(proba)
        sign = self.le.inverse_transform([pred_index])[0]
        
        return {
            "predicted_sign": sign if conf >= threshold else "UNCERTAIN",
            "confidence": float(conf),
            "alternatives": [
                {"class": self.le.inverse_transform([i])[0], "probability": float(p)}
                for i, p in sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[:5]
            ]
        }