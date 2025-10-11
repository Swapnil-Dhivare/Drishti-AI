import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
import mediapipe as mp
import joblib
from pathlib import Path
import logging
import warnings
from collections import deque
import time

warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraPredictor:
    def __init__(self):
        # --- Model Loading ---
        pipeline_path = Path(__file__).parent.parent / "models" / "pipeline_xgb.pkl"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Model pipeline file not found at {pipeline_path}")
        
        pipeline = joblib.load(pipeline_path)
        self.clf, self.pca, self.scaler_frames, self.scaler_feats, self.le = (
            pipeline["clf"], pipeline["pca"], pipeline["scaler_frames"],
            pipeline["scaler_feats"], pipeline["label_enc"]
        )
        self.standard_dim = self.scaler_frames.mean_.shape[0]
        logger.info(f"✅ Model loaded with {len(self.le.classes_)} classes.")

        # --- MediaPipe Initialization ---
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            refine_face_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # self.mp_drawing is not needed for the clean feed

        # --- Camera and Real-time Prediction State ---
        self.camera = None
        self.landmark_buffer = deque(maxlen=30)  # Buffer for ~1 second of frames at 30fps
        self.last_prediction_time = 0
        self.current_prediction = {"predicted_sign": "---", "confidence": 0.0}
        self.motion_threshold = 0.01 # Threshold to detect hand movement
        self.confidence_threshold = 0.65 # Only show predictions above 65% confidence

    # --- Core Logic ---

    def _extract_landmarks(self, frame):
        """Internal method to extract 1040 features from a single frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        
        lm = []
        for hand_landmarks in [results.right_hand_landmarks, results.left_hand_landmarks]:
            if hand_landmarks:
                for p in hand_landmarks.landmark: lm.extend([p.x, p.y])
            else: lm.extend([0.0] * 42)
        
        if results.face_landmarks:
            face_landmarks = results.face_landmarks.landmark
            for i in range(478):
                if i < len(face_landmarks): lm.extend([face_landmarks[i].x, face_landmarks[i].y])
                else: lm.extend([0.0, 0.0])
        else:
            lm.extend([0.0] * 956)
            
        return np.array(lm, dtype=np.float32)

    def _create_aggregated_feature_vector(self, sequence):
        """Internal method identical to training script's feature creation."""
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
        except:
            dist_mean, dist_std = 0.0, 0.0
            
        return np.concatenate([mean, std, mn, mx, v_mean, v_std, [dist_mean, dist_std], [energy]])

    # --- Camera-Specific Methods ---

    def get_frame(self):
        """Generator function to yield clean camera frames for the web feed."""
        if not self.camera or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Could not open camera.")
                return

        while True:
            success, frame = self.camera.read()
            if not success:
                self.camera.release()
                break
            
            frame = cv2.flip(frame, 1)
            landmarks = self._extract_landmarks(frame)
            self.landmark_buffer.append(landmarks)
            
            # --- NO DRAWING ---
            # The landmarks are extracted for prediction but not drawn on the frame.
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        self.camera.release()


    def predict_from_buffer(self):
        """Predicts from the live landmark buffer with motion and confidence checks."""
        if time.time() - self.last_prediction_time < 0.5:
            return self.current_prediction

        if len(self.landmark_buffer) < self.landmark_buffer.maxlen:
            return {"predicted_sign": "Collecting...", "confidence": 0.0}

        try:
            sequence = np.array(self.landmark_buffer)
            
            # --- MOTION DETECTION ---
            # Get hand landmarks (first 84 features)
            hand_landmarks = sequence[:, :84]
            # Calculate standard deviation of hand positions
            motion_magnitude = np.mean(np.std(hand_landmarks, axis=0))
            
            if motion_magnitude < self.motion_threshold:
                self.current_prediction = {"predicted_sign": "---", "confidence": 0.0}
                self.last_prediction_time = time.time()
                return self.current_prediction

            # --- PREDICTION LOGIC ---
            feature_vector = self._create_aggregated_feature_vector(sequence)
            final_features = self.scaler_feats.transform(feature_vector.reshape(1, -1))
            proba = self.clf.predict_proba(final_features)[0]
            
            conf = np.max(proba)
            sign = self.le.inverse_transform([np.argmax(proba)])[0]
            
            # --- CONFIDENCE THRESHOLD ---
            if conf >= self.confidence_threshold:
                 self.current_prediction = {"predicted_sign": sign, "confidence": float(conf)}
            else:
                 self.current_prediction = {"predicted_sign": "---", "confidence": 0.0}
                 
            self.last_prediction_time = time.time()
        except Exception as e:
            logger.error(f"Error in buffer prediction: {e}")
            self.current_prediction = {"predicted_sign": "Error", "confidence": 0.0}

        return self.current_prediction

    # --- File-Specific Method (Unchanged) ---
    def predict_from_video(self, video_path: str, threshold: float = 0.3):
        # This reuses the same internal logic as the camera
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            landmarks = self._extract_landmarks(frame)
            landmarks_list.append(landmarks)
        cap.release()

        if not landmarks_list:
            raise RuntimeError("No landmarks detected in video.")

        sequence = np.stack(landmarks_list)
        if sequence.shape[1] != self.standard_dim:
            raise ValueError(f"Landmark dimension mismatch. Expected {self.standard_dim}, got {sequence.shape[1]}")

        feature_vector = self._create_aggregated_feature_vector(sequence)
        final_features = self.scaler_feats.transform(feature_vector.reshape(1, -1))
        proba = self.clf.predict_proba(final_features)[0]
        
        conf = np.max(proba)
        sign = self.le.inverse_transform([np.argmax(proba)])[0]
        
        return {
            "predicted_sign": sign if conf >= threshold else "UNCERTAIN",
            "confidence": float(conf),
            "alternatives": [
                {"class": self.le.inverse_transform([i])[0], "probability": float(p)}
                for i, p in sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[:5]
            ]
        }