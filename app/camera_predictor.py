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

        # --- Real-time Prediction State ---
        self.landmark_buffer = deque(maxlen=30)
        self.current_sign = "---"
        self.motion_threshold = 0.0015
        self.confidence_threshold = 0.50

    def get_class_names(self):
        return sorted(self.le.classes_)

    def _extract_landmarks(self, frame):
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

    def create_advanced_feature_vector(self, sequence):
        seq_scaled = self.scaler_frames.transform(sequence)
        seq_pca = self.pca.transform(seq_scaled)
        mean=seq_pca.mean(axis=0); std=seq_pca.std(axis=0)
        start_pos = seq_pca[0]; mid_pos = seq_pca[len(seq_pca) // 2]; end_pos = seq_pca[-1]
        if seq_pca.shape[0] > 1:
            diffs = np.diff(seq_pca, axis=0)
            v_mean=diffs.mean(axis=0); v_std=diffs.std(axis=0)
        else:
            v_mean=np.zeros(self.pca.n_components_); v_std=np.zeros(self.pca.n_components_)
        return np.concatenate([mean, std, start_pos, mid_pos, end_pos, v_mean, v_std])

    def predict_from_live_frame(self, frame):
        """Processes a single frame from the browser and updates the prediction state."""
        landmarks = self._extract_landmarks(frame)
        self.landmark_buffer.append(landmarks)

        motion_magnitude = 0.0
        motion_detected = False

        if len(self.landmark_buffer) < self.landmark_buffer.maxlen:
            return {"predicted_sign": "Collecting...", "diagnostics": {"motion_magnitude": 0, "motion_detected": False}}

        try:
            sequence = np.array(self.landmark_buffer)
            motion_magnitude = np.mean(np.std(sequence[:, :84], axis=0))
            motion_detected = motion_magnitude >= self.motion_threshold

            if not motion_detected:
                self.current_sign = "---"
            else:
                feature_vector = self.create_advanced_feature_vector(sequence)
                final_features = self.scaler_feats.transform(feature_vector.reshape(1, -1))
                proba = self.clf.predict_proba(final_features)[0]
                conf = np.max(proba)
                
                if conf >= self.confidence_threshold:
                    sign = self.le.inverse_transform([np.argmax(proba)])[0]
                    self.current_sign = sign
            
            return {
                "predicted_sign": self.current_sign,
                "diagnostics": { "motion_magnitude": motion_magnitude, "motion_detected": motion_detected }
            }
        except Exception as e:
            return {"predicted_sign": "Error"}

    def predict_from_video(self, video_path: str, threshold: float = 0.3):
        landmarks_list = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            landmarks = self._extract_landmarks(frame)
            landmarks_list.append(landmarks)
        cap.release()
        if not landmarks_list: raise RuntimeError("No landmarks detected in video.")
        sequence = np.stack(landmarks_list)
        if sequence.shape[1] != self.standard_dim:
            raise ValueError(f"Landmark dimension mismatch. Expected {self.standard_dim}, got {sequence.shape[1]}")
        feature_vector = self.create_advanced_feature_vector(sequence)
        final_features = self.scaler_feats.transform(feature_vector.reshape(1, -1))
        proba = self.clf.predict_proba(final_features)[0]
        conf = np.max(proba)
        sign = self.le.inverse_transform([np.argmax(proba)])[0]
        return {
            "predicted_sign": sign if conf >= threshold else "UNCERTAIN",
            "confidence": float(conf),
            "alternatives": [{"class": self.le.classes_[i], "probability": float(p)} for i, p in sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[:5]]
        }

