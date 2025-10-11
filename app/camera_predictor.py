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

        # --- Camera and Real-time Prediction State ---
        self.camera = None
        self.camera_active = False
        self.landmark_buffer = deque(maxlen=30)
        self.last_prediction_time = 0
        self.current_sign = "---"
        self.motion_threshold = 0.001
        self.confidence_threshold = 0.50

    def get_class_names(self):
        """Returns the list of class names the model was trained on."""
        return sorted(self.le.classes_)

    def _extract_landmarks(self, frame):
        """Internal method to extract landmark features from a single frame."""
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
        """Creates the advanced feature vector with temporal information."""
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

    def get_frame(self):
        """Generator function to yield camera frames with diagnostics."""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Could not open camera.")
            return
        self.camera_active = True
        logger.info("Camera started.")
        while self.camera_active:
            success, frame = self.camera.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            landmarks = self._extract_landmarks(frame)
            self.landmark_buffer.append(landmarks)
            motion_magnitude = np.mean(np.std(np.array(self.landmark_buffer)[:, :84], axis=0))
            color = (0, 255, 0) if motion_magnitude >= self.motion_threshold else (0, 0, 255)
            cv2.putText(frame, f"Motion: {motion_magnitude:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Prediction: {self.current_sign}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        self.camera.release()
        self.camera = None
        logger.info("Camera released.")

    def release_camera(self):
        """Sets the flag to stop the camera feed generator."""
        self.camera_active = False

    def predict_from_buffer(self):
        """Predicts from the live buffer using stateful logic."""
        if time.time() - self.last_prediction_time < 0.75:
            return {"predicted_sign": self.current_sign}

        if len(self.landmark_buffer) < self.landmark_buffer.maxlen:
            return {"predicted_sign": "Collecting..."}

        try:
            sequence = np.array(self.landmark_buffer)
            motion_magnitude = np.mean(np.std(sequence[:, :84], axis=0))
            if motion_magnitude < self.motion_threshold:
                self.current_sign = "---"
            else:
                feature_vector = self.create_advanced_feature_vector(sequence)
                final_features = self.scaler_feats.transform(feature_vector.reshape(1, -1))
                proba = self.clf.predict_proba(final_features)[0]
                conf = np.max(proba)
                if conf >= self.confidence_threshold:
                    sign = self.le.inverse_transform([np.argmax(proba)])[0]
                    self.current_sign = sign
            self.last_prediction_time = time.time()
        except Exception as e:
            logger.error(f"Error in buffer prediction: {e}")
            self.current_sign = "Error"
        
        return {"predicted_sign": self.current_sign}

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