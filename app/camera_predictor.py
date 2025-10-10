import cv2
import numpy as np
import mediapipe as mp
import joblib
from pathlib import Path
import threading
import time
from collections import Counter, deque

class CameraPredictor:
    def __init__(self):
        # Load your trained pipeline
        pipeline_path = Path(__file__).parent.parent / "models" / "pipeline_xgb.pkl"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline model not found at {pipeline_path}")
        
        pipeline = joblib.load(pipeline_path)
        self.clf = pipeline["clf"]
        self.pca = pipeline["pca"]
        self.scaler_frames = pipeline["scaler_frames"]
        self.scaler_feats = pipeline["scaler_feats"]
        self.le = pipeline["label_enc"]
        self.standard_dim = self.scaler_frames.mean_.shape[0]
        
        print(f"✅ Loaded model with {len(self.le.classes_)} classes: {self.le.classes_}")
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # ✅ KEY FIX: Store recent frames for sequence-based prediction
        self.frame_buffer = deque(maxlen=100)  # Store last 10 frames
        self.landmark_buffer = deque(maxlen=100)
        
        # Camera and prediction state
        self.cap = None
        self.current_prediction = "No Detection"
        self.confidence = 0.0
        self.is_running = False
        
    def extract_landmarks_from_frame(self, frame, use_static_mode=False):
        """Extract landmarks exactly as your training pipeline"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if use_static_mode:
            with self.mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                smooth_landmarks=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as holistic_static:
                results = holistic_static.process(rgb_frame)
        else:
            results = self.holistic.process(rgb_frame)
        
        lm = []
        
        # ✅ Extract landmarks exactly as in training (matching your test.py)
        # Right hand (21 × 2 = 42)
        if results.right_hand_landmarks:
            for p in results.right_hand_landmarks.landmark:
                lm.extend([p.x, p.y])
        else:
            lm.extend([0] * 42)
            
        # Left hand (21 × 2 = 42)
        if results.left_hand_landmarks:
            for p in results.left_hand_landmarks.landmark:
                lm.extend([p.x, p.y])
        else:
            lm.extend([0] * 42)
            
        # Face landmarks (468 × 2 = 936, same as training)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            # Extract exactly 468 face landmarks to match training
            for i in range(468):
                if i < len(face_lms):
                    p = face_lms[i]
                    lm.extend([p.x, p.y])
                else:
                    lm.extend([0.0, 0.0])
        else:
            lm.extend([0.0] * 936)  # 468 × 2
        
        # Total should be 1020 (42 + 42 + 936)
        assert len(lm) == 1020, f"Expected 1020 features, got {len(lm)}"
        
        # Pad/truncate to match training standard_dim if needed
        if len(lm) < self.standard_dim:
            lm.extend([0.0] * (self.standard_dim - len(lm)))
        elif len(lm) > self.standard_dim:
            lm = lm[:self.standard_dim]
        
        return np.array(lm).reshape(1, -1), results
    
    def predict_from_sequence(self, landmark_sequence):
        """Predict using sequence of landmarks (like training)"""
        try:
            if len(landmark_sequence) == 0:
                return "No Detection", 0.0
            
            # Convert list of landmarks to numpy array
            seq = np.vstack(landmark_sequence)  # Shape: (frames, features)
            
            # ✅ Apply exact feature engineering from training
            # Frame scaling and PCA
            seq_scaled = self.scaler_frames.transform(seq)
            seq_pca = self.pca.transform(seq_scaled)
            
            # Statistical features
            mean = seq_pca.mean(axis=0)
            std = seq_pca.std(axis=0)
            mn = seq_pca.min(axis=0)
            mx = seq_pca.max(axis=0)
            
            # Velocity features (frame differences)
            if seq_pca.shape[0] > 1:
                diffs = np.diff(seq_pca, axis=0)
                v_mean = diffs.mean(axis=0)
                v_std = diffs.std(axis=0)
            else:
                v_mean = np.zeros(self.pca.n_components_)
                v_std = np.zeros(self.pca.n_components_)
            
            # Inter-hand distance features
            # Extract hand landmarks from original sequence
            right_hands = seq[:, :42].reshape(-1, 21, 2)  # All frames, right hand
            left_hands = seq[:, 42:84].reshape(-1, 21, 2)  # All frames, left hand
            
            if right_hands.shape[0] > 0 and left_hands.shape[0] > 0:
                # Calculate centroid for each frame
                right_centroids = right_hands.mean(axis=1)  # (frames, 2)
                left_centroids = left_hands.mean(axis=1)    # (frames, 2)
                
                # Calculate distances for each frame
                distances = np.linalg.norm(right_centroids - left_centroids, axis=1)
                dist_mean = distances.mean()
                dist_std = distances.std()
            else:
                dist_mean = 0.0
                dist_std = 0.0
            
            # Cumulative movement energy
            if seq_pca.shape[0] > 1:
                energy = np.sum(np.abs(diffs)) / seq_pca.shape[0]
            else:
                energy = 0.0
            
            # ✅ Combine features exactly like training
            feat = np.concatenate([
                mean, std, mn, mx,      # PCA statistics
                v_mean, v_std,          # Velocity features
                [dist_mean, dist_std],  # Inter-hand distance
                [energy]                # Movement energy
            ])
            
            # Scale features
            feat_scaled = self.scaler_feats.transform(feat.reshape(1, -1))
            
            # Predict
            pred_enc = self.clf.predict(feat_scaled)[0]
            proba = self.clf.predict_proba(feat_scaled)[0]
            confidence = np.max(proba)
            
            predicted_sign = self.le.inverse_transform([pred_enc])[0]
            
            print(f"🔍 Prediction: {predicted_sign} ({confidence:.3f}) from {len(landmark_sequence)} frames")
            
            return predicted_sign, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0
    
    def get_frame_with_prediction(self):
        """Get current frame with prediction overlay"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Extract landmarks
        landmarks, results = self.extract_landmarks_from_frame(frame, use_static_mode=False)
        
        # ✅ Add landmarks to buffer for sequence prediction
        if landmarks is not None and np.sum(landmarks) > 0:  # Check if not all zeros
            self.landmark_buffer.append(landmarks[0])  # Store single frame landmarks
        
        # ✅ Predict using sequence of recent frames
        if len(self.landmark_buffer) >= 3:  # Need at least 3 frames
            prediction, confidence = self.predict_from_sequence(list(self.landmark_buffer))
            
            # Only update if confidence is reasonable
            if confidence > 0.3:
                self.current_prediction = prediction
                self.confidence = confidence
            
            # Draw landmarks
            if results and results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, 
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
            
            if results and results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, 
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        else:
            self.current_prediction = "Collecting frames..."
            self.confidence = 0.0
        
        # Add overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (frame.shape[1]-10, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add prediction text
        cv2.putText(frame, f"Sign: {self.current_prediction}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames: {len(self.landmark_buffer)}/10", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.is_running = True
        
        # Clear buffers
        self.landmark_buffer.clear()
        
        return self.cap.isOpened()
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.landmark_buffer.clear()
    
    def get_prediction_data(self):
        """Get current prediction data for API"""
        return {
            "prediction": self.current_prediction,
            "confidence": float(self.confidence),
            "timestamp": time.time(),
            "frames_collected": len(self.landmark_buffer)
        }
    
    def predict_from_video(self, video_path):
        """Analyze uploaded video using proper sequence processing"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return "Error: Cannot open video", 0.0
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📹 Processing video: {total_frames} frames, {duration:.1f}s duration")
            
            # ✅ Collect landmarks from entire video sequence
            video_landmarks = []
            frame_count = 0
            sample_rate = max(1, total_frames // 30)  # Sample ~30 frames max
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    landmarks, _ = self.extract_landmarks_from_frame(frame, use_static_mode=True)
                    
                    if landmarks is not None and np.sum(landmarks) > 0:
                        video_landmarks.append(landmarks[0])
                
                frame_count += 1
            
            cap.release()
            
            if len(video_landmarks) < 3:
                return "Video too short or no landmarks detected", 0.0
            
            print(f"✅ Extracted {len(video_landmarks)} landmark frames from video")
            
            # ✅ Use sequence prediction on video landmarks
            prediction, confidence = self.predict_from_sequence(video_landmarks)
            
            return prediction, confidence
            
        except Exception as e:
            if cap and cap.isOpened():
                cap.release()
            print(f"❌ Video processing error: {str(e)}")
            return f"Error: {str(e)}", 0.0
