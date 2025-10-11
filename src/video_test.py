import os
# Suppress TensorFlow and MediaPipe warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import numpy as np
import mediapipe as mp
import joblib
from pathlib import Path
import argparse
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
PIPELINE_PATH = Path("../models/pipeline_xgb.pkl")
# ---------------------

def extract_landmarks_from_video(video_path: str, holistic):
    """
    Extracts landmarks from a video, ensuring a consistent 1040 feature dimension
    by correctly handling refined face landmarks (478 points).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

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
            for i in range(478):
                if i < len(face_landmarks):
                    p = face_landmarks[i]
                    lm.extend([p.x, p.y])
                else:
                    # Pad with zeros if less than 478 landmarks are detected
                    lm.extend([0.0, 0.0])
        else:
            lm.extend([0.0] * 956)

        frames.append(np.array(lm, dtype=np.float32))

    cap.release()

    if not frames:
        raise RuntimeError("No landmarks were detected in the video.")

    return np.stack(frames, axis=0)

def create_aggregated_feature_vector(sequence, scaler_frames, pca):
    """Transforms a landmark sequence into a single feature vector."""
    seq_scaled = scaler_frames.transform(sequence)
    seq_pca = pca.transform(seq_scaled)

    mean=seq_pca.mean(axis=0); std=seq_pca.std(axis=0); mn=seq_pca.min(axis=0); mx=seq_pca.max(axis=0)

    if seq_pca.shape[0] > 1:
        diffs = np.diff(seq_pca, axis=0)
        v_mean=diffs.mean(axis=0); v_std=diffs.std(axis=0)
        energy = np.sum(np.abs(diffs)) / seq_pca.shape[0]
    else:
        v_mean=np.zeros(pca.n_components_); v_std=np.zeros(pca.n_components_); energy=0.0

    try:
        right = sequence[:, :42].reshape(-1, 21, 2).mean(axis=1)
        left = sequence[:, 42:84].reshape(-1, 21, 2).mean(axis=1)
        dists = np.linalg.norm(right - left, axis=1)
        dist_mean = np.mean(dists); dist_std = np.std(dists) if dists.size > 1 else 0.0
    except Exception:
        dist_mean, dist_std = 0.0, 0.0

    return np.concatenate([mean, std, mn, mx, v_mean, v_std, [dist_mean, dist_std], [energy]])


def main():
    parser = argparse.ArgumentParser(description="Run a definitive end-to-end prediction on a video file.")
    parser.add_argument("video_file", type=str, help="Path to the video file to test.")
    args = parser.parse_args()
    video_path = Path(args.video_file)

    print("="*60)
    print("        🚀 Definitive End-to-End Video Prediction Test 🚀        ")
    print("="*60)

    if not video_path.exists():
        print(f"❌ FATAL ERROR: Video file not found at '{video_path.resolve()}'")
        return

    try:
        print("✅ Loading model pipeline...")
        pipeline = joblib.load(PIPELINE_PATH)
        clf, pca, scaler_frames, scaler_feats, le = (
            pipeline["clf"], pipeline["pca"], pipeline["scaler_frames"],
            pipeline["scaler_feats"], pipeline["label_enc"]
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load model pipeline. Error: {e}")
        return

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, refine_face_landmarks=True) as holistic:
        try:
            print(f"\n[PHASE 1] Extracting landmarks from '{video_path.name}'...")
            landmarks = extract_landmarks_from_video(str(video_path), holistic)
            print(f"✅ Landmarks extracted with shape: {landmarks.shape}")

            print("\n[PHASE 2] Creating aggregated feature vector...")
            feature_vector = create_aggregated_feature_vector(landmarks, scaler_frames, pca)

            print("\n[PHASE 3] Scaling final vector and making prediction...")
            final_features = scaler_feats.transform(feature_vector.reshape(1, -1))

            print(f"  - Final Vector Mean: {np.mean(final_features):.4f}, Std Dev: {np.std(final_features):.4f} (Should be ~0 and ~1)")

            proba = clf.predict_proba(final_features)[0]
            pred_index = np.argmax(proba)
            confidence = proba[pred_index]
            predicted_sign = le.inverse_transform([pred_index])[0]

            print("\n[FINAL RESULT]")
            print(f"  - ✅ Predicted Sign: '{predicted_sign}'")
            print(f"  - Confidence: {confidence:.4f}")

        except Exception as e:
            print(f"❌ An error occurred during processing: {e}")

if __name__ == "__main__":
    main()