import time
import numpy as np
import xgboost as xgb
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

# --- Configuration ---
# Corrected path to point one level up from the 'src' directory
DATA_ROOT = Path("../Data/INCLUDE")
# Path where the new, correct model will be saved.
OUTPUT_MODEL_PATH = Path("../models/pipeline_xgb.pkl")
# ---------------------

def load_filter_and_standardize_data(data_root):
    """
    Loads all data, filters out classes with only one sample, standardizes
    feature dimensions, and returns the cleaned data.
    """
    print(f"Searching for landmark data in: {data_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"The specified DATA_ROOT does not exist: {data_root}")

    # --- Step 1: Load all sequences and labels initially ---
    temp_sequences = []
    labels = []
    
    for category_folder in data_root.iterdir():
        if not category_folder.is_dir(): continue
        landmarks_root = category_folder / "Landmarks"
        if not landmarks_root.exists(): continue
        for class_folder in landmarks_root.iterdir():
            if not class_folder.is_dir(): continue
            for npy_path in class_folder.glob("*.npy"):
                try:
                    seq = np.load(npy_path)
                    if seq.ndim != 2 or seq.shape[0] == 0: continue
                    temp_sequences.append(seq)
                    labels.append(class_folder.name)
                except Exception as e:
                    print(f"Warning: Could not read {npy_path}. Error: {e}")

    if not temp_sequences:
        raise ValueError("No valid landmark files found.")

    # --- Step 2: Filter out classes with fewer than 2 samples ---
    label_counts = Counter(labels)
    labels_to_keep = {label for label, count in label_counts.items() if count >= 2}
    
    if len(labels_to_keep) < len(label_counts):
        labels_removed = sorted([label for label in label_counts if label not in labels_to_keep])
        print(f"⚠️  Found {len(labels_removed)} classes with only 1 sample. Removing them: {labels_removed}")
    
    filtered_sequences_with_labels = [
        (seq, label) for seq, label in zip(temp_sequences, labels) if label in labels_to_keep
    ]

    if not filtered_sequences_with_labels:
        raise ValueError("No classes with sufficient samples found after filtering.")

    temp_sequences, labels = zip(*filtered_sequences_with_labels)

    # --- Step 3: Standardize feature dimensions ---
    dim_counts = Counter(seq.shape[1] for seq in temp_sequences)
    standard_dim = dim_counts.most_common(1)[0][0]
    print(f"✅ Standard feature dimension set to: {standard_dim} (most common shape)")

    X_sequences = []
    all_frames_list = []
    
    for seq in temp_sequences:
        current_dim = seq.shape[1]
        if current_dim == standard_dim:
            conformed_seq = seq
        elif current_dim > standard_dim:
            conformed_seq = seq[:, :standard_dim]
        else:
            conformed_seq = np.zeros((seq.shape[0], standard_dim), dtype=np.float32)
            conformed_seq[:, :current_dim] = seq
        
        X_sequences.append(conformed_seq)
        all_frames_list.append(conformed_seq)
    
    print(f"Loaded and standardized {len(X_sequences)} sequences from {len(set(labels))} classes.")
    
    all_frames_stacked = np.vstack(all_frames_list)
    return X_sequences, np.array(labels), all_frames_stacked


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
    """Main function to run the complete training pipeline."""
    
    print("\n[PHASE 1] Loading, Filtering, and Standardizing Data...")
    sequences, labels, all_raw_frames = load_filter_and_standardize_data(DATA_ROOT)
        
    print("\n[PHASE 2] Fitting Frame Scaler and PCA...")
    scaler_frames = StandardScaler().fit(all_raw_frames)
    print("✅ Frame scaler ('scaler_frames') fitted.")
    
    pca = PCA(n_components=100).fit(scaler_frames.transform(all_raw_frames))
    print("✅ PCA model fitted.")

    print("\n[PHASE 3] Creating Aggregated Feature Vectors...")
    X_features = np.array([create_aggregated_feature_vector(seq, scaler_frames, pca) for seq in sequences])
    print(f"✅ Created final feature matrix with shape: {X_features.shape}")

    print("\n[PHASE 4] Encoding Labels and Fitting Final Feature Scaler...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    print(f"✅ Labels encoded. Training with {len(label_encoder.classes_)} unique classes.")
    
    scaler_feats = StandardScaler().fit(X_features)
    print("✅ Feature scaler ('scaler_feats') fitted.")
    X_scaled_final = scaler_feats.transform(X_features)
    
    print("\n[PHASE 5] Training the XGBoost Model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    clf = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
    
    # CORRECTED LINE: Removed extra arguments to ensure compatibility.
    clf.fit(X_train, y_train)
    
    print("\n[PHASE 6] Evaluating Model and Saving Pipeline...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Validation Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    pipeline = {"clf": clf, "pca": pca, "scaler_frames": scaler_frames, "scaler_feats": scaler_feats, "label_enc": label_encoder}
    
    os.makedirs(OUTPUT_MODEL_PATH.parent, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_MODEL_PATH)
    
    print(f"\n🎉 Successfully saved new pipeline to: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()