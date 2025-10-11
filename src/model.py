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
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

# --- Configuration ---
DATA_ROOT = Path("../Data/INCLUDE")
OUTPUT_MODEL_PATH = Path("../models/pipeline_xgb.pkl")
TOP_N_CLASSES = 50
OTHER_LABEL = "Other"
MIN_SAMPLES_FOR_AUGMENTATION = 10
AUGMENTATION_FACTOR = 2
PCA_COMPONENTS = 100
# ---------------------

def load_and_process_data():
    """
    Loads all data, standardizes dimensions, identifies top classes,
    groups others, and returns sequences with their final labels.
    """
    dim_counts = Counter()
    all_sequences, all_labels = [], []
    for category_folder in DATA_ROOT.iterdir():
        if not category_folder.is_dir(): continue
        landmarks_root = category_folder / "Landmarks"
        if not landmarks_root.exists(): continue
        for class_folder in landmarks_root.iterdir():
            if not class_folder.is_dir(): continue
            for npy_path in class_folder.glob("*.npy"):
                try:
                    seq = np.load(npy_path)
                    if seq.ndim != 2 or seq.shape[0] == 0: continue
                    dim_counts[seq.shape[1]] += 1
                    all_sequences.append(seq)
                    all_labels.append(class_folder.name)
                except Exception as e:
                    print(f"Warning: Could not read {npy_path}. Error: {e}")

    if not all_sequences:
        raise ValueError("No valid landmark files found.")

    standard_dim = dim_counts.most_common(1)[0][0]
    print(f"✅ Standard feature dimension set to: {standard_dim}")

    label_counts = Counter(all_labels)
    top_classes = {label for label, count in label_counts.most_common(TOP_N_CLASSES)}
    print(f"✅ Identified top {len(top_classes)} classes to focus on.")

    final_sequences, final_labels = [], []
    for seq, label in zip(all_sequences, all_labels):
        if seq.shape[1] != standard_dim:
            if seq.shape[1] > standard_dim:
                seq = seq[:, :standard_dim]
            else:
                padded = np.zeros((seq.shape[0], standard_dim), dtype=np.float32)
                padded[:, :seq.shape[1]] = seq
                seq = padded
        final_sequences.append(seq)
        final_labels.append(label if label in top_classes else OTHER_LABEL)

    return final_sequences, final_labels, standard_dim

def augment_data(sequences, labels):
    """Augments data for classes with few samples."""
    print("✨ Augmenting data for less common classes...")
    label_counts = Counter(labels)
    X_aug, y_aug = [], []
    
    for seq, label in zip(sequences, labels):
        X_aug.append(seq)
        y_aug.append(label)
        if label != OTHER_LABEL and label_counts[label] < MIN_SAMPLES_FOR_AUGMENTATION:
            for _ in range(AUGMENTATION_FACTOR):
                noise = np.random.normal(0, 0.01, seq.shape)
                X_aug.append(seq + noise)
                y_aug.append(label)

    print(f"📊 Dataset size increased from {len(labels)} to {len(y_aug)} samples.")
    return X_aug, y_aug
    
def create_aggregated_feature_vector(sequence, scaler_frames, pca):
    """Transforms a landmark sequence into a single feature vector."""
    seq_scaled = scaler_frames.transform(sequence)
    seq_pca = pca.transform(seq_scaled)
    mean=seq_pca.mean(axis=0); std=seq_pca.std(axis=0); mn=seq_pca.min(axis=0); mx=seq_pca.max(axis=0)
    if seq_pca.shape[0] > 1:
        diffs = np.diff(seq_pca, axis=0)
        v_mean=diffs.mean(axis=0); v_std=diffs.std(axis=0); energy = np.sum(np.abs(diffs)) / seq_pca.shape[0]
    else:
        v_mean=np.zeros(pca.n_components_); v_std=np.zeros(pca.n_components_); energy=0.0
    try:
        right = sequence[:, :42].reshape(-1, 21, 2).mean(axis=1)
        left = sequence[:, 42:84].reshape(-1, 21, 2).mean(axis=1)
        dists = np.linalg.norm(right - left, axis=1)
        dist_mean = np.mean(dists); dist_std = np.std(dists) if dists.size > 1 else 0.0
    except:
        dist_mean, dist_std = 0.0, 0.0
    return np.concatenate([mean, std, mn, mx, v_mean, v_std, [dist_mean, dist_std], [energy]])

def main():
    print("\n[PHASE 1] Loading and Processing Data...")
    sequences, labels, standard_dim = load_and_process_data()
    
    sequences, labels = augment_data(sequences, labels)
    all_raw_frames = np.vstack(sequences)
    
    print("\n[PHASE 2] Fitting Scalers and PCA...")
    scaler_frames = StandardScaler().fit(all_raw_frames)
    pca = PCA(n_components=PCA_COMPONENTS).fit(scaler_frames.transform(all_raw_frames))
    
    print("\n[PHASE 3] Creating Final Feature Dataset...")
    X_features = np.array([create_aggregated_feature_vector(seq, scaler_frames, pca) for seq in sequences])
    
    scaler_feats = StandardScaler().fit(X_features)
    X_scaled_final = scaler_feats.transform(X_features)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    print(f"\n[PHASE 4] Training Tuned XGBoost Model on {len(label_encoder.classes_)} final classes...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # --- THE FIX: Tuned Hyperparameters for better generalization ---
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=300,          # More trees to learn complex patterns
        learning_rate=0.05,        # Lower learning rate to prevent overfitting
        max_depth=6,               # Deeper trees for more detail
        subsample=0.7,             # Use 70% of data per tree to reduce variance
        colsample_bytree=0.7,      # Use 70% of features per tree
        gamma=0.1,                 # Regularization to prevent overfitting
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    print("\n[PHASE 5] Evaluating and Saving Model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Validation Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    pipeline = {"clf": clf, "pca": pca, "scaler_frames": scaler_frames, "scaler_feats": scaler_feats, "label_enc": label_encoder}
    os.makedirs(OUTPUT_MODEL_PATH.parent, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_MODEL_PATH)
    
    print(f"\n🎉 Successfully saved new, smarter pipeline to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()