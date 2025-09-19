import time
import numpy as np
import xgboost as xgb
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Configuration
DATA_ROOT = Path("Data/INCLUDE")
TOP_N_CLASSES = 50
OTHER_LABEL = "Other"
MIN_SAMPLES_AUG = 10
AUGMENT_FACTOR = 2
PCA_COMPONENTS = 100
SUMMARY_STATS = ["mean", "std", "min", "max"]

# 1. Determine standard feature dimension
dim_counts = Counter()
for cat in DATA_ROOT.iterdir():
    lm_root = cat / "Landmarks"
    if not lm_root.exists(): continue
    for sign in lm_root.iterdir():
        if not sign.is_dir(): continue
        for npy in sign.glob("*.npy"):
            seq = np.load(npy)
            dim_counts[seq.shape[1]] += 1
standard_dim = dim_counts.most_common(1)[0][0]
print(f"✅ Standard feature dimension: {standard_dim}")

# 2. Load and pad/truncate sequences and labels
sequences, labels = [], []
for cat in DATA_ROOT.iterdir():
    lm_root = cat / "Landmarks"
    if not lm_root.exists(): continue
    for sign in lm_root.iterdir():
        if not sign.is_dir(): continue
        for npy in sign.glob("*.npy"):
            seq = np.load(npy)
            if seq.shape[0] == 0: continue
            if seq.shape[1] < standard_dim:
                pad = np.zeros((seq.shape[0], standard_dim), dtype=np.float32)
                pad[:, :seq.shape[1]] = seq
                seq = pad
            elif seq.shape[1] > standard_dim:
                seq = seq[:, :standard_dim]
            sequences.append(seq)
            labels.append(sign.name)
print(f"✅ Loaded {len(sequences)} sequences")

# 3. Remap rare labels to "Other"
freq = Counter(labels)
top_classes = set(lbl for lbl,_ in freq.most_common(TOP_N_CLASSES))
labels = [lbl if lbl in top_classes else OTHER_LABEL for lbl in labels]

# 4. Stack frames for PCA
all_frames = np.vstack(sequences)
scaler_frames = StandardScaler().fit(all_frames)
all_frames_scaled = scaler_frames.transform(all_frames)

pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full").fit(all_frames_scaled)
print(f"✅ PCA fitted on {all_frames_scaled.shape[0]} frames")

# 5. Feature extraction per sequence
X, y = [], []
for seq, lbl in zip(sequences, labels):
    seq_scaled = scaler_frames.transform(seq)
    seq_pca    = pca.transform(seq_scaled)
    stats = []
    if "mean" in SUMMARY_STATS: stats.append(seq_pca.mean(axis=0))
    if "std"  in SUMMARY_STATS: stats.append(seq_pca.std(axis=0))
    if "min"  in SUMMARY_STATS: stats.append(seq_pca.min(axis=0))
    if "max"  in SUMMARY_STATS: stats.append(seq_pca.max(axis=0))
    if seq_pca.shape[0] > 1:
        diffs = np.diff(seq_pca, axis=0)
        stats.append(diffs.mean(axis=0))
        stats.append(diffs.std(axis=0))
    else:
        stats.append(np.zeros(PCA_COMPONENTS))
        stats.append(np.zeros(PCA_COMPONENTS))
    right = seq[:, :42].reshape(-1,21,2).mean(axis=1)
    left  = seq[:, 42:84].reshape(-1,21,2).mean(axis=1)
    dists = np.linalg.norm(right-left, axis=1)
    stats.append([dists.mean(), dists.std()])
    energy = np.sum(np.abs(np.diff(seq_pca, axis=0))) / max(1, seq_pca.shape[0])
    stats.append([energy])
    feat = np.concatenate(stats)
    X.append(feat)
    y.append(lbl)

X = np.vstack(X)
y = np.array(y)
print(f"✅ Extracted feature matrix: {X.shape}")

# 6. Filter classes with <2 samples
cnt = Counter(y)
mask = np.array([cnt[lab] > 1 for lab in y])
X, y = X[mask], y[mask]
print(f"📊 After filtering: {X.shape[0]} samples, {len(set(y))} classes")

# 7. Oversample minority classes
cnt = Counter(y)
X_aug, y_aug = [], []
for xi, yi in zip(X, y):
    X_aug.append(xi); y_aug.append(yi)
    if cnt[yi] < MIN_SAMPLES_AUG:
        for _ in range(AUGMENT_FACTOR):
            noise = np.random.normal(0, 0.01, xi.shape)
            X_aug.append(xi+noise); y_aug.append(yi)
X = np.vstack(X_aug); y = np.array(y_aug)
print(f"📊 After augmentation: {X.shape[0]} samples")

# 8. Scale final features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# 9. Encode & split
le = LabelEncoder().fit(y)
y_enc = le.transform(y)
Xtr, Xte, ytr, yte = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 10. Train XGBoost
clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    eval_metric='mlogloss',
    random_state=42
)
print("🚀 Training XGBoost...")
t0 = time.time()
clf.fit(Xtr, ytr, eval_set=[(Xtr,ytr),(Xte,yte)], verbose=1)
print(f"⏳ Training done in {time.time()-t0:.1f}s")

# 11. Evaluate
pred = clf.predict(Xte)
acc = accuracy_score(yte, pred)
print(f"🎉 Accuracy: {acc:.4f}")
print(classification_report(yte, pred, target_names=le.classes_, zero_division=0))
disp = ConfusionMatrixDisplay(confusion_matrix(yte,pred), display_labels=le.classes_)
plt.figure(figsize=(12,12)); disp.plot(cmap='Blues', xticks_rotation=90); plt.show()

# 12. Save pipeline
joblib.dump({
    "clf":clf, "pca":pca,
    "scaler_frames":scaler_frames,
    "scaler_feats":scaler,
    "label_enc":le
}, "pipeline_xgb.pkl")
print("💾 Pipeline saved as pipeline_xgb.pkl")
