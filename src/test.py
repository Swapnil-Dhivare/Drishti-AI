import joblib
from pathlib import Path
import numpy as np


pipeline_path = Path(__file__).parent.parent / "models" / "pipeline_xgb.pkl"
if not pipeline_path.exists():
    raise FileNotFoundError(f"Pipeline model not found at {pipeline_path}")

pipeline = joblib.load(pipeline_path)
clf = pipeline["clf"]
pca = pipeline["pca"]
scaler_frames = pipeline["scaler_frames"]
scaler_feats = pipeline["scaler_feats"]
le = pipeline["label_enc"]




standard_dim = scaler_frames.mean_.shape[0]

def predict_sign(npy_path):
 

    seq = np.load(npy_path)
    if seq.shape[1] < standard_dim:
        pad = np.zeros((seq.shape[0], standard_dim), dtype=np.float32)
        pad[:, :seq.shape[1]] = seq
        seq = pad
    elif seq.shape[1] > standard_dim:
        seq = seq[:, :standard_dim]
    seq_scaled = scaler_frames.transform(seq)
    seq_pca = pca.transform(seq_scaled)
    mean  = seq_pca.mean(axis=0)
    std   = seq_pca.std(axis=0)
    mn    = seq_pca.min(axis=0)
    mx    = seq_pca.max(axis=0)
    if seq_pca.shape[0] > 1:
        diffs = np.diff(seq_pca, axis=0)
        v_mean = diffs.mean(axis=0)
        v_std  = diffs.std(axis=0)
    else:
        v_mean = np.zeros(pca.n_components_)
        v_std  = np.zeros(pca.n_components_)
    right = seq[:, :42].reshape(-1,21,2).mean(axis=1)
    left  = seq[:, 42:84].reshape(-1,21,2).mean(axis=1)
    dists = np.linalg.norm(right-left, axis=1)
    dist_mean = np.mean(dists)
    dist_std  = np.std(dists)
    energy = np.sum(np.abs(np.diff(seq_pca, axis=0))) / max(1, seq_pca.shape[0])
    feat = np.concatenate([
        mean, std, mn, mx,
        v_mean, v_std,
        [dist_mean, dist_std],
        [energy]
    ])
    feat_scaled = scaler_feats.transform(feat.reshape(1, -1))
    pred_enc = clf.predict(feat_scaled)[0]
    return le.inverse_transform([pred_enc])[0]

# Example usage
npy_file = Path("../Data/INCLUDE/Home/Landmarks/Paint/MVI_4413.npy")
print("Predicted sign:", predict_sign(npy_file))
