import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data(data_root):
    X, y = [], []
    for category_folder in Path(data_root).iterdir():
        landmarks_root = category_folder/"Landmarks"
        if not landmarks_root.exists(): continue
        for sign_folder in landmarks_root.iterdir():
            if not sign_folder.is_dir(): continue
            for npy_path in sign_folder.glob("*.npy"):
                seq = np.load(npy_path)
                feat = np.concatenate([
                    seq.mean(axis=0),
                    seq.std(axis=0),
                    seq.min(axis=0),
                    seq.max(axis=0)
                ])
                X.append(feat)
                y.append(sign_folder.name)
    return np.vstack(X), np.array(y)

if __name__=="__main__":
    data_root = "Data/INCLUDE"
    X, y = load_data(data_root)

    # Encode class labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=0.2,
        random_state=42, stratify=y_enc
    )

    # XGBoost classifier
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        verbose=True
    )

    # Predict and compute accuracy
    y_pred_enc = clf.predict(Xte)
    accuracy = accuracy_score(yte, y_pred_enc)
    print(f"Accuracy: {accuracy:.4f}")

    # Save model to file
    clf.save_model("xgboost_model.json")

    # Decode numeric predictions to class names
    y_test_names = le.inverse_transform(yte)
    y_pred_names = le.inverse_transform(y_pred_enc)

    # Print a few example predictions
    print("\nSample predictions:")
    for true, pred in zip(y_test_names[:10], y_pred_names[:10]):
        print(f"True: {true:15s}  Predicted: {pred}")

    # Plot confusion matrix
    cm = confusion_matrix(yte, y_pred_enc, labels=range(len(le.classes_)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=90)
    plt.title("Confusion Matrix")
    plt.show()
