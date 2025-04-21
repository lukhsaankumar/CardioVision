import os
import wfdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def compute_hrv_features(rr_intervals):
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0
    mean_rr = np.mean(rr_intervals)
    min_rr = np.min(rr_intervals)
    max_rr = np.max(rr_intervals)
    
    # Triangular index
    hist, _ = np.histogram(rr_intervals, bins=100)
    tri_index = len(rr_intervals) / np.max(hist) if np.max(hist) > 0 else 0

    # Shannon entropy
    probs = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
    entropy = -np.sum(probs * np.log2(probs + 1e-8))  # add epsilon to avoid log(0)
    
    return [rmssd, sdnn, nn50, pnn50, mean_rr, min_rr, max_rr, tri_index, entropy]

def extract_rr_intervals(record):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # ms
    return rr_intervals, ann.symbol[1:]

def prepare_hrv_data(records, window_size=10):
    features, labels = [], []
    normal_symbols = ['N', 'L', 'R', 'e', 'j']
    
    for record in records:
        try:
            rr_intervals, symbols = extract_rr_intervals(record)
        except Exception as e:
            print(f"[Skip] {record}: {e}")
            continue

        for i in range(len(rr_intervals) - window_size):
            window_rr = rr_intervals[i:i + window_size]
            feats = compute_hrv_features(window_rr)
            label = 0 if symbols[i + window_size] in normal_symbols else 1
            features.append(feats)
            labels.append(label)

    return np.array(features), np.array(labels)

def train_hrv_model():
    print("📥 Extracting HRV features...")
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + list(range(205, 206)) + list(range(207, 211)) +
          list(range(212, 216)) + list(range(217, 218)) + list(range(219, 224)) +
          list(range(228, 229)) + list(range(230, 235))]
    ]

    X, y = prepare_hrv_data(records)
    print(f"✅ Extracted {len(X)} samples.")

    print("📊 Scaling and balancing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print("🔍 Running hyperparameter tuning...")
    params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [1]  # balanced dataset via SMOTE
    }

    clf = XGBClassifier(eval_metric='logloss')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(clf, params, scoring='f1', cv=skf, n_jobs=-1, verbose=0)
    grid.fit(X_resampled, y_resampled)

    print("🏆 Best Parameters:", grid.best_params_)
    best_model = grid.best_estimator_

    preds = best_model.predict(X_resampled)
    acc = accuracy_score(y_resampled, preds)
    print("✅ Training Accuracy:", acc)
    print(classification_report(y_resampled, preds, zero_division=0))
    print(confusion_matrix(y_resampled, preds))

    model_dir = "../CardioVision/models/heartratevariability"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "xgb_hrv_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("✅ Model and scaler saved.")

if __name__ == "__main__":
    train_hrv_model()
