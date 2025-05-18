"""
HRV Ensemble Model Training
----------------------------
This script trains an ensemble model for Heart Rate Variability (HRV) classification using an ensemble approach (XGBoost, LightGBM, CatBoost, and a Logistic Regression meta-classifier).

Description:
- Extracts HRV features from ECG records (RR intervals).
- Trains an ensemble model using three base learners:
  - XGBoost
  - LightGBM
  - CatBoost
- Combines their predictions using a Logistic Regression meta-classifier (Stacking).
- Evaluates model performance on the training set.
- Saves the trained model and scaler for future use.

Dataset:
- Source: MIT-BIH Arrhythmia Database (MITDB)
- Records used: 100-109, 111-119, 121-124, 200-203, 205, 207-210, 212-215, 217, 219-223, 228, 230-234
- Features: HRV features extracted from RR intervals (14 features).

Results:
- The model (stacked classifier) is saved at:
  ../CardioVision/models/heartratevariability/hrv_ensemble_model.pkl
- The scaler is saved at:
  ../CardioVision/models/heartratevariability/scaler.pkl
"""

import os
import numpy as np
import joblib
import wfdb
from scipy.signal import welch
from scipy.integrate import trapezoid
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# HRV feature computation
def compute_hrv_features(rr_intervals):
    """
    Compute 14 HRV features from RR intervals.

    Args:
        rr_intervals (np.array): Array of RR intervals in milliseconds.

    Returns:
        list: 14 HRV features (time-domain and frequency-domain).
    """
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    if len(rr_intervals) < 2:
        return [0] * 14

    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) else 0
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(diff_rr)

    f, Pxx = welch(rr_intervals, fs=4.0, nperseg=len(rr_intervals))
    lf = trapezoid(Pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]) if len(Pxx) else 0
    hf = trapezoid(Pxx[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]) if len(Pxx) else 0
    lf_hf = lf / hf if hf > 0 else 0

    sd1 = np.sqrt(0.5 * np.var(diff_rr))
    sd2 = np.sqrt(2 * sdnn ** 2 - sd1 ** 2) if 2 * sdnn ** 2 > sd1 ** 2 else 0

    hist, _ = np.histogram(rr_intervals, bins=50, density=True)
    probs = hist / np.sum(hist) if np.sum(hist) else np.zeros_like(hist)
    shannon = -np.sum(probs * np.log2(probs + 1e-8))

    cvnni = sdnn / np.mean(rr_intervals) if np.mean(rr_intervals) else 0
    tinn = np.max(hist)
    median_rr = np.median(rr_intervals)

    return [rmssd, sdnn, nn50, pnn50, lf, hf, lf_hf, sd1, sd2, shannon, cvnni, tinn, median_rr, len(rr_intervals)]

# Extract RR intervals from an ECG record
def extract_rr_intervals(record):
    """
    Extract RR intervals and symbols from an ECG record.

    Args:
        record (str): Record ID (filename without extension).

    Returns:
        np.array: RR intervals (ms).
        list: Annotation symbols (e.g., 'N', 'V').
    """
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # Convert to milliseconds
    return rr_intervals, ann.symbol[1:]

# Prepare HRV dataset (feature extraction)
def prepare_hrv_data(records, window=30, step=5):
    """
    Extract HRV features and labels from multiple ECG records.

    Args:
        records (list): List of record IDs.
        window (int): Number of RR intervals per feature set.
        step (int): Step size for window sliding.

    Returns:
        np.array: Feature matrix (HRV features).
        np.array: Labels (0 = Normal, 1 = Arrhythmic).
    """
    feats, labels = [], []
    normal_syms = ['N', 'L', 'R', 'e', 'j']
    for rec_id in records:
        try:
            rr, syms = extract_rr_intervals(rec_id)
        except Exception as e:
            print(f"[Skip] {rec_id}: {e}")
            continue
        
        for i in range(0, len(rr) - window, step):
            window_rr = rr[i:i + window]
            feature_vec = compute_hrv_features(window_rr)
            feats.append(feature_vec)
            labels.append(0 if syms[i + window] in normal_syms else 1)
    
    return np.array(feats), np.array(labels)

# Train the ensemble HRV model
def train_ensemble_model():
    print("Extracting HRV features...")
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
           list(range(212, 216)) + [217] + list(range(219, 224)) +
           [228] + list(range(230, 235))]
    ]
    X, y = prepare_hrv_data(records)
    print(f"Extracted {len(X)} samples.")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training ensemble model...")
    xgb = XGBClassifier(tree_method='hist', eval_metric='logloss')
    lgbm = LGBMClassifier()
    catboost = CatBoostClassifier(verbose=0)
    meta_model = LogisticRegression()

    stack_clf = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)],
        final_estimator=meta_model,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )

    stack_clf.fit(X_scaled, y)

    # Save the trained model and scaler
    os.makedirs("../CardioVision/models/heartratevariability", exist_ok=True)
    joblib.dump(stack_clf, "../CardioVision/models/heartratevariability/hrv_ensemble_model.pkl")
    joblib.dump(scaler, "../CardioVision/models/heartratevariability/scaler.pkl")
    print("✅ Ensemble model and scaler saved.")

if __name__ == "__main__":
    train_ensemble_model()
