"""
Enhanced HRV XGBoost Model Training
------------------------------------
This script trains an XGBoost model for Heart Rate Variability (HRV) classification using enhanced HRV features and hyperparameter tuning.

Description:
- Extracts HRV features from ECG records (RR intervals).
- Enhances features with both time-domain and frequency-domain metrics:
  - Time-domain: RMSSD, SDNN, NN50, pNN50
  - Frequency-domain: LF, HF, LF/HF ratio
  - Nonlinear: SD1, SD2 (Poincaré), Shannon Entropy
  - Additional: CVNNI, TINN, Median RR
- Uses XGBoost for classification with hyperparameter tuning (RandomizedSearchCV).
- Implements GroupKFold cross-validation to prevent data leakage.
- Saves the trained model and scaler for future use.

Dataset:
- Source: MIT-BIH Arrhythmia Database (MITDB)
- Records used: 100-109, 111-119, 121-124, 200-203, 205, 207-210, 212-215, 217, 219-223, 228, 230-234
- Features: Enhanced HRV features extracted from RR intervals.

Results:
- The trained XGBoost model is saved at:
  ../CardioVision/models/heartratevariability/xgb_hrv_model.pkl
- The scaler is saved at:
  ../CardioVision/models/heartratevariability/scaler.pkl
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import wfdb

# Enhanced HRV features computation
def compute_hrv_features(rr_intervals):
    """
    Compute enhanced HRV features from RR intervals.
    
    Args:
        rr_intervals (np.array): Array of RR intervals in milliseconds.
    
    Returns:
        list: 14 HRV features (time-domain, frequency-domain, nonlinear).
    """
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]  # filter outliers
    if len(rr_intervals) < 2:
        return [0] * 14

    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) else 0
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(diff_rr)

    # Frequency-domain (Welch method)
    f, Pxx = welch(rr_intervals, fs=4.0, nperseg=len(rr_intervals))
    lf = np.trapezoid(Pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]) if len(Pxx) else 0
    hf = np.trapezoid(Pxx[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]) if len(Pxx) else 0
    lf_hf = lf / hf if hf > 0 else 0

    # Nonlinear features (Poincaré)
    sd1 = np.sqrt(0.5 * np.var(diff_rr))
    sd2 = np.sqrt(2 * sdnn ** 2 - sd1 ** 2) if 2 * sdnn ** 2 > sd1 ** 2 else 0

    # Shannon entropy
    hist, _ = np.histogram(rr_intervals, bins=50, density=True)
    probs = hist / np.sum(hist) if np.sum(hist) else np.zeros_like(hist)
    shannon = -np.sum(probs * np.log2(probs + 1e-8))

    # Additional features
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

# Prepare HRV dataset with enhanced features
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
        np.array: Record groups for GroupKFold.
    """
    feats, labels, groups = [], [], []
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
            groups.append(rec_id)
    
    return np.array(feats), np.array(labels), np.array(groups)

# Train the XGBoost HRV model
def train_model():
    print("Extracting features...")
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
           list(range(212, 216)) + [217] + list(range(219, 224)) +
           [228] + list(range(230, 235))]
    ]
    X, y, groups = prepare_hrv_data(records)
    print(f"Extracted {len(X)} samples.")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost with GroupKFold (no data leakage)
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist')
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=10,
        cv=GroupKFold(n_splits=5).split(X_scaled, y, groups),
        scoring='f1', n_jobs=-1, verbose=1
    )

    search.fit(X_scaled, y)
    best_model = search.best_estimator_

    # Save model and scaler
    os.makedirs("../CardioVision/models/heartratevariability", exist_ok=True)
    joblib.dump(best_model, "../CardioVision/models/heartratevariability/xgb_hrv_model.pkl")
    joblib.dump(scaler, "../CardioVision/models/heartratevariability/scaler.pkl")
    print("✅ Model and scaler saved.")

if __name__ == "__main__":
    train_model()
