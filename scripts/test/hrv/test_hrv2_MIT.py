import os
import wfdb
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.integrate import trapezoid
from sklearn.metrics import (
    classification_report, confusion_matrix
)

# ---------- Load LightGBM model and scaler ----------
def load_hrv_model(path='../CardioVision/models/heartratevariability/lgb_hrv_model.pkl'):
    return load(path)

def load_scaler(path='../CardioVision/models/heartratevariability/scaler.pkl'):
    return load(path)

# ---------- Feature extraction ----------
def compute_hrv_features(rr_intervals):
    rr_intervals = np.asarray(rr_intervals)
    if len(rr_intervals) < 3 or np.any(np.isnan(rr_intervals)) or np.all(rr_intervals == rr_intervals[0]):
        return None

    diff_rr = np.diff(rr_intervals)
    if len(diff_rr) == 0 or np.any(np.isnan(diff_rr)):
        return None

    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(diff_rr)
    mean_rr = np.mean(rr_intervals)
    min_rr = np.min(rr_intervals)
    max_rr = np.max(rr_intervals)

    try:
        skew_rr = skew(rr_intervals, bias=False)
        kurt_rr = kurtosis(rr_intervals, bias=False)
    except Exception:
        return None

    fxx, pxx = welch(rr_intervals, fs=4.0, nperseg=min(10, len(rr_intervals)))
    lf = trapezoid(pxx[(fxx >= 0.04) & (fxx < 0.15)], fxx[(fxx >= 0.04) & (fxx < 0.15)]) if np.any((fxx >= 0.04) & (fxx < 0.15)) else 0
    hf = trapezoid(pxx[(fxx >= 0.15) & (fxx < 0.4)], fxx[(fxx >= 0.15) & (fxx < 0.4)]) if np.any((fxx >= 0.15) & (fxx < 0.4)) else 1
    lf_hf_ratio = lf / hf if hf > 0 else 0

    return [
        rmssd, sdnn, nn50, pnn50, mean_rr, min_rr,
        max_rr, skew_rr, kurt_rr, lf_hf_ratio
    ]

# ---------- Extract RR intervals and symbols ----------
def extract_rr_intervals(record):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000
    return rr_intervals, ann.symbol[1:]

# ---------- Evaluate on one record ----------
def evaluate_record(model, scaler, record, window_size=10):
    try:
        rr_intervals, symbols = extract_rr_intervals(record)
    except Exception as e:
        print(f"[Skip] {record}: {e}")
        return

    normal_symbols = ['N', 'L', 'R', 'e', 'j']
    features, labels = [], []

    for i in range(len(rr_intervals) - window_size):
        window_rr = rr_intervals[i:i + window_size]
        feats = compute_hrv_features(window_rr)
        if feats is not None:
            label = 0 if symbols[i + window_size] in normal_symbols else 1
            features.append(feats)
            labels.append(label)

    if not features:
        print(f"[No Valid Samples] Record {record}")
        return

    X = scaler.transform(np.array(features))
    y_true = np.array(labels)
    y_pred = model.predict(X)

    print(f"\n[Test] Record {record}:")
    print(f"Total: {len(y_true)}  |  Arrhythmic: {sum(y_true)}  |  Predicted: {sum(y_pred)}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# ---------- Main ----------
def main():
    model = load_hrv_model()
    scaler = load_scaler()

    test_records = [
        *[str(i) for i in list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
           list(range(212, 216)) + [217] + list(range(219, 224)) +
           [228] + list(range(230, 235))]
    ]

    for record in test_records:
        evaluate_record(model, scaler, record)

if __name__ == "__main__":
    main()
