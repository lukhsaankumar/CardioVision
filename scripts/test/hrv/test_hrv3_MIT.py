"""
CatBoost HRV Model - MIT-BIH Evaluation
----------------------------------------
This script tests the CatBoost HRV model using the MIT-BIH dataset.

Description:
- Loads a pre-trained CatBoost model for HRV-based arrhythmia detection.
- Extracts HRV features (14 features) from RR intervals of ECG records in the MIT-BIH dataset.
- Classifies each window of RR intervals as normal or arrhythmic.
- Evaluates the model using classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/mitbih/MITBIH_HRV3.txt
"""

import os
import wfdb
import numpy as np
from joblib import load
from scipy.signal import welch
from scipy.integrate import trapezoid
from sklearn.metrics import classification_report, confusion_matrix

# ---------- Load CatBoost model and scaler ----------
def load_hrv_model(path='../CardioVision/models/heartratevariability/catboost_hrv_model.pkl'):
    """
    Loads the CatBoost HRV model.
    """
    return load(path)

def load_scaler(path='../CardioVision/models/heartratevariability/scaler.pkl'):
    """
    Loads the scaler for HRV feature normalization.
    """
    return load(path)

# ---------- Feature extraction (14 features for CatBoost) ----------
def compute_hrv_features(rr_intervals):
    """
    Computes HRV features from RR intervals for model input.

    Args:
        rr_intervals (array): Array of RR intervals (in ms).

    Returns:
        list: HRV feature vector (14 features).
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

# ---------- Extract RR intervals and symbols ----------
def extract_rr_intervals(record):
    """
    Extracts RR intervals and symbols from a given MIT-BIH ECG record.

    Args:
        record (str): Record filename (without extension).

    Returns:
        tuple: (RR intervals, annotation symbols)
    """
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # Convert to milliseconds
    return rr_intervals, ann.symbol[1:]

# ---------- Evaluate on one record ----------
def evaluate_record(model, scaler, record, window_size=30):
    """
    Evaluates the CatBoost HRV model on a single MIT-BIH record.

    Args:
        model: Loaded HRV model.
        scaler: Loaded scaler for feature normalization.
        record (str): Record filename.
        window_size (int): Size of the RR window for feature extraction.
    """
    try:
        rr_intervals, symbols = extract_rr_intervals(record)
    except Exception as e:
        print(f"[Skip] {record}: {e}")
        return

    normal_symbols = ['N', 'L', 'R', 'e', 'j']
    features, labels = [], []

    for i in range(0, len(rr_intervals) - window_size):
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
    """
    Main function to test the CatBoost HRV model on multiple MIT-BIH records.
    """
    model = load_hrv_model()
    scaler = load_scaler()

    test_records = [
        *[str(i) for i in list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
           list(range(212, 216)) + [217] + list(range(219, 224)) +
           [228] + list(range(230, 235))]
    ]

    print("\nTesting HRV CatBoost Model on MIT-BIH Dataset\n")

    for record in test_records:
        evaluate_record(model, scaler, record)

if __name__ == "__main__":
    main()
