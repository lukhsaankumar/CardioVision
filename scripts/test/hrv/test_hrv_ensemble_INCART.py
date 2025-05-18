"""
Heart Rate Variability (HRV) Ensemble Model - INCART Evaluation
---------------------------------------------------------------
This script tests the HRV ensemble model using the INCART dataset, leveraging RR interval features 
to detect arrhythmias.

Description:
- Loads a pre-trained HRV ensemble model (XGBoost + LightGBM + CatBoost) and associated scaler.
- Extracts HRV features from RR intervals of ECG records in the INCART dataset.
- Classifies each window of RR intervals as normal or arrhythmic.
- Evaluates the model using classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Results are displayed in the console and can be found at:
  testresults/incart/INCART_HRV.txt
"""

from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import welch
from scipy.integrate import trapezoid
from joblib import load
import numpy as np
import wfdb
import os

# -------------------------------
# MODEL LOADING
# -------------------------------
def load_hrv_model(path='../CardioVision/models/heartratevariability/hrv_ensemble_model.pkl'):
    """
    Loads the HRV ensemble model (XGBoost + LightGBM).
    """
    return load(path)

def load_scaler(path='../CardioVision/models/heartratevariability/scaler.pkl'):
    """
    Loads the scaler used for feature normalization.
    """
    return load(path)

# -------------------------------
# HRV FEATURE EXTRACTION
# -------------------------------
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
    hf = trapezoid(Pxx[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]) if len(Pxx) else 1
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

# -------------------------------
# RR INTERVAL EXTRACTION
# -------------------------------
def extract_rr_intervals(record):
    """
    Extracts RR intervals from a given INCART ECG record.

    Args:
        record (str): Record filename (without extension).

    Returns:
        tuple: (RR intervals, annotation symbols)
    """
    rec = wfdb.rdrecord(f'../CardioVision/data/incart/files/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/incart/files/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000  # Convert to milliseconds
    return rr_intervals, ann.symbol[1:]

# -------------------------------
# MODEL EVALUATION
# -------------------------------
def evaluate_record(model, scaler, record, window_size=10):
    """
    Evaluates the HRV model on a single INCART record.

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

    for i in range(len(rr_intervals) - window_size):
        window_rr = rr_intervals[i:i + window_size]
        feats = compute_hrv_features(window_rr)
        label = 0 if symbols[i + window_size] in normal_symbols else 1
        features.append(feats)
        labels.append(label)

    if not features:
        print(f"[No Valid Samples] Record {record}")
        return

    X = scaler.transform(np.array(features))
    y_true = np.array(labels)
    y_pred = model.predict(X)

    print(f"\n[Test] INCART Record {record}:")
    print(f"Total: {len(y_true)}  |  Arrhythmic: {sum(y_true)}  |  Predicted: {sum(y_pred)}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# -------------------------------
# MAIN
# -------------------------------
def main():
    """
    Main function to test the HRV model on multiple INCART records.
    """
    model = load_hrv_model()
    scaler = load_scaler()
    incart_records = [f'I{i}' for i in range(1, 75)]

    print("\nTesting HRV Ensemble Model on INCART Dataset\n")

    for record in incart_records:
        evaluate_record(model, scaler, record)

if __name__ == "__main__":
    main()
