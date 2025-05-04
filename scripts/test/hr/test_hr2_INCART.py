import os
import json
import numpy as np
import wfdb
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, confusion_matrix

# Load fallback HR threshold model
def load_hr_threshold(model_path='../CardioVision/models/heartrate/hr_model2.json'):
    with open(model_path, 'r') as f:
        config = json.load(f)
    return config['threshold']

# Compute HR series from ECG (R-peaks)
def compute_hr_from_ecg(record_id, base_path='../CardioVision/data/incart/files'):
    record_path = os.path.join(base_path, record_id)
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    if 'MLII' in rec.sig_name:
        ecg_signal = rec.p_signal[:, rec.sig_name.index('MLII')]
    else:
        ecg_signal = rec.p_signal[:, 0]  # fallback

    fs = rec.fs

    if ann:
        r_peaks = ann.sample
    else:
        peaks, _ = find_peaks(ecg_signal, distance=fs*0.6)
        r_peaks = peaks

    rr_intervals = np.diff(r_peaks) / fs
    hr_series = 60 / (rr_intervals + 1e-6)
    return hr_series

# Evaluate single record
def evaluate_record(record_id, threshold, base_path):
    try:
        hr_series = compute_hr_from_ecg(record_id, base_path)
    except Exception as e:
        print(f"[Skip] {record_id}: {e}")
        return

    y_true = (hr_series > threshold).astype(int)
    y_pred = (hr_series > threshold).astype(int)

    print(f"\n[Test] Record {record_id}")
    print(f"Windows: {len(hr_series)} | High HR Events (True): {np.sum(y_true)}")

    print(classification_report(y_true, y_pred, labels=[0, 1], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

# Main
def main():
    threshold = load_hr_threshold()
    base_path = '../CardioVision/data/incart/files'

    test_records = [
        fname.split('.')[0] for fname in os.listdir(base_path) if fname.endswith('.dat')
    ]

    for record in sorted(set(test_records)):
        evaluate_record(record, threshold, base_path)

if __name__ == "__main__":
    main()
