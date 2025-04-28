import os
import json
import numpy as np
import wfdb
from scipy.signal import find_peaks

def compute_hr_from_ecg(record_id, base_path='../CardioVision/data/mitdb'):
    """
    Compute instantaneous HR from R-peaks for a record without an 'HR' channel.
    """
    record_path = os.path.join(base_path, record_id)
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')  # Read annotations if available

    if 'MLII' in rec.sig_name:
        ecg_signal = rec.p_signal[:, rec.sig_name.index('MLII')]
    else:
        ecg_signal = rec.p_signal[:, 0]  # fallback to first channel

    fs = rec.fs

    if ann:
        r_peaks = ann.sample
    else:
        # Very rough R-peak detection if annotations missing (not optimal)
        peaks, _ = find_peaks(ecg_signal, distance=fs*0.6)  # at least 600ms between beats
        r_peaks = peaks

    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    hr_series = 60 / (rr_intervals + 1e-6)  # bpm

    return hr_series

def train_hr2(records, threshold=160,
              model_path='../CardioVision/models/heartrate/hr_model2.json'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save threshold rule
    with open(model_path, 'w') as f:
        json.dump({'threshold': threshold}, f)
    print(f"✅ Saved fallback HR model to {model_path}")

    # Collect HR data
    all_hr = []
    for rec in records:
        try:
            hr = compute_hr_from_ecg(rec)
            print(f"[Train2] Record {rec}: {len(hr)} samples, Mean HR = {np.mean(hr):.2f} bpm")
            all_hr.append(hr)
        except Exception as e:
            print(f"[Skip] {rec}: {e}")

    all_hr = np.concatenate(all_hr)
    hr_labels = (all_hr > threshold).astype(int)

    np.save('../CardioVision/models/heartrate/hr_values_train2.npy', all_hr)
    np.save('../CardioVision/models/heartrate/hr_labels_train2.npy', hr_labels)
    print(f"✅ Saved fallback training HR values and labels.")

if __name__ == '__main__':
    training_records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))]
    ]
    train_hr2(training_records)
