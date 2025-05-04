import os
import wfdb
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import find_peaks

def extract_rr_intervals(ecg_signal, fs):
    peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)
    rr_intervals = np.diff(peaks) / fs * 1000  # in ms
    return rr_intervals

def compute_rhr(rr_intervals, window_size=60, fs=360):
    rhr_values = []
    step = window_size * fs
    for i in range(0, len(rr_intervals) - step, step):
        segment = rr_intervals[i:i + step]
        if len(segment) > 0:
            mean_hr = 60000 / np.mean(segment)
            std_hr = np.std(60000 / segment)
            if std_hr < 5:
                rhr_values.append(mean_hr)
    return rhr_values

def label_rhr_values(rhr_values, threshold=75):
    return [1 if hr > threshold else 0 for hr in rhr_values]

def get_all_segments(record_dir):
    segments = set()
    for fname in os.listdir(record_dir):
        if fname.endswith('.dat') and '_' in fname:
            base = fname.split('.dat')[0]
            if os.path.exists(os.path.join(record_dir, base + '.hea')):
                segments.add(base)
    return sorted(list(segments))

def process_records(record_ids, base_path, label_threshold=75):
    features, labels = [], []
    for rec_id in record_ids:
        record_dir = os.path.join(base_path, rec_id)
        if not os.path.isdir(record_dir):
            print(f"[Skip] {record_dir} does not exist.")
            continue

        segments = get_all_segments(record_dir)
        for segment in segments:
            path = os.path.join(record_dir, segment)
            try:
                rec = wfdb.rdrecord(path)
                ecg = rec.p_signal[:, 0]
                rr_intervals = extract_rr_intervals(ecg, rec.fs)
                rhr_vals = compute_rhr(rr_intervals, fs=rec.fs)
                rhr_labels = label_rhr_values(rhr_vals, threshold=label_threshold)
                features.extend(np.array(rhr_vals).reshape(-1, 1))
                labels.extend(rhr_labels)
            except Exception as e:
                print(f"[Skip] {segment}: {e}")
    return np.array(features), np.array(labels)

def test_rhr_model():
    base_path = "../CardioVision/data/mimic3wdb/1.0/30"
    record_dirs = [
        "3000003", "3000031", "3000051", "3000060", "3000063", "3000065", "3000086",
        "3000100", "3000103", "3000105", "3000125", "3000126", "3000142", "3000154",
        "3000189", "3000190", "3000203", "3000221", "3000282", "3000336", "3000358",
        "3000386", "3000393", "3000397", "3000428", "3000435", "3000458", "3000480",
        "3000484", "3000497", "3000531", "3000544", "3000577", "3000596", "3000598",
        "3000611", "3000686", "3000701", "3000710", "3000713", "3000714", "3000715",
        "3000716", "3000717", "3000724", "3000762", "3000775", "3000781", "3000801",
        "3000834", "3000847", "3000855", "3000858", "3000860", "3000866", "3000878",
        "3000879", "3000885", "3000912", "3000960", "3000983", "3000995", "3001002"
    ]

    print("📥 Extracting RHR values from test records...")
    X, y = process_records(record_dirs, base_path)

    print(f"✅ Extracted {len(X)} test samples.")

    model_path = "../CardioVision/models/restingheartrate/rhr_model.pkl"
    scaler_path = "../CardioVision/models/restingheartrate/scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ Model or scaler not found. Please run train_rhr.py first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    acc = accuracy_score(y, preds)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("📋 Classification Report:\n", classification_report(y, preds))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y, preds))

if __name__ == "__main__":
    test_rhr_model()
