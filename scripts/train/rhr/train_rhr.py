# Updated Python code for training RHR model using MIMIC record folder "31"

import os
import wfdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import find_peaks

def extract_rr_intervals(ecg_signal, fs):
    peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)
    rr_intervals = np.diff(peaks) / fs * 1000  # ms
    return rr_intervals

def compute_rhr(rr_intervals, window_size=60, fs=360):
    rhr_values = []
    step = window_size
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

def train_rhr_model():
    base_path = "../CardioVision/data/mimic3wdb/1.0/31"
    train_records = [
        "3100011", "3100033", "3100038", "3100069", "3100101", "3100105", "3100112",
        "3100119", "3100124", "3100132", "3100140", "3100156", "3100165", "3100181",
        "3100196", "3100198", "3100209", "3100237", "3100240", "3100288", "3100305",
        "3100308", "3100312", "3100329", "3100331", "3100340", "3100399", "3100418",
        "3100438", "3100442", "3100461", "3100486", "3100503", "3100524", "3100525",
        "3100538", "3100566", "3100568", "3100574", "3100594", "3100611", "3100618",
        "3100626", "3100643", "3100644", "3100669", "3100673", "3100677", "3100705",
        "3100712", "3100721", "3100733", "3100745", "3100754", "3100757", "3100799",
        "3100822", "3100826", "3100827"
    ]

    print("📥 Extracting RHR values...")
    X, y = process_records(train_records, base_path)

    print(f"✅ Extracted {len(X)} samples.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("✅ Accuracy:", acc)
    print(classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    os.makedirs("../CardioVision/models/restingheartrate", exist_ok=True)
    joblib.dump(model, "../CardioVision/models/restingheartrate/rhr_model.pkl")
    joblib.dump(scaler, "../CardioVision/models/restingheartrate/scaler.pkl")
    print("✅ Model and scaler saved.")

if __name__ == "__main__":
    train_rhr_model()
