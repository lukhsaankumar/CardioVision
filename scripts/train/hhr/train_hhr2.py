import os
import wfdb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def extract_hr_from_beats(record, window_size=10, threshold_bpm=150):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    fs = rec.fs

    features = []
    labels = []

    rr_intervals = np.diff(r_peaks) / fs * 1000  # RR intervals in ms
    hr_series = 60000 / rr_intervals  # Heart rate in bpm

    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        duration_above = np.sum(window > threshold_bpm)
        sustained = int(np.all(window > threshold_bpm))
        max_hr = np.max(window)
        min_hr = np.min(window)
        avg_hr = np.mean(window)
        hr_slope = (window[-1] - window[0]) / window_size
        spike_freq = np.sum(np.diff(window) > 10)
        hr_stddev = np.std(window)  # NEW: standard deviation

        feats = [
            duration_above, sustained, max_hr, min_hr, avg_hr, hr_slope,
            spike_freq, hr_stddev
        ]
        label = 1 if sustained else 0
        features.append(feats)
        labels.append(label)

    return features, labels

def prepare_dataset(records):
    all_features = []
    all_labels = []
    for record in records:
        try:
            feats, labels = extract_hr_from_beats(record)
            all_features.extend(feats)
            all_labels.extend(labels)
        except Exception as e:
            print(f"[Skip] {record}: {e}")
    return np.array(all_features), np.array(all_labels)

def train_hhr2():
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
          list(range(212, 216)) + [217] + list(range(219, 224)) + [228] + list(range(230, 235))]
    ]

    print("📥 Extracting features from MIT-BIH...")
    X, y = prepare_dataset(records)
    print(f"✅ Extracted {len(y)} samples.")
    print(f"🔎 Label Distribution: {np.bincount(y)}")

    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🔄 Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    print(f"🔎 New Balanced Label Distribution: {np.bincount(y_balanced)}")

    print("📈 Training Random Forest model for HHR detection...")
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, stratify=y_balanced, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("✅ Model Performance:")
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs('../CardioVision/models/highheartrateevents', exist_ok=True)
    joblib.dump(model, '../CardioVision/models/highheartrateevents/rf_hhr2_model.pkl')
    joblib.dump(scaler, '../CardioVision/models/highheartrateevents/scaler.pkl')
    print("✅ Improved model and scaler saved to models/highheartrateevents/")

if __name__ == "__main__":
    train_hhr2()
