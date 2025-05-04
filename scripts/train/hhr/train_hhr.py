import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

def extract_hr_from_beats(record, window_size=10, threshold_bpm=160, required_spikes=3):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    fs = rec.fs

    features = []
    labels = []

    rr_intervals = np.diff(r_peaks) / fs * 1000  # RR in ms
    hr_series = 60000 / rr_intervals  # bpm

    for i in range(len(hr_series) - window_size):
        window = hr_series[i:i + window_size]
        duration_above = np.sum(window > threshold_bpm)
        sustained = 0
        count = 0
        for hr in window:
            if hr > threshold_bpm:
                count += 1
                if count >= required_spikes:
                    sustained = 1
                    break
            else:
                count = 0
        max_hr = np.max(window)
        min_hr = np.min(window)
        avg_hr = np.mean(window)
        hr_slope = (window[-1] - window[0]) / window_size
        spike_freq = np.sum(np.diff(window) > 10)

        feats = [duration_above, sustained, max_hr, min_hr, avg_hr, hr_slope, spike_freq]
        features.append(feats)
        labels.append(sustained)

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

def train_hhr_model():
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
          list(range(212, 216)) + [217] + list(range(219, 224)) + [228] + list(range(230, 235))]
    ]

    print("📥 Extracting features from MIT-BIH...")
    X, y = prepare_dataset(records)
    print(f"✅ Extracted {len(y)} samples.")
    print("🔎 Label Distribution:", np.bincount(y))

    # Optional: Visualize class distribution
    plt.hist(y, bins=2)
    plt.title("Class Distribution: High HR Events")
    plt.xticks([0, 1], ['Normal', 'High HR'])
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()

    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("📈 Training XGBoost model for HHR detection...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("✅ Model Performance:")
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs('../CardioVision/models/highheartrateevents', exist_ok=True)
    joblib.dump(model, '../CardioVision/models/highheartrateevents/xgb_hhr_model.pkl')
    joblib.dump(scaler, '../CardioVision/models/highheartrateevents/scaler.pkl')
    print("✅ Model and scaler saved to models/highheartrateevents/")

train_hhr_model()
