from pathlib import Path
import wfdb
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

# --- Helper Functions ---

def extract_rr_intervals(ecg_signal, fs):
    peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)
    rr_intervals = np.diff(peaks) / fs * 1000  # ms
    return rr_intervals

def compute_rhr(rr_intervals, window_rr_count=60):
    rhr_values = []
    step = window_rr_count
    for i in range(0, len(rr_intervals) - step, step):
        segment = rr_intervals[i:i + step]
        if len(segment) > 0:
            mean_hr = 60000 / np.mean(segment)
            std_hr = np.std(60000 / segment)
            if std_hr < 15:  # relaxed std threshold
                rhr_values.append(mean_hr)
    return rhr_values

def label_rhr_values(rhr_values, threshold=75):
    return [1 if hr > threshold else 0 for hr in rhr_values]

def test_rhr_model_incart():
    base_path = Path("../CardioVision/data/incart/files")
    record_ids = [f"I{i:02d}" for i in range(1, 76)]

    features, labels = [], []
    print("🔄 Extracting features...")
    for rec_id in record_ids:
        path = base_path / rec_id
        try:
            rec = wfdb.rdrecord(str(path))
            ecg = rec.p_signal[:, 0]
            print(f"✅ Loaded {rec_id} with {len(ecg)} samples at {rec.fs} Hz")
            rr_intervals = extract_rr_intervals(ecg, rec.fs)
            print(f"🔍 RR intervals extracted: {len(rr_intervals)}")
            rhr_vals = compute_rhr(rr_intervals)
            print(f"✅ RHR values extracted: {len(rhr_vals)}")
            rhr_labels = label_rhr_values(rhr_vals)
            features.extend(np.array(rhr_vals).reshape(-1, 1))
            labels.extend(rhr_labels)
        except Exception as e:
            print(f"[Skip] {rec_id}: {e}")

    if not features:
        print("❌ No valid RHR data extracted. Check ECG signal quality or segment length.")
        return

    X = np.array(features)
    y = np.array(labels)

    model_path = "../CardioVision/models/restingheartrate/rhr_model.pkl"
    scaler_path = "../CardioVision/models/restingheartrate/scaler.pkl"
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        print("❌ Model or scaler not found. Please train it first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    acc = accuracy_score(y, preds)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("📋 Classification Report:\n", classification_report(y, preds))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y, preds))

# Run the test
test_rhr_model_incart()
