import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import wfdb
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.integrate import trapezoid
import warnings

warnings.filterwarnings("ignore")

def compute_advanced_hrv_features(rr_intervals):
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0
    mean_rr = np.mean(rr_intervals)
    min_rr = np.min(rr_intervals)
    max_rr = np.max(rr_intervals)
    skew_rr = skew(rr_intervals)
    kurt_rr = kurtosis(rr_intervals)

    fxx, pxx = welch(rr_intervals, fs=4.0, nperseg=min(len(rr_intervals), 256))
    lf_band = (fxx >= 0.04) & (fxx < 0.15)
    hf_band = (fxx >= 0.15) & (fxx < 0.4)
    lf = trapezoid(pxx[lf_band], fxx[lf_band]) if np.any(lf_band) else 0
    hf = trapezoid(pxx[hf_band], fxx[hf_band]) if np.any(hf_band) else 1
    lf_hf_ratio = lf / hf if hf > 0 else 0

    return [
        rmssd, sdnn, nn50, pnn50, mean_rr, min_rr,
        max_rr, skew_rr, kurt_rr, lf_hf_ratio
    ]

def extract_rr_intervals(record):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000
    return rr_intervals, ann.symbol[1:]

def prepare_hrv_data(records, window_size=10, stride=3):
    features, labels = [], []
    normal_symbols = ['N', 'L', 'R', 'e', 'j']

    for record in records:
        try:
            rr_intervals, symbols = extract_rr_intervals(record)
        except Exception:
            continue

        for i in range(0, len(rr_intervals) - window_size, stride):
            window_rr = rr_intervals[i:i + window_size]
            feats = compute_advanced_hrv_features(window_rr)
            label = 0 if symbols[i + window_size] in normal_symbols else 1
            features.append(feats)
            labels.append(label)

    return np.array(features), np.array(labels)

# Full record set
records = [
    *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
    *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
       list(range(212, 216)) + [217] + list(range(219, 224)) +
       [228] + list(range(230, 235))]
]

print("🔄 Extracting features...")
X, y = prepare_hrv_data(records)
print(f"✅ Extracted {len(X)} samples.")
if len(X) == 0:
    raise RuntimeError("❌ No features were extracted. Check RR extraction or record loading.")

# Scale and resample
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train LightGBM with recall optimization
lgbm = lgb.LGBMClassifier(verbose=-1)
params = {
    'num_leaves': [15, 31],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 200],
    'max_depth': [3, 5, -1]
}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(lgbm, params, n_iter=8, scoring='recall', cv=skf, verbose=1, n_jobs=-1)
search.fit(X_resampled, y_resampled)

# Evaluate
best_model = search.best_estimator_
preds = best_model.predict(X_resampled)
acc = accuracy_score(y_resampled, preds)
clf_report = classification_report(y_resampled, preds, zero_division=0)
conf_mat = confusion_matrix(y_resampled, preds)

print(f"\n✅ Accuracy: {acc:.4f}")
print("📋 Classification Report:\n", clf_report)
print("📊 Confusion Matrix:\n", conf_mat)

# Save
os.makedirs("../CardioVision/models/heartratevariability", exist_ok=True)
joblib.dump(best_model, "../CardioVision/models/heartratevariability/lgb_hrv_model.pkl")
joblib.dump(scaler, "../CardioVision/models/heartratevariability/scaler.pkl")
print("💾 Model and scaler saved.")
