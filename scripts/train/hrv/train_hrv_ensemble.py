import os
import numpy as np
import joblib
import wfdb
from scipy.signal import welch
from scipy.integrate import trapezoid
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# HRV feature computation
def compute_hrv_features(rr_intervals):
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

def extract_rr_intervals(record):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000
    return rr_intervals, ann.symbol[1:]

def prepare_hrv_data(records, window=30, step=5):
    feats, labels = [], []
    normal_syms = ['N', 'L', 'R', 'e', 'j']
    for rec_id in records:
        try:
            rr, syms = extract_rr_intervals(rec_id)
        except Exception as e:
            print(f"[Skip] {rec_id}: {e}")
            continue
        for i in range(0, len(rr) - window, step):
            window_rr = rr[i:i + window]
            feature_vec = compute_hrv_features(window_rr)
            feats.append(feature_vec)
            labels.append(0 if syms[i + window] in normal_syms else 1)
    return np.array(feats), np.array(labels)

def train_ensemble_model():
    print("🔄 Extracting HRV features...")
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
           list(range(212, 216)) + [217] + list(range(219, 224)) +
           [228] + list(range(230, 235))]
    ]
    X, y = prepare_hrv_data(records)
    print(f"✅ Extracted {len(X)} samples.")

    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🧠 Training ensemble model...")

    # Define base models
    xgb = XGBClassifier(tree_method='hist', eval_metric='logloss')
    lgbm = LGBMClassifier()
    catboost = CatBoostClassifier(verbose=0)

    # Meta-model
    meta_model = LogisticRegression()

    # Stacking ensemble
    stack_clf = StackingClassifier(
        estimators=[
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('catboost', catboost)
        ],
        final_estimator=meta_model,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )

    stack_clf.fit(X_scaled, y)

    preds = stack_clf.predict(X_scaled)
    print(f"\n✅ Accuracy: {accuracy_score(y, preds):.4f}")
    print("📋 Classification Report:")
    print(classification_report(y, preds))
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y, preds))

    os.makedirs("../CardioVision/models/heartratevariability", exist_ok=True)
    joblib.dump(stack_clf, "../CardioVision/models/heartratevariability/hrv_ensemble_model.pkl")
    joblib.dump(scaler, "../CardioVision/models/heartratevariability/scaler.pkl")
    print("💾 Ensemble model and scaler saved.")

if __name__ == "__main__":
    train_ensemble_model()
import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wfdb
from scipy.signal import welch
from scipy.integrate import trapezoid

# HRV feature computation (same as before)
def compute_hrv_features(rr_intervals):
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

def extract_rr_intervals(record):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000
    return rr_intervals, ann.symbol[1:]

def prepare_hrv_data(records, window=30, step=5):
    feats, labels = [], []
    normal_syms = ['N', 'L', 'R', 'e', 'j']
    for rec_id in records:
        try:
            rr, syms = extract_rr_intervals(rec_id)
        except Exception as e:
            print(f"[Skip] {rec_id}: {e}")
            continue
        for i in range(0, len(rr) - window, step):
            window_rr = rr[i:i + window]
            feature_vec = compute_hrv_features(window_rr)
            feats.append(feature_vec)
            labels.append(0 if syms[i + window] in normal_syms else 1)
    return np.array(feats), np.array(labels)

def train_meta_classifier():
    print("🔄 Loading HRV data...")
    records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
           list(range(212, 216)) + [217] + list(range(219, 224)) +
           [228] + list(range(230, 235))]
    ]
    X, y = prepare_hrv_data(records)
    print(f"✅ Extracted {len(X)} samples.")

    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("📦 Loading base models...")
    xgb = joblib.load("../CardioVision/models/heartratevariability/xgb_hrv_model.pkl")
    lgb = joblib.load("../CardioVision/models/heartratevariability/lgb_hrv_model.pkl")
    cat = joblib.load("../CardioVision/models/heartratevariability/catboost_hrv_model.pkl")

    print("🔮 Getting prediction probabilities...")
    xgb_probs = xgb.predict_proba(X_scaled)
    lgb_probs = lgb.predict_proba(X_scaled)
    cat_probs = cat.predict_proba(X_scaled)

    print("🔗 Concatenating base predictions...")
    stacked_features = np.hstack([
        xgb_probs,  # shape (n_samples, 2)
        lgb_probs,
        cat_probs
    ])  # shape (n_samples, 6)

    print("🧠 Training meta-classifier (Logistic Regression)...")
    X_train, X_test, y_train, y_test = train_test_split(stacked_features, y, test_size=0.2, random_state=42)
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_train, y_train)

    y_pred = meta_clf.predict(X_test)
    print(f"\n✅ Meta Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("📋 Meta Classification Report:")
    print(classification_report(y_test, y_pred))
    print("📊 Meta Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(meta_clf, "../CardioVision/models/heartratevariability/meta_logistic_hrv.pkl")
    print("💾 Meta-classifier saved.")

if __name__ == "__main__":
    train_meta_classifier()
