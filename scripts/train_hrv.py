import os
import wfdb
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def compute_hrv_features(rr_intervals):
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    sdnn = np.std(rr_intervals)
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(diff_rr) if len(diff_rr) > 0 else 0
    mean_rr = np.mean(rr_intervals)
    return [rmssd, sdnn, nn50, pnn50, mean_rr]

def extract_rr_intervals(record):
    rec = wfdb.rdrecord(f'../CardioVision/data/mitdb/{record}')
    ann = wfdb.rdann(f'../CardioVision/data/mitdb/{record}', 'atr')
    r_peaks = ann.sample
    rr_intervals = np.diff(r_peaks) / rec.fs * 1000
    return rr_intervals, ann.symbol[1:]

def prepare_hrv_data(records, window_size=10):
    features = []
    labels = []
    normal_symbols = ['N', 'L', 'R', 'e', 'j']

    for record in records:
        try:
            rr_intervals, symbols = extract_rr_intervals(record)
        except Exception as e:
            print(f"Skipping {record}: {e}")
            continue

        for i in range(len(rr_intervals) - window_size):
            window_rr = rr_intervals[i:i + window_size]
            feats = compute_hrv_features(window_rr)
            label = 0 if symbols[i + window_size] in normal_symbols else 1
            features.append(feats)
            labels.append(label)

    return np.array(features), np.array(labels)


def train_hrv_model():
    records = [*map(str, list(range(100,110)) + list(range(111,120)) + list(range(121,125))
                    + list(range(200,204)) + [205] + list(range(207,211))
                    + list(range(212,216)) + [217] + list(range(219,224))
                    + [228] + list(range(230,235)))]
    X, y = prepare_hrv_data(records)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=pos_weight)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    
    os.makedirs('../CardioVision/models/heartratevariability', exist_ok=True)
    joblib.dump(model, '../CardioVision/models/heartratevariability/xgb_hrv_model.pkl')
    joblib.dump(scaler, '../CardioVision/models/heartratevariability/scaler.pkl')
    print("Saved model and scaler.")

if __name__ == "__main__":
    train_hrv_model()
