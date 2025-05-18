import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import wfdb
import json
import joblib
from tqdm import tqdm
import os
import glob
from imblearn.over_sampling import SMOTE

# BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Focal Loss with Class Weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, class_weights=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else torch.ones(3)
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecg_model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3).to(device)
try:
    ecg_model.load_state_dict(torch.load("../CardioVision/models/ecg/bilstm_model_multiclass.pth", map_location=device))
except FileNotFoundError:
    print("❌ BiLSTM model not found.")
    exit(1)
ecg_model.eval()

# Load submodels for FN analysis
with open("../CardioVision/models/heartrate/hr_model2.json") as f:
    hr_model_data = json.load(f)
hrv_model = joblib.load("../CardioVision/models/heartratevariability/hrv_ensemble_model.pkl")
scaler_hrv = joblib.load("../CardioVision/models/heartratevariability/scaler.pkl")
rhr_model = joblib.load("../CardioVision/models/restingheartrate/rhr_model.pkl")
scaler_rhr = joblib.load("../CardioVision/models/restingheartrate/scaler.pkl")
hhr_model = joblib.load("../CardioVision/models/highheartrateevents/rf_hhr2_model.pkl")

# Data Loading
def load_record(record_path):
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        signal = record.p_signal[:, 0]
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print(f"⚠️ Imputing NaN/Inf in signal for record {os.path.basename(record_path)}")
            signal = np.nan_to_num(signal, nan=np.mean(signal[~np.isnan(signal)]), 
                                   posinf=np.max(signal[~np.isinf(signal)]), 
                                   neginf=np.min(signal[~np.isinf(signal)]))
        return signal, annotation.symbol, annotation.sample
    except Exception as e:
        print(f"⚠️ Skipped {record_path}: {e}")
        return None, None, None

def segment_beat(signal, peak, window_size=250, fs=360):
    start = max(0, peak - window_size // 2)
    end = min(len(signal), peak + window_size // 2)
    segment = signal[start:end]
    if len(segment) < window_size:
        segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')
    return segment[:window_size]

def extract_feedback_samples():
    mit_records = glob.glob("../CardioVision/data/mitdb/*.dat")
    holter_records = [r for r in glob.glob("../CardioVision/data/holter/*.dat") 
                     if not any(x in r for x in ['33', '37', '38', '39', '40', '42', '43', '44', '47', '48', '50'])]
    incart_records = glob.glob("../CardioVision/data/incart/files/*.dat")
    all_records = mit_records + holter_records + incart_records
    low_risk = ['N', 'L', 'R', 'e', 'j']
    med_risk = ['A', 'a', 'J', 'S']
    high_risk = ['V', 'F', 'E']
    symbol_to_label = {s: 0 for s in low_risk}
    symbol_to_label.update({s: 1 for s in med_risk})
    symbol_to_label.update({s: 2 for s in high_risk})

    feedback_samples = []
    fn_submodel_stats = {'hr': [], 'hrv': [], 'rhr': [], 'hhr': []}
    sample_counts = {0: 0, 1: 0, 2: 0}
    max_samples_per_class = 8000
    total_high = 0
    high_fn = 0
    total_fn = 0

    print("🔄 Extracting Feedback Samples...")
    for record_path in tqdm(all_records, desc="Processing Records"):
        signal, symbols, peaks = load_record(record_path.replace('.dat', ''))
        if signal is None:
            continue
        fs = 360
        scaler = StandardScaler()

        hr_series = []
        for i in range(1, len(peaks)):
            rr = (peaks[i] - peaks[i-1]) / fs
            if rr > 0 and np.isfinite(rr):
                hr_series.append(60 / rr)
        hr_series = np.array(hr_series) if hr_series else np.array([60])

        for peak, symbol in zip(peaks, symbols):
            if symbol not in symbol_to_label:
                continue
            label = symbol_to_label[symbol]
            if sample_counts[label] >= max_samples_per_class:
                continue
            if label == 2:
                total_high += 1
            segment = segment_beat(signal, peak, window_size=250, fs=fs)
            if len(segment) != 250:
                continue

            segment_normalized = scaler.fit_transform(segment.reshape(-1, 1)).reshape(-1)
            if np.any(np.isnan(segment_normalized)) or np.any(np.isinf(segment_normalized)):
                continue

            ecg_tensor = torch.tensor(segment_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                ecg_output = torch.softmax(ecg_model(ecg_tensor), dim=-1).cpu().numpy().flatten()
            if np.any(np.isnan(ecg_output)):
                continue
            ecg_pred = np.argmax(ecg_output)
            pred_confidence = ecg_output[label]
            high_confidence = ecg_output[2] if label != 2 else 0

            # Submodel predictions for all samples
            idx = min(len(hr_series) - 1, len(peaks) - 2)
            hr_val = hr_series[idx] if idx >= 0 else 60
            pred_hr = 1 if hr_val > 100 else 0

            sustained_high_hr = int(np.all(hr_series[max(0, idx-10):idx+1] > 130)) if idx > 0 else 0
            pred_hhr = 1 if sustained_high_hr else 0

            rhr = np.mean(hr_series) if len(hr_series) > 0 else 60
            pred_rhr = 1 if rhr > 70 else 0

            rr_intervals = np.diff(peaks) / fs * 1000
            hrv_features = np.zeros(14)
            if len(rr_intervals) >= 2:
                rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
                if len(rr_intervals) >= 2:
                    diff_rr = np.diff(rr_intervals)
                    hrv_features = [
                        np.sqrt(np.mean(diff_rr ** 2)),
                        np.std(rr_intervals),
                        np.mean(rr_intervals),
                        np.median(rr_intervals),
                        np.percentile(rr_intervals, 25),
                        np.percentile(rr_intervals, 75),
                        np.max(diff_rr) if len(diff_rr) > 0 else 0,
                        np.min(diff_rr) if len(diff_rr) > 0 else 0,
                        np.mean(diff_rr) if len(diff_rr) > 0 else 0,
                        np.std(diff_rr) if len(diff_rr) > 0 else 0,
                        np.sum(np.abs(diff_rr) > 50) / len(diff_rr) if len(diff_rr) > 0 else 0,
                        len(rr_intervals),
                        np.min(rr_intervals) if len(rr_intervals) > 0 else 0,
                        np.max(rr_intervals) if len(rr_intervals) > 0 else 0
                    ]
            scaled_hrv = scaler_hrv.transform([hrv_features]) if len(hrv_features) == 14 else np.zeros((1, 14))
            hrv_pred_binary = int(hrv_model.predict(pd.DataFrame(scaled_hrv, columns=[
                'rmssd', 'sdnn', 'mean_rr', 'median_rr', 'p25', 'p75', 'max_diff', 'min_diff',
                'mean_diff', 'std_diff', 'pnn50', 'nn', 'min_rr', 'max_rr']))[0]) if len(hrv_features) == 14 else 0

            # Collect samples with reduced submodel influence
            if label == 2 and ecg_pred != 2:  # High-risk FN
                # Only include if at least one submodel flags abnormality
                if pred_hr or hrv_pred_binary or pred_rhr or pred_hhr:
                    high_fn += 1
                    total_fn += 1
                    feedback_samples.append((segment_normalized, label))
                    sample_counts[label] += 1
                    fn_submodel_stats['hr'].append(pred_hr)
                    fn_submodel_stats['hhr'].append(pred_hhr)
                    fn_submodel_stats['rhr'].append(pred_rhr)
                    fn_submodel_stats['hrv'].append(hrv_pred_binary)
            elif label != ecg_pred or (label == ecg_pred and pred_confidence < 0.5):  # Misclassifications or low-confidence correct
                # Exclude Low/Med if submodels strongly predict High-risk (e.g., HRV and HHR both abnormal)
                if label in [0, 1] and ecg_pred == 2 and high_confidence < 0.9 and not (hrv_pred_binary and pred_hhr):
                    total_fn += 1
                    feedback_samples.append((segment_normalized, label))
                    sample_counts[label] += 1
                elif label == 2 and ecg_pred == 2:  # Correct High-risk
                    feedback_samples.append((segment_normalized, label))
                    sample_counts[label] += 1
            elif label == ecg_pred and pred_confidence >= 0.5:  # High-confidence correct
                if np.random.random() < 0.03:  # 3% chance to include
                    feedback_samples.append((segment_normalized, label))
                    sample_counts[label] += 1

    print(f"\n✅ Total High-risk Samples: {total_high}, High-risk FNs: {high_fn} ({high_fn/total_high*100:.2f}%)")
    print("Submodel FN Trends:")
    for model in fn_submodel_stats:
        abnormal_rate = np.mean(fn_submodel_stats[model]) if fn_submodel_stats[model] else 0
        print(f"  {model.upper()}: Abnormal in {abnormal_rate*100:.2f}% of High-risk FNs")
    print(f"Total False Negatives: {total_fn}")
    print(f"Feedback Samples per Class: Low: {sample_counts[0]}, Med: {sample_counts[1]}, High: {sample_counts[2]}")

    return feedback_samples

# Collect Feedback Samples
feedback_samples = extract_feedback_samples()
if not feedback_samples:
    print("❌ No feedback samples collected.")
    exit(1)

print(f"\n✅ Collected {len(feedback_samples)} feedback samples for fine-tuning.")

# Prepare Training Data
segments = np.array([seg for seg, _ in feedback_samples])
labels = np.array([lbl for _, lbl in feedback_samples])

# Split into train and validation sets
total_samples = len(segments)
train_size = int(0.9 * total_samples)
val_size = total_samples - train_size
train_dataset = TensorDataset(torch.tensor(segments[:train_size], dtype=torch.float32).unsqueeze(-1).to(device),
                              torch.tensor(labels[:train_size], dtype=torch.long).to(device))
val_dataset = TensorDataset(torch.tensor(segments[train_size:], dtype=torch.float32).unsqueeze(-1).to(device),
                            torch.tensor(labels[train_size:], dtype=torch.long).to(device))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Apply SMOTE to training set only
train_segments = segments[:train_size]
train_labels = labels[:train_size]
if len(train_segments) < 11700:
    print("🔄 Applying SMOTE to reach ~11,700 training samples...")
    smote = SMOTE(sampling_strategy={0: 3150, 1: 3600, 2: 4950}, random_state=42, k_neighbors=5)
    train_segments, train_labels = smote.fit_resample(train_segments, train_labels)
train_tensor = torch.tensor(train_segments, dtype=torch.float32).unsqueeze(-1).to(device)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
train_dataset = TensorDataset(train_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Cap at 150,000 to avoid excessive noise
if len(train_segments) > 150000:
    indices = np.random.choice(len(train_segments), 150000, replace=False)
    train_segments = train_segments[indices]
    train_labels = train_labels[indices]

# Fine-Tune BiLSTM
ecg_model.train()
optimizer = optim.Adam(ecg_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
class_weights = torch.tensor([1.0, 1.4, 1.6], device=device)
criterion = FocalLoss(alpha=0.95, gamma=2.0, class_weights=class_weights)

print("\n🔄 Fine-tuning BiLSTM...")
for epoch in range(25):
    # Training Phase
    ecg_model.train()
    total_loss = 0
    all_train_preds, all_train_labels = [], []
    for batch_segments, batch_labels in tqdm(train_loader, desc=f"Fine-tuning Epoch [{epoch+1}/25]"):
        batch_segments, batch_labels = batch_segments.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = ecg_model(batch_segments)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_train_preds.extend(preds)
        all_train_labels.extend(batch_labels.cpu().numpy())

    # Validation Phase
    ecg_model.eval()
    all_val_preds, all_val_labels = [], []
    with torch.no_grad():
        for batch_segments, batch_labels in val_loader:
            batch_segments, batch_labels = batch_segments.to(device), batch_labels.to(device)
            outputs = ecg_model(batch_segments)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_val_preds.extend(preds)
            all_val_labels.extend(batch_labels.cpu().numpy())

    train_accuracy = accuracy_score(all_train_labels, all_train_preds)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    train_cm = confusion_matrix(all_train_labels, all_train_preds)
    val_cm = confusion_matrix(all_val_labels, all_val_preds)
    fn_train_high = train_cm[2, 0] + train_cm[2, 1] if train_cm.shape[0] > 2 else 0
    fn_val_high = val_cm[2, 0] + val_cm[2, 1] if val_cm.shape[0] > 2 else 0
    fn_train_high_rate = fn_train_high / sum(train_cm[2, :]) if train_cm.shape[0] > 2 and sum(train_cm[2, :]) > 0 else 0
    fn_val_high_rate = fn_val_high / sum(val_cm[2, :]) if val_cm.shape[0] > 2 and sum(val_cm[2, :]) > 0 else 0

    print(f"Fine-tuning Epoch [{epoch+1}/25], Loss: {total_loss/len(train_loader):.4f}")
    print(f"   Train Accuracy: {train_accuracy*100:.2f}%, Val Accuracy: {val_accuracy*100:.2f}%")
    print(f"   Train High-risk FN: {fn_train_high_rate*100:.2f}%, Val High-risk FN: {fn_val_high_rate*100:.2f}%")

    # Learning Rate Scheduler Update
    scheduler.step(fn_val_high_rate)

torch.save(ecg_model.state_dict(), "../CardioVision/models/healthkit/bilstm_finetuned.pth")
print("\n✅ Fine-tuned BiLSTM saved to models/healthkit/bilstm_finetuned.pth")