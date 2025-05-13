import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils")
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wfdb
from tqdm import tqdm
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# === Load All Submodels ===
with open("models/heartrate/hr_model2.json") as f:
    hr_model_data = json.load(f)
hrv_model = joblib.load("models/heartratevariability/hrv_ensemble_model.pkl")
scaler_hrv = joblib.load("models/heartratevariability/scaler.pkl")
rhr_model = joblib.load("models/restingheartrate/rhr_model.pkl")
scaler_rhr = joblib.load("models/restingheartrate/scaler.pkl")
hhr_model = joblib.load("models/highheartrateevents/rf_hhr2_model.pkl")

# ECG model (LSTM, to be replaced with CNN-LSTM if better)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

ecg_model = LSTMModel()
ecg_model.load_state_dict(torch.load("models/ecg/lstm_model_multiclass.pth", map_location=torch.device('cpu')))
ecg_model.eval()

# === Meta-Learner Model ===
class EnsembleMetaLearner(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, output_size=3):  # 3 (LSTM) + 4 * 3 (binary models)
        super(EnsembleMetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# === Paths and Label Maps ===
paths = {
    "mitdb": "data/mitdb",
    "holter": "data/holter",
    "incart": "data/incart/files"
}
low_syms = ['N', 'L', 'R', 'e', 'j']
med_syms = ['A', 'S', 'a', 'J', '?']
high_syms = ['V', 'F', 'E']
symbol_to_label = {s: 0 for s in low_syms}
symbol_to_label.update({s: 1 for s in med_syms})
symbol_to_label.update({s: 2 for s in high_syms})

# === Helper Functions ===
def get_r_peaks(signal, fs):
    peaks, _ = find_peaks(signal, distance=fs * 0.6)
    return peaks

def extract_hr(signal, fs):
    r_peaks = get_r_peaks(signal, fs)
    rr = np.diff(r_peaks) / fs
    hr_series = 60 / (rr + 1e-6)
    return hr_series

def compute_hrv_features(rr_intervals):
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
    if len(rr_intervals) < 2:
        return np.zeros(14)  # Ensure 14 features
    diff_rr = np.diff(rr_intervals)
    features = [
        np.sqrt(np.mean(diff_rr ** 2)),  # RMSSD
        np.std(rr_intervals),            # SDNN
        np.mean(rr_intervals),
        np.median(rr_intervals),
        np.percentile(rr_intervals, 25),
        np.percentile(rr_intervals, 75),
        np.max(diff_rr),
        np.min(diff_rr),
        np.mean(diff_rr),
        np.std(diff_rr),
        np.sum(np.abs(diff_rr) > 50) / len(diff_rr),  # pNN50
        len(rr_intervals),
        np.min(rr_intervals),
        np.max(rr_intervals)
    ]
    return np.array(features)

def extract_ensemble_features(path, label_map, dataset_type, lstm_weight=0.8):
    X, y = [], []
    try:
        rec = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')
        signal = rec.p_signal[:, 0]
        fs = rec.fs
        r_peaks = ann.sample
        syms = ann.symbol

        hr_series = extract_hr(signal, fs)
        rr_intervals = np.diff(r_peaks) / fs * 1000
        hrv_feat = compute_hrv_features(rr_intervals)

        for idx, sym in enumerate(syms):
            if sym not in label_map:
                continue

            # Adaptive Segment Extraction
            beat = r_peaks[idx] if idx < len(r_peaks) else None
            if beat is None:
                continue

            start = max(0, beat - 125)
            end = min(len(signal), beat + 125)
            segment = signal[start:end]

            if len(segment) < 50:
                continue

            # ECG Prediction (LSTM - 3-class)
            if len(segment) < 250:
                segment = np.pad(segment, (0, 250 - len(segment)), mode='constant')
            segment = StandardScaler().fit_transform(segment.reshape(-1, 1)).reshape(-1)  # Normalize
            ecg_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                ecg_output = torch.softmax(ecg_model(ecg_tensor), dim=-1).numpy().flatten()  # [p_low, p_med, p_high]
            ecg_pred = np.argmax(ecg_output)
            ecg_conf = ecg_output[ecg_pred]

            # HR Model (Binary - Arrhythmia Detection)
            hr_val = hr_series[min(idx, len(hr_series) - 1)] if len(hr_series) > idx else 0
            pred_hr = 1 if hr_val > 120 else 0
            hr_probs = [0, 0.5, 0.5] if pred_hr == 1 else [1, 0, 0]  # Map to 3-class

            # HHR Model (Sustained High HR)
            sustained_high_hr = int(np.all(hr_series[max(0, idx - 10):idx] > 150))
            pred_hhr = 1 if sustained_high_hr else 0
            hhr_probs = [0, 0.5, 0.5] if pred_hhr == 1 else [1, 0, 0]

            # RHR Model (Binary - Arrhythmia Detection)
            if len(rr_intervals) > 0:
                rhr = 60000 / np.mean(rr_intervals)
                pred_rhr = 1 if rhr > 75 else 0
                rhr_probs = [0, 0.5, 0.5] if pred_rhr == 1 else [1, 0, 0]
            else:
                rhr_probs = [1, 0, 0]

            # HRV Model (Binary - Arrhythmia Detection)
            scaled_hrv = scaler_hrv.transform([hrv_feat]) if len(hrv_feat) == 14 else np.zeros((1, 14))
            hrv_pred_binary = int(hrv_model.predict(scaled_hrv)[0])
            hrv_probs = [0, 0.5, 0.5] if hrv_pred_binary == 1 else [1, 0, 0]

            # Combine probabilities with weights
            binary_weight = (1 - lstm_weight) / 4  # Equal weight for each binary model
            combined_probs = (
                lstm_weight * ecg_output +
                binary_weight * np.array(hr_probs) +
                binary_weight * np.array(hhr_probs) +
                binary_weight * np.array(rhr_probs) +
                binary_weight * np.array(hrv_probs)
            )
            features = np.concatenate([ecg_output, hr_probs, hhr_probs, rhr_probs, hrv_probs])

            X.append(features)
            y.append(label_map[sym])
    except Exception as e:
        print(f"⚠️ Skipped {path}: {e}")

    return X, y

# === Process Selected Records with Progress Bar ===
X_all, y_all = [], []
record_paths = []

for rec_dir in paths:
    for rec in os.listdir(paths[rec_dir]):
        if rec.endswith(".hea"):
            record_paths.append((os.path.join(paths[rec_dir], rec.replace(".hea", "")), rec_dir))

print("🔄 Extracting Features from Records...")
for record_path, dataset_type in tqdm(record_paths, desc="Processing Records"):
    X, y = extract_ensemble_features(record_path, symbol_to_label, dataset_type)
    X_all.extend(X)
    y_all.extend(y)

# === Train Meta-Learner ===
if len(X_all) == 0:
    print("⚠️ No training data collected. Ensure the records are not being skipped.")
else:
    X = np.array(X_all)
    y = np.array(y_all)
    print(f"\n✅ Collected {len(X)} samples for training meta-learner.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Compute class weights
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize meta-learner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta_learner = EnsembleMetaLearner(input_size=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(meta_learner.parameters(), lr=1e-3)

    # Train meta-learner
    print("\n🔄 Training Meta-Learner...")
    for epoch in range(10):
        meta_learner.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch [{epoch+1}/10]"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = meta_learner(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        meta_learner.eval()
        correct = 0
        class_correct = [0] * 3
        class_total = [0] * 3
        with torch.no_grad():
            outputs = meta_learner(X_test_tensor.to(device))
            predictions = torch.argmax(outputs, dim=1).cpu()
            correct += (predictions == y_test_tensor).sum().item()
            for label, pred in zip(y_test_tensor, predictions):
                class_correct[label] += (pred == label).item()
                class_total[label] += 1

        accuracy = 100 * correct / len(y_test_tensor)
        class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(3)]
        print(f"📊 Epoch [{epoch+1}/10] Loss: {total_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")
        print(f"   Class Accuracies - Low: {class_accuracies[0]:.2f}%, Med: {class_accuracies[1]:.2f}%, High: {class_accuracies[2]:.2f}%")

    # Final Evaluation
    meta_learner.eval()
    with torch.no_grad():
        y_pred = torch.argmax(meta_learner(X_test_tensor.to(device)), dim=1).cpu().numpy()
    print("\n🎯 Final Ensemble Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save meta-learner
    torch.save(meta_learner.state_dict(), "models/ensemble_meta_learner.pth")
    print("\n✅ Meta-learner saved to models/ensemble_meta_learner.pth")