import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import wfdb
import random
from tqdm import tqdm

# Define Bidirectional LSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Double hidden size for forward + backward

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Enhanced Augmentation
def augment_signal(signal, target_len=250, noise_std=0.01, max_shift=20, scale_range=(0.8, 1.2)):
    # Noise
    if random.random() < 0.5:
        noise = np.random.normal(0, noise_std, signal.shape)
        signal = signal + noise
    # Time Shift
    if random.random() < 0.5:
        shift = random.randint(-max_shift, max_shift)
        signal = np.roll(signal, shift)
    # Scaling
    if random.random() < 0.5:
        scale = random.uniform(*scale_range)
        signal = signal * scale
    # Stretch
    scale = random.uniform(0.9, 1.1)
    stretched_len = int(len(signal) * scale)
    stretched = np.interp(np.linspace(0, len(signal), stretched_len), np.arange(len(signal)), signal)
    signal = np.interp(np.linspace(0, stretched_len, target_len), np.arange(stretched_len), stretched)
    return signal

# Load MIT-BIH/Holter/INCART beats
def load_beats_multiclass(record, symbols, label, base_path, augment=False, window_size=250):
    record_path = os.path.join(base_path, record)
    print(f"🔍 Reading record {record_path} ...")
    try:
        rec = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, extension='atr')
    except Exception as e:
        print(f"⚠️ Skipping {record} due to error: {e}")
        return np.empty((0, window_size)), np.empty((0,), dtype=int)

    signal = rec.p_signal[:, 0]
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        print(f"⚠️ Imputing NaN/Inf in signal for record {record}")
        valid_mask = np.isfinite(signal)
        if np.sum(valid_mask) > 0:
            signal[~valid_mask] = np.mean(signal[valid_mask])
        else:
            print(f"⚠️ Skipping {record}: No valid signal values")
            return np.empty((0, window_size)), np.empty((0,), dtype=int)

    scaler = StandardScaler()
    beats, labels = [], []

    for sym, beat in zip(ann.symbol, ann.sample):
        if sym not in symbols:
            continue
        start = beat - window_size // 2
        end = beat + window_size // 2
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end]
        if len(segment) != window_size:
            segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')
        if augment:
            segment = augment_signal(segment, window_size)
        segment = scaler.fit_transform(segment.reshape(-1, 1)).reshape(-1)
        if np.any(np.isnan(segment)) or np.any(np.isinf(segment)) or np.std(segment) == 0:
            continue
        beats.append(segment)
        labels.append(label)

    print(f"✅ Loaded {len(beats)} beats from {record}")
    return np.array(beats), np.array(labels)

def prepare_data(window_size=250):
    mit_path = "../CardioVision/data/mitdb"
    holter_path = "../CardioVision/data/holter"
    incart_path = "../CardioVision/data/incart/files"

    incart_low_recs = [f"I{i:02d}" for i in [8, 9, 17, 25, 28, 38]]
    low_risk_recs = ["100", "101", "103", "105", "108", "112", "113", "115", "117", "122", "123", "230"]
    med_risk_recs = ["106", "114", "116", "118", "124", "200", "201", "202", "203", "205", "213", "214", "215", "219", "223", "233"]
    high_risk_recs = ["30", "31", "32", "34", "35", "36", "41", "45", "46", "49", "51", "52"]
    incart_high_recs = [f"I{i:02d}" for i in [3, 5, 7, 10, 11, 16, 21, 24, 27, 31]]
    incart_med_recs = [f"I{i:02d}" for i in [12, 13, 14, 20, 21, 22, 40, 41, 47, 48, 72, 73]]

    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_beats, all_labels = [], []

    print("📥 Loading MIT-BIH Low Risk...")
    for rec in tqdm(low_risk_recs, desc="MIT-BIH Low"):
        beats, labels = load_beats_multiclass(rec, low_syms, 0, mit_path, augment=True, window_size=window_size)
        all_beats.append(beats)
        all_labels.append(labels)

    print("📥 Loading INCART Low Risk...")
    for rec in tqdm(incart_low_recs, desc="INCART Low"):
        beats, labels = load_beats_multiclass(rec, low_syms, 0, incart_path, augment=True, window_size=window_size)
        all_beats.append(beats)
        all_labels.append(labels)

    print("📥 Loading MIT-BIH Medium Risk...")
    for rec in tqdm(med_risk_recs, desc="MIT-BIH Med"):
        beats, labels = load_beats_multiclass(rec, med_syms, 1, mit_path, augment=False, window_size=window_size)
        if len(beats) > 0:
            aug_beats = np.array([augment_signal(b, window_size) for b in beats for _ in range(10)])
            aug_labels = np.ones(len(aug_beats), dtype=int)
            all_beats.append(aug_beats)
            all_labels.append(aug_labels)

    print("📥 Loading INCART Medium Risk...")
    for rec in tqdm(incart_med_recs, desc="INCART Med"):
        beats, labels = load_beats_multiclass(rec, med_syms, 1, incart_path, augment=False, window_size=window_size)
        if len(beats) > 0:
            aug_beats = np.array([augment_signal(b, window_size) for b in beats for _ in range(10)])
            aug_labels = np.ones(len(aug_beats), dtype=int)
            all_beats.append(aug_beats)
            all_labels.append(aug_labels)

    print("📥 Loading MIT-BIH/Holter High Risk...")
    for rec in tqdm(high_risk_recs, desc="Holter High"):
        beats, labels = load_beats_multiclass(rec, high_syms, 2, holter_path, augment=True, window_size=window_size)
        all_beats.append(beats)
        all_labels.append(labels)

    print("📥 Loading INCART High Risk...")
    for rec in tqdm(incart_high_recs, desc="INCART High"):
        beats, labels = load_beats_multiclass(rec, high_syms, 2, incart_path, augment=True, window_size=window_size)
        all_beats.append(beats)
        all_labels.append(labels)

    # Validate beat shapes
    valid_beats, valid_labels = [], []
    for beats, labels in zip(all_beats, all_labels):
        if beats.size > 0:
            if beats.shape[1] != window_size:
                print(f"⚠️ Skipping beats with shape {beats.shape}, expected ({beats.shape[0]}, {window_size})")
                continue
            valid_beats.append(beats)
            valid_labels.append(labels)

    if not valid_beats:
        raise ValueError("❌ No valid data loaded. Check data paths and files.")
    
    signals = np.concatenate(valid_beats, axis=0)
    labels = np.concatenate(valid_labels, axis=0)
    if len(signals) == 0 or len(labels) == 0:
        raise ValueError("❌ No valid data after concatenation. Check data integrity.")
    
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor(labels, dtype=torch.long)
    return signals, labels

def train_model():
    signals, labels = prepare_data(window_size=250)
    print(f"✅ Total Beats: {len(labels)} | Low: {(labels==0).sum().item()} | Med: {(labels==1).sum().item()} | High: {(labels==2).sum().item()}")

    X_train, X_test, y_train, y_test = train_test_split(
        signals, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Compute class weights
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    print(f"🔍 Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            if torch.isnan(X_batch).any():
                print("⚠️ Skipping NaN batch.")
                continue
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            if torch.isnan(loss):
                print("⚠️ NaN loss encountered. Skipping batch.")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 3
        class_total = [0] * 3
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                for i in range(3):
                    class_correct[i] += ((predictions == i) & (y_batch == i)).sum().item()
                    class_total[i] += (y_batch == i).sum().item()

        accuracy = 100 * correct / total
        class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(3)]
        print(f"📊 Epoch [{epoch+1}/30] Loss: {total_loss/len(train_loader):.4f} Accuracy: {accuracy:.2f}%")
        print(f"   Per-class Accuracies: Low: {class_accuracies[0]:.2f}%, Med: {class_accuracies[1]:.2f}%, High: {class_accuracies[2]:.2f}%")

    save_path = "../CardioVision/models/ecg/bilstm_model_multiclass.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ BiLSTM model saved to {save_path}")

if __name__ == "__main__":
    train_model()