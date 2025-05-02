import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import wfdb
import random

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Augmentation with fixed length
def augment_signal(signal, target_len=250):
    scale = random.uniform(0.9, 1.1)
    stretched_len = int(len(signal) * scale)
    stretched = np.interp(np.linspace(0, len(signal), stretched_len), np.arange(len(signal)), signal)
    # Resample back to target_len
    return np.interp(np.linspace(0, stretched_len, target_len), np.arange(stretched_len), stretched)

# Load beat segments
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
    beats, labels = [], []

    for sym, beat in zip(ann.symbol, ann.sample):
        if sym not in symbols:
            continue
        start = beat - window_size // 2
        end = beat + window_size // 2
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end]
        if augment:
            segment = augment_signal(segment, window_size)
        beats.append(segment)
        labels.append(label)

    return np.array(beats), np.array(labels)

# Prepare dataset
def prepare_data(window_size=250):
    mit_path = "../CardioVision/data/mitdb"
    holter_path = "../CardioVision/data/holter"

    low_risk_recs = ["100", "101", "103", "105", "108", "112", "113", "115", "117", "122", "123", "230"]
    med_risk_recs = ["106", "114", "116", "118", "124", "200", "201", "202", "203", "205", "213", "214", "215", "219", "223", "233"]
    high_risk_recs = ["30", "31", "32", "34", "35", "36", "41", "45", "46", "49", "51", "52"]

    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_beats, all_labels = [], []

    for rec in low_risk_recs:
        beats, labels = load_beats_multiclass(rec, low_syms, 0, mit_path, window_size=window_size)
        all_beats.append(beats)
        all_labels.append(labels)

    for rec in med_risk_recs:
        beats, labels = load_beats_multiclass(rec, med_syms, 1, mit_path, window_size=window_size)
        if len(beats) > 0:
            # Augment each beat 10 times
            aug_beats = np.array([augment_signal(b, window_size) for b in beats for _ in range(10)])
            aug_labels = np.ones(len(aug_beats), dtype=int)
            all_beats.append(aug_beats)
            all_labels.append(aug_labels)

    for rec in high_risk_recs:
        beats, labels = load_beats_multiclass(rec, high_syms, 2, holter_path, window_size=window_size)
        all_beats.append(beats)
        all_labels.append(labels)

    signals = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor(labels, dtype=torch.long)
    return signals, labels

# Training
def train_model():
    signals, labels = prepare_data()
    print(f"✅ Total Beats: {len(labels)} | Low: {(labels==0).sum().item()} | Med: {(labels==1).sum().item()} | High: {(labels==2).sum().item()}")

    X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=0.2, stratify=labels, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            if torch.isnan(X_batch).any():
                print("⚠️ Skipping NaN batch.")
                continue
            X_batch, y_batch = X_batch.to(model.fc.weight.device), y_batch.to(model.fc.weight.device)
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
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(model.fc.weight.device)
                outputs = model(X_batch)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions.cpu() == y_batch).sum().item()

        accuracy = 100 * correct / len(test_loader.dataset)
        print(f"📊 Epoch [{epoch+1}/30] Loss: {total_loss/len(train_loader):.4f} Accuracy: {accuracy:.2f}%")

    save_path = "../CardioVision/models/ecg/lstm_model_multiclass.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Multiclass model saved to {save_path}")

train_model()
