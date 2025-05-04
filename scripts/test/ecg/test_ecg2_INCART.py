import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# LSTM Model Definition (same as used in training)
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

def extract_beats_from_incart(record_path, symbols, label, window_size=250):
    beats, labels = [], []
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]  # Use lead I by default
    for sym, loc in zip(ann.symbol, ann.sample):
        if sym not in symbols:
            continue
        start = loc - window_size // 2
        end = loc + window_size // 2
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end]
        if np.isnan(segment).any() or np.std(segment) == 0:
            continue
        beats.append(segment)
        labels.append(label)
    return beats, labels

def test_incart_ecg_model():
    base_path = "../CardioVision/data/incart/files"
    incart_recs = [f"I{str(i).zfill(2)}" for i in range(1, 76)]
    window_size = 250

    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_beats, all_labels = [], []

    print("📥 Loading and preprocessing INCART data...")
    for rec in tqdm(incart_recs):
        path = os.path.join(base_path, rec)
        try:
            low, l_lbl = extract_beats_from_incart(path, low_syms, 0, window_size)
            med, m_lbl = extract_beats_from_incart(path, med_syms, 1, window_size)
            high, h_lbl = extract_beats_from_incart(path, high_syms, 2, window_size)
            all_beats.extend(low + med + high)
            all_labels.extend(l_lbl + m_lbl + h_lbl)
        except Exception as e:
            print(f"⚠️ Skipping {rec} due to error: {e}")

    print(f"✅ Total samples: {len(all_beats)}")

    X = np.array(all_beats).astype(np.float32)
    y = np.array(all_labels)
    X_tensor = torch.tensor(X).unsqueeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3)
    model.load_state_dict(torch.load("../CardioVision/models/ecg/lstm_model_multiclass.pth", map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y[i:i+batch_size])

    print("📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Med", "High"]))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_incart_ecg_model()
