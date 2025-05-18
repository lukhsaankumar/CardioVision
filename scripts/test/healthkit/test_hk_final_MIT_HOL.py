import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import wfdb
from scipy.signal import resample
from tqdm import tqdm

# BiLSTM Model Definition
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

# Load and preprocess ECG segments
def load_ecg_segments(record_path, symbols, label, window_size=250, target_fs=250):
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        signal = record.p_signal[:, 0]  # First lead
        # Impute NaN/Inf
        signal = np.nan_to_num(signal, nan=np.mean(signal[np.isfinite(signal)]), posinf=np.max(signal[np.isfinite(signal)]), neginf=np.min(signal[np.isfinite(signal)]))
        fs = record.fs
        ann_samples = annotation.sample
        ann_symbols = annotation.symbol

        segments, labels = [], []
        scaler = StandardScaler()

        for idx, (sample, symbol) in enumerate(zip(ann_samples, ann_symbols)):
            if symbol not in symbols:
                continue

            start = max(0, sample - window_size // 2)
            end = min(len(signal), sample + window_size // 2)
            segment = signal[start:end]

            if len(segment) < window_size:
                segment = np.pad(segment, (0, window_size - len(segment)), mode='constant')

            # Resample to 250 Hz
            segment_resampled = resample(segment, window_size)

            # Normalize
            segment_normalized = scaler.fit_transform(segment_resampled.reshape(-1, 1)).reshape(-1)
            if np.any(np.isnan(segment_normalized)) or np.any(np.isinf(segment_normalized)) or np.std(segment_normalized) == 0:
                continue

            segments.append(segment_normalized)
            labels.append(label)

        segments = np.array(segments)
        labels = np.array(labels)
        print(f"✅ Loaded {len(segments)} beats from {os.path.basename(record_path)}")
        return segments, labels
    except Exception as e:
        print(f"⚠️ Skipping {record_path}: {e}")
        return np.array([]), np.array([])

# Test model
def test_model():
    # Paths and symbols
    mitdb_path = "../CardioVision/data/mitdb"
    holter_path = "../CardioVision/data/holter"
    low_risk_recs = ["100", "101", "103", "105", "108", "112", "113", "115", "117", "122", "123", "230"]
    med_risk_recs = ["106", "114", "116", "118", "124", "200", "201", "202", "203", "205", "213", "214", "215", "219", "223", "233"]
    high_risk_recs = ["30", "31", "32", "34", "35", "36", "41", "45", "46", "49", "51", "52"]
    low_syms = ['N', 'L', 'R', 'e', 'j']
    med_syms = ['A', 'S', 'a', 'J', '?']
    high_syms = ['V', 'F', 'E']

    all_segments, all_labels = [], []

    # Load Low-risk beats
    print("📥 Loading Low-risk beats...")
    for rec in low_risk_recs:
        print(f"🔍 Reading record {os.path.join(mitdb_path, rec)} ...")
        segments, labels = load_ecg_segments(os.path.join(mitdb_path, rec), low_syms, 0)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)

    # Load Med-risk beats
    print("📥 Loading Med-risk beats...")
    for rec in med_risk_recs:
        print(f"🔍 Reading record {os.path.join(mitdb_path, rec)} ...")
        segments, labels = load_ecg_segments(os.path.join(mitdb_path, rec), med_syms, 1)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)

    # Load High-risk beats
    print("📥 Loading High-risk beats...")
    for rec in high_risk_recs:
        print(f"🔍 Reading record {os.path.join(holter_path, rec)} ...")
        segments, labels = load_ecg_segments(os.path.join(holter_path, rec), high_syms, 2)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)

    if not all_segments:
        print("❌ No valid segments collected.")
        return

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"🎯 Evaluating on {len(X)} beats...")

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [N, 250, 1]
    y_tensor = torch.tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3)
    try:
        model.load_state_dict(torch.load("../CardioVision/models/healthkit/bilstm_finetuned.pth", map_location=device))
    except FileNotFoundError:
        print("❌ Model file '../CardioVision/models/healthkit/bilstm_finetuned.pth' not found.")
        return
    model.to(device)
    model.eval()

    batch_size = 128
    y_true, y_pred = [], []

    print("🔄 Running inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Inference"):
            X_batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_tensor[i:i+batch_size].numpy())

    print("\n🎯 MIT-BIH/Holter Results:")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Med", "High"], digits=2))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"Total Samples: {len(y_true)}")
    print(f"Accuracy: {classification_report(y_true, y_pred, output_dict=True)['accuracy']:.2f}")
    cm = confusion_matrix(y_true, y_pred)
    fn_total = sum(cm[i, j] for i in range(3) for j in range(3) if i != j)
    fn_high = cm[2, 0] + cm[2, 1] if cm.shape[0] > 2 else 0
    fn_total_rate = fn_total / len(y_true) if len(y_true) > 0 else 0
    fn_high_rate = fn_high / sum(cm[2, :]) if cm.shape[0] > 2 and sum(cm[2, :]) > 0 else 0
    print(f"High-risk FN: {fn_high}/{sum(cm[2, :])} = {fn_high_rate*100:.2f}%")
    print(f"Overall FN: {fn_total}/{len(y_true)} = {fn_total_rate*100:.2f}%")

if __name__ == "__main__":
    test_model()
