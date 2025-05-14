import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

# Load and preprocess OHCA JSON data
def load_ohca_segments(json_path, window_size=250, stride=125, target_fs=250, label=2):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        voltages = np.array(data['voltages'], dtype=np.float32)
        sampling_frequency = data.get('samplingFrequency', 512)  # Default to 512 Hz

        if np.any(np.isnan(voltages)) or np.any(np.isinf(voltages)):
            print(f"⚠️ Skipping {json_path} due to NaN/Inf in voltages")
            return np.array([]), np.array([]), []

        if len(voltages) < window_size:
            print(f"⚠️ Skipping {json_path} due to insufficient length: {len(voltages)} < {window_size}")
            return np.array([]), np.array([]), []

        segments, labels, metadata = [], [], []
        scaler = StandardScaler()

        for idx, start in enumerate(range(0, len(voltages) - window_size + 1, stride)):
            segment = voltages[start:start + window_size]
            if len(segment) != window_size:
                continue

            # Resample to 250 Hz, ensuring window_size=250
            segment_resampled = resample(segment, window_size)

            # Normalize segment
            segment_normalized = scaler.fit_transform(segment_resampled.reshape(-1, 1)).reshape(-1)
            if np.any(np.isnan(segment_normalized)) or np.any(np.isinf(segment_normalized)) or np.std(segment_normalized) == 0:
                continue

            segments.append(segment_normalized)
            labels.append(label)
            metadata.append((os.path.basename(json_path), idx))

        segments = np.array(segments)
        labels = np.array(labels)
        print(f"✅ Processed {json_path}: {len(segments)} segments")
        return segments, labels, metadata
    except Exception as e:
        print(f"⚠️ Skipping {json_path} due to error: {e}")
        return np.array([]), np.array([]), []

# Evaluate model on OHCA dataset
def test_ohca_ecg_model():
    base_path = "../CardioVision/data/mockhealthkit/high_risk"
    json_files = [f for f in os.listdir(base_path) if f.endswith(".json")]
    window_size = 250
    stride = 125
    target_fs = 250
    high_risk_label = 2

    all_segments, all_labels, all_metadata = [], [], []

    print("📥 Loading and preprocessing OHCA data...")
    for json_file in tqdm(json_files, desc="Processing JSONs"):
        json_path = os.path.join(base_path, json_file)
        segments, labels, metadata = load_ohca_segments(json_path, window_size, stride, target_fs, high_risk_label)
        if segments.size > 0:
            all_segments.extend(segments)
            all_labels.extend(labels)
            all_metadata.extend(metadata)

    if not all_segments:
        print("❌ No valid segments collected for testing.")
        return

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"✅ Total samples: {len(X)} (High: {(y == high_risk_label).sum()})")

    if len(X) == 0:
        print("❌ No valid samples after processing.")
        return

    X_tensor = torch.tensor(X).unsqueeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMModel(input_size=1, hidden_size=128, num_layers=3, output_size=3, dropout=0.3)
    try:
        model.load_state_dict(torch.load("../CardioVision/models/ecg/bilstm_model_multiclass.pth", map_location=device))
    except FileNotFoundError:
        print("❌ Model file '../CardioVision/models/ecg/bilstm_model_multiclass.pth' not found.")
        return
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    batch_size = 128

    print("🔄 Running inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Inference"):
            X_batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y[i:i+batch_size])

    print("\n🎯 OHCA Results:")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Low", "Med", "High"], digits=2))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Log classified segments
    class_names = ["Low", "Med", "High"]
    classified_segments = {name: [] for name in class_names}
    for idx, pred in enumerate(y_pred):
        classified_segments[class_names[pred]].append(all_metadata[idx])

    print("\n📋 Classified Segments:")
    for class_name in class_names:
        segments = classified_segments[class_name]
        print(f"{class_name} Classified Segments ({len(segments)}):")
        if segments:
            for segment in segments[:10]:
                print(f"  - {segment[0]}, Segment {segment[1]}")
            if len(segments) > 10:
                print(f"  ... and {len(segments) - 10} more")
        else:
            print("  - None")

if __name__ == "__main__":
    test_ohca_ecg_model()