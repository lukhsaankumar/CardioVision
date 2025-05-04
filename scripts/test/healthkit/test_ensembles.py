import os
import numpy as np
import torch
import wfdb
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report

# --- Globals ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ensemble models
ensemble4_model = load('./models/ensemble/ensemble4_model.pkl')
ensemble5_model = load('./models/ensemble/ensemble5_model.pkl')

# Define ECG LSTM model
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Load ECG model
ecg_model = LSTMModel().to(device)
ecg_model.load_state_dict(torch.load('./models/ecg/lstm_model.pth', map_location=device))
ecg_model.eval()

# --- Main Testing Logic ---
def run_test():
    mitdb_records = [
        *[str(i) for i in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))],
        *[str(i) for i in list(range(200, 204)) + [205] + list(range(207, 211)) +
          list(range(212, 216)) + [217] + list(range(219, 224)) + [228] + list(range(230, 235))]
    ]

    results = []

    for record in mitdb_records:
        try:
            signal, _ = wfdb.rdsamp(f'../CardioVision/data/mitdb/{record}')
            ecg_signal = signal[:, 0]

            # Predict with ECG model
            ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                out = ecg_model(ecg_tensor)
                ecg_pred = (torch.sigmoid(out.squeeze()) > 0.5).float().item()

            # Dummy feature values for HR, HRV, RHR, HHR
            features4 = np.array([1, 1, 1, 1]).reshape(1, -1)
            features5 = np.append(features4.flatten(), ecg_pred).reshape(1, -1)

            risk4 = ensemble4_model.predict(features4)[0]
            risk5 = ensemble5_model.predict(features5)[0]

            results.append({
                "Record": record,
                "4-Ensemble Risk": int(risk4),
                "5-Ensemble Risk": int(risk5),
                "ECG Score": round(ecg_pred, 4)
            })

        except Exception as e:
            print(f"[⚠️] Skipped {record}: {e}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_test()
