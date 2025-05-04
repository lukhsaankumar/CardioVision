import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Paths
BASE_DIR = "../CardioVision/data/mockhealthkit"
CATEGORIES = [("no_risk", 0), ("risk", 1)]

data = []
labels = []

# Load JSON files
for folder, label in CATEGORIES:
    folder_path = os.path.join(BASE_DIR, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, "r") as f:
                    sample = json.load(f)
                    data.append([
                        sample["hr"],
                        sample["hrv"],
                        sample["rhr"],
                        sample["hhr"]
                    ])
                    labels.append(label)
            except KeyError as e:
                print(f"Missing key {e} in file: {filepath}")


# Prepare DataFrame
X = pd.DataFrame(data, columns=["hr", "hrv", "rhr", "hhr"])
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/healthkit/healthkit_rf_model.pkl")
print("Model saved to models/healthkit_rf_model.pkl")
