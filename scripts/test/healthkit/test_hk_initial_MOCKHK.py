import os
import json
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the trained model
model = joblib.load("models//healthkit/healthkit_rf_model.pkl")

# Define test directories
base_dir = "../CardioVision/data/mockHealthkit"
labels = {"no_risk": 0, "risk": 1}

X_test = []
y_test = []
misclassified = []

# Load test data
for label_name, label_value in labels.items():
    folder_path = os.path.join(base_dir, label_name)
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                features = [data["hr"], data["hrv"], data["rhr"], data["hhr"]]
                X_test.append(features)
                y_test.append(label_value)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🧪 Test Accuracy: {accuracy:.2f}")

# Detailed Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Risk", "Risk"]))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Log misclassified examples
for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    if true != pred:
        misclassified.append((i, true, pred, X_test[i]))

if misclassified:
    print(f"\n⚠️ Misclassified {len(misclassified)} samples:")
    for idx, true, pred, features in misclassified:
        print(f"Sample {idx}: True={true}, Pred={pred}, Features={features}")
else:
    print("\n✅ No misclassifications found.")
