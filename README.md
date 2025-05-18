# CardioVision 

**Real-time Cardiac Arrest Risk Prediction using ECG and HealthKit Data**

CardioVision leverages advanced machine learning to predict cardiac arrest risk in real-time by analyzing ECG waveforms and physiological data gathered from wearable devices such as the Apple Watch. It aims to improve cardiac event outcomes through early detection and timely interventions.

---

## Project Overview

CardioVision integrates multiple machine learning models to analyze:

- **Electrocardiogram (ECG) Waveforms**
- **Heart Rate (HR)**
- **Heart Rate Variability (HRV)**
- **Resting Heart Rate (RHR)**
- **High Heart Rate Events (HHR)**

Predictions are classified into **Low**, **Medium**, and **High** risk categories, enabling proactive management of cardiac health.

---

## Project Structure 
```yaml
CardioVision/
├── data/ # Raw datasets and mock data
│   ├── zip/
│   │   ├── mitdb/
│   │   ├── holter/
│   │   ├── incart/
│   │   ├── ohca/
│   │   └── mimic3/
│   └── mockhealthkit/
│
├── models/ # Trained model files
│ ├── ecg/
│ ├── healthkit/
│ ├── heartrate/
│ ├── heartratevariability/
│ ├── highheartrateevents/
│ └── restingheartrate/
│
├── pipeline/ # FastAPI backend pipeline
│ ├── main.py
│ └── ecg_model.py
│
├── scripts/ # Scripts for data generation, training, testing
│ ├── demodata/
│ │ ├── generate_healthkit_final.py
│ │ └── generate_healthkit_initial.py
│ │
│ ├── test/ # Testing scripts on various datasets
│ │ ├── ecg/
│ │ ├── healthkit/
│ │ ├── hhr/
│ │ ├── hr/
│ │ ├── hrv/
│ │ └── rhr/
│ │
│ └── train/ # Model training scripts
│ ├── ecg/
│ ├── healthkit/
│ ├── hhr/
│ ├── hr/
│ ├── hrv/
│ └── rhr/
│
├── testresults/ # Model performance results
│ ├── final/
│ ├── holter/
│ ├── incart/
│ ├── mimic3/
│ ├── mitbih/
│ └── ohca/
│
├── venv/ # Python virtual environment
│
├── xcode/ # watchOS/iOS frontend
│ ├── CV Watch App/
│ │ ├── AppSettings.swift
│ │ ├── CardioWatchApp.swift
│ │ ├── ContentView.swift
│ │ └── ECGUploader.swift
│ │
│ └── CV.xcodeproj/ # Xcode project file
│
├── .gitignore
├── README.md
├── requirements.txt
└── references.md
```


## Backend Setup 

### Python Environment Setup

```bash
python -m venv venv
.\venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### If you have already setup the virtual enviroment, to reactivate it do:
```bash
.\venv\Scripts\Activate.ps1
```
### Running FastAPI Server
```bash
uvicorn pipeline.main:app --reload --host 0.0.0.0 --port 8000
```
Verify that the server is running and listening on
```bash
http://0.0.0.0:8000
```

Model Training & Testing 🧑‍💻
Training Models
bash
Copy code
# ECG
python scripts/train/ecg/train_ecg.py

# HealthKit Initial Metrics
python scripts/train/healthkit/train_healthkit.py

# Individual Metrics
python scripts/train/hr/train_hr.py
python scripts/train/hrv/train_hrv.py
python scripts/train/rhr/train_rhr.py
python scripts/train/hhr/train_hhr.py
Testing Models
bash
Copy code
# ECG
python scripts/test/ecg/test_ecg.py

# HealthKit Metrics
python scripts/test/healthkit/test_healthkit.py

# Individual Metrics on respective datasets
python scripts/test/hr/test_hr.py
python scripts/test/hrv/test_hrv.py
python scripts/test/rhr/test_rhr.py
python scripts/test/hhr/test_hhr.py
Results stored in testresults/.

Frontend (watchOS/iOS) Setup ⌚️📱
Requirements
Xcode 14+

watchOS 9.4+ (Physical Apple Watch Series 6+ or Simulator)

Setup
Open the project at xcode/CV.xcodeproj.

Configure demo mode in AppSettings.swift:

swift
Copy code
static var demoMode: Bool = true  // Simulator or demo data
static var demoMode: Bool = false // Live data with HealthKit
Build and run on selected watchOS target (Simulator or physical device).

⚠️ Note:
Due to limited access to an Apple Watch Series 6+ (required for ECG data), HealthKit integration is currently limited to simulated data (demoMode = true).

Key Features 🚀
✅ Real-time ECG waveform classification

✅ HealthKit metrics integration (HR, HRV, RHR, HHR)

✅ Real-time cardiac risk predictions (Low, Medium, High)

✅ Continual learning from user data

Future Enhancements 🌱
Time-series forecasting of cardiac risk (24–72 hours ahead)

Extended dataset integration (e.g., OHCA)

Deployment optimization (Docker, Kubernetes)

Automated CI/CD pipeline for updates

References 📚
MIT-BIH Arrhythmia Database

INCART ECG Database

OHCA Database

Apple HealthKit

FastAPI

Team Members 👥
[Your Name or Team Members Here]

License 📄
[Your License Information Here]

❤️ CardioVision aims to empower users and healthcare providers by delivering accurate, actionable cardiac health insights in real-time.
