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
├── scripts/ 
│ ├── demodata/ # Data generation scripts for demo
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
├── xcode/ 
│ ├── CV Watch App/ # watchOS/iOS frontend
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
If you have already setup the virtual enviroment, to reactivate it do:
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

## Extracting Datasets
Before training or testing models, ensure each dataset ZIP file under 
```bash data/zip/``` is extracted into its respective folder inside the ```bash data/``` directory.

You can use copy these commands to do it at once:
```bash
unzip data/zip/holter.zip -d data/holter/
unzip data/zip/mimic3.zip -d data/mimic3/
unzip data/zip/ohca.zip -d data/ohca/
unzip data/zip/mitbih.zip -d data/mitbih/
unzip data/zip/incart.zip -d data/incart/
```

## Model Training & Testing
# Training Models
Each metric (e.g., ECG, HR, HRV, RHR, HHR, HealthKit) has its own training script located at ```bash scripts/train/{metric}/train_{metric}.py.```

For example:
```bash
python scripts/train/ecg/train_ecg.py
```
Some metrics may have multiple iterations (e.g., ```bash train_hrv2.py``` for a second iteration of HRV training).

Each training script contains a description at the top that clearly outline the model, feature extraction, and where the models and scalars are saved: ```bash models/{metric}/```

# Testing Models
Testing scripts evaluate trained models on various datasets.
Scripts are located at ```bash scripts/test/{metric}/test_{metric}_{DATASET}.py.```

For example:
```bash
python scripts/test/ecg/test_ecg_OHCA.py
```
Similar to training, test scripts might have numbered iterations (e.g., ```bash test_hrv2_MITBIH.py```).
Each test script includes a description at the top outlining the model being tested, dataset and evaluation, location where results will be saved: ```bash testresults/{dataset}/```

# Viewing Test Results
All test results from running the scripts are saved in testresults/ with clear naming conventions:
```bash
testresults/testresults/{dataset}/{DATASET}_{METRIC}.txt
```

For example:
```bash testresults/ohca/OHCA_ECG3.txt``` contains the results of ```bash python scripts/test/ecg/test_ecg3_OHCA.py```

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
