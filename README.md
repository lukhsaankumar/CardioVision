# CardioVision 

**Real-time Cardiac Arrest Risk Prediction using ECG and HealthKit Data**

CardioVision is team LNV's RBC Borealis Let's Solve it Spring 2025 project. It leverages advanced machine learning to predict cardiac arrest risk in real-time by analyzing ECG waveforms and physiological data gathered from wearable devices such as the Apple Watch. It aims to improve cardiac event outcomes through early detection and timely interventions.

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
## DEMO

![](presentation/DEMO.gif)

 - Starts with loop of no risk data from data/mockhealthkitdata/no_risk/
 - Sends a random json from data/mockhealthkitdata/risk/ to FastAPI which sends to initial model
 - Initial model detects risk and prompts user for ECG
 - Once "ECG Recorded" is pressed, demo sends ECG from OHCA database of patient experiencing cardiac arrest
 - Final model predicts High Risk

## Project Structure 
```yaml
CardioVision/
├── data/ # Raw datasets and mock data
│   ├── zip/ # Contains compressed versions of the datasets
│   ├── mitdb/
│   ├── holter/
│   ├── incart/
│   ├── ohca/
│   └── mimic3wdb/
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

## Python Environment Setup

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

OHCA
Download the dataset from https://data.mendeley.com/datasets/wpr5nzyn2z/1 and extract the zip into the ```bash data/ohca/``` folder

INCART:
Download by doing ```bash wget -r -N -c -np -P data/incart/ https://physionet.org/files/incartdb/1.0.0/```

HOLTER:
Download by doing ```bash wget -r -N -c -np -P data/holter/ https://physionet.org/files/sddb/1.0.0/```

MITBIH:
Download by doing ```bash wget -r -N -c -np -P data/mitdb/ https://physionet.org/files/mitdb/1.0.0/```

MIMIC3:
Download by doing ```bash wget -r -N -c -np -P data/mimic3wdb/ https://physionet.org/files/mimic3wdb/1.0/```

## Model Training & Testing
### Training Models
Each metric (e.g., ECG, HR, HRV, RHR, HHR, HealthKit) has its own training script located at ```bash scripts/train/{metric}/train_{metric}.py.```

For example:
```bash
python scripts/train/ecg/train_ecg.py
```
Some metrics may have multiple iterations (e.g., ```bash train_hrv2.py``` for a second iteration of HRV training).

Each training script contains a description at the top that clearly outline the model, feature extraction, and where the models and scalars are saved: ```bash models/{metric}/```

### Testing Models
Testing scripts evaluate trained models on various datasets.
Scripts are located at ```bash scripts/test/{metric}/test_{metric}_{DATASET}.py.```

For example:
```bash
python scripts/test/ecg/test_ecg_OHCA.py
```
Similar to training, test scripts might have numbered iterations (e.g., ```bash test_hrv2_MITBIH.py```).
Each test script includes a description at the top outlining the model being tested, dataset and evaluation, location where results will be saved: ```bash testresults/{dataset}/```

### Viewing Test Results
All test results from running the scripts are saved in testresults/ with clear naming conventions:
```bash
testresults/testresults/{dataset}/{DATASET}_{METRIC}.txt
```

For example:
```bash testresults/ohca/OHCA_ECG3.txt``` contains the results of ```bash python scripts/test/ecg/test_ecg3_OHCA.py```

## Frontend (watchOS/iOS) Setup
### Requirements:
 - Xcode 14+
 - watchOS 9.4+ (Physical Apple Watch Series 6+ or Simulator)

### Setup
Open the project at xcode/CV.xcodeproj.
Configure mode in AppSettings.swift:

```swift
static var demoMode: Bool = true  // Simulator or demo data
static var demoMode: Bool = false // Live data with HealthKit
```

### Run
 - Select watchOS target in Xcode
 - Choose a simulator such as Watch Series 7 on watchOS 9.4 or connect a physical Apple Watch
 - Build and run the project
 - Observe the disclaimer screen authorize, Healthkit access and confirm live heart rate and ECG workflows appear

## Disclaimers
 - Apple Watch Series 6 or newer is required to test the ECG functionality of the app, as earlier models do not support ECG data collection via HealthKit.
 - Due to the unavailability of an Apple Watch Series 6 or above, and an existing bug preventing the appearance of "Developer Positions" to enable app testing on physical watches, the live HealthKit metrics integration has not been fully tested on actual hardware. Current testing relies on mock data.
 - The currently fine-tuned Bidirectional LSTM (```bash models/healthkit/bilstm_finetuned.pth```) exhibits some limitations due to suboptimal performance of associated submodels (HR, HRV, RHR, HHR). Therefore, all tests (including demonstrations) currently use - and should continue to use - the original model located at: ```bash models/ecg/bilstm_model_multiclass.pth```




