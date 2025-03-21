
# CardioVision

CardioVision is a machine learning-powered system designed to predict cardiac arrest risk in real-time by analyzing ECG waveforms and health data from wearable devices like the Apple Watch. The project aims to improve early detection and preventive care for cardiac events by combining time-series ECG data with physiological signals such as heart rate, heart rate variability (HRV), and oxygen saturation.

---

## Project Overview
Cardiac arrest is a leading cause of death worldwide, with survival rates as low as 10% outside of a hospital setting. Early detection of irregular heart activity can drastically improve survival rates through timely medical intervention.

CardioVision leverages deep learning models, including an LSTM-based classifier, to analyze ECG waveforms and predict the likelihood of cardiac arrest or arrhythmic episodes. The model is trained on the MIT-BIH Arrhythmia Database and will be extended using additional datasets (e.g., OHCA and real-time HealthKit data) to improve prediction accuracy and generalization.

The system is designed to:
✅ Classify heartbeats as normal or arrhythmic using ECG waveforms  
✅ Adapt to new data from HealthKit and other real-world sources using continual learning  
✅ Provide real-time risk prediction for cardiac events over a 24–72 hour window  
✅ Output a risk score (Low, Medium, High) based on predicted heart activity patterns  

---

## Setup
### 1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
.env\Scriptsctivate
pip install -r requirements.txt
```

---

### 2. Train the Model
Train the LSTM model on the MIT-BIH database:
```bash
python models/train_lstm.py
```

---

### 3. Test the Model on MIT-BIH Records
Evaluate the model's performance on individual records from the MIT-BIH dataset:
```bash
python models/inference.py
```

---

## HealthKit Metrics
Apple's HealthKit offers several key metrics that can enhance cardiac risk prediction:  
- **Heart Rate (HR):** Measures the user’s heartbeats per minute  
- **Heart Rate Variability (HRV):** Measures variation in time intervals between heartbeats (indicator of autonomic function)  
- **Resting Heart Rate:** Captures the lowest heart rate during periods of rest  
- **High Heart Rate Events:** Flags when the user’s heart rate exceeds a set threshold  
- **Electrocardiogram (ECG):** Provides detailed electrical activity of the heart to detect irregular rhythms  

These metrics will be integrated into the model to improve accuracy and enable real-time cardiac risk monitoring directly from Apple Watch.

---

## Future Scope
✅ Add more diverse datasets (e.g., OHCA) to improve model generalization  
✅ Integrate Apple HealthKit data for real-time cardiac monitoring  
✅ Implement a time-series forecasting model to predict cardiac risk over the next 24–72 hours  
✅ Deploy using FastAPI and Kubernetes for scalable inference and monitoring  

---

CardioVision aims to deliver an advanced, real-time cardiac risk prediction system that empowers users and healthcare professionals with actionable insights. ❤️
