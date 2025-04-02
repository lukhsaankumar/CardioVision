import HealthKit
import Foundation
import Combine

class ECGUploader: ObservableObject {
    private let healthStore = HKHealthStore()
    @Published var predictionResult: String = "Press button to predict"
    @Published var ecgSignal: [Double] = []

    func fetchAndSendLatestECG() {
        guard HKHealthStore.isHealthDataAvailable() else {
            predictionResult = "Health data not available"
            return
        }

        let ecgType = HKObjectType.electrocardiogramType()
        let query = HKSampleQuery(sampleType: ecgType, predicate: nil, limit: 1,
                                   sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]) { _, samples, error in

            guard let ecgSample = samples?.first as? HKElectrocardiogram else {
                DispatchQueue.main.async {
                    self.predictionResult = "No ECG data found"
                }
                return
            }

            var voltageValues: [Double] = []

            let voltageQuery = HKElectrocardiogramQuery(ecgSample) { _, result in
                switch result {
                case .measurement(let measurement):
                    if let voltage = measurement.quantity(for: .appleWatchSimilarToLeadI)?.doubleValue(for: .volt()) {
                        voltageValues.append(voltage)
                    }
                case .done:
                    DispatchQueue.main.async {
                        self.ecgSignal = voltageValues
                    }
                    self.sendToBackend(ecgArray: voltageValues)
                case .error(let error):
                    DispatchQueue.main.async {
                        self.predictionResult = "ECG Query error: \(error.localizedDescription)"
                    }
                }
            }
            self.healthStore.execute(voltageQuery)
        }
        healthStore.execute(query)
    }

    func sendToBackend(ecgArray: [Double]) {
        guard let url = URL(string: "http://127.0.0.1:8000/predict_ecg") else {
            DispatchQueue.main.async {
                self.predictionResult = "Invalid backend URL"
            }
            return
        }

        let json: [String: Any] = ["ecg": ecgArray]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: json) else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let data = data,
               let output = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let risk = output["risk"] as? String,
               let probability = output["probability"] as? Double {

                DispatchQueue.main.async {
                    self.predictionResult = "Risk: \(risk.capitalized) (\(String(format: "%.2f", probability * 100))%)"
                }
            } else {
                DispatchQueue.main.async {
                    self.predictionResult = "Prediction failed: \(error?.localizedDescription ?? "unknown error")"
                }
            }
        }.resume()
    }
}
