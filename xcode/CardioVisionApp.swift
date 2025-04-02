import HealthKit
import Foundation

class ECGUploader {
    let healthStore = HKHealthStore()
    
    func fetchAndSendLatestECG() {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        let ecgType = HKObjectType.electrocardiogramType()
        let query = HKSampleQuery(sampleType: ecgType, predicate: nil, limit: 1, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]) { _, samples, error in
            
            guard let ecgSample = samples?.first as? HKElectrocardiogram else {
                print("No ECG data found")
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
                    self.sendToBackend(ecgArray: voltageValues)
                case .error(let error):
                    print("ECG Query error: \(error.localizedDescription)")
                }
            }
            self.healthStore.execute(voltageQuery)
        }
        healthStore.execute(query)
    }
    
    func sendToBackend(ecgArray: [Double]) {
        guard let url = URL(string: "http://127.0.0.1:8000/predict_ecg") else { return } // Replace with real URL
        
        let json: [String: Any] = ["ecg": ecgArray]
        let jsonData = try? JSONSerialization.data(withJSONObject: json)
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let data = data, let output = String(data: data, encoding: .utf8) {
                print("Prediction: \(output)")
            } else {
                print("Request failed: \(error?.localizedDescription ?? "No error info")")
            }
        }.resume()
    }
}
