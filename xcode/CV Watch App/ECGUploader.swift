import Foundation
import HealthKit
import Combine

class ECGUploader: ObservableObject {
    private let healthStore = HKHealthStore()
    private var timer: Timer?

    @Published var latestHeartRate: Double = 0.00
    @Published var latestHRV: Double = 0.00
    @Published var latestRHR: Double = 0.00
    @Published var latestHHR: Int = 0
    @Published var predictionResult: String = "Press button to predict"
    @Published var finalPrediction: String = ""
    @Published var lastUpdated: String = "Never"
    @Published var latestECGValues: [Double] = []    // <-- store ECG sample here

    // MARK: - Public API

    func requestAuthorization() {
        let typesToRead: Set = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
            HKObjectType.electrocardiogramType()
        ]

        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            if success {
                print("HealthKit access granted")
            } else {
                print("HealthKit access denied: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }

    func startSendingData() {
        timer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            self.fetchAndSendAllMetrics()
        }
    }

    deinit {
        timer?.invalidate()
    }

    // MARK: - Fetch and Send All Metrics

    func fetchAndSendAllMetrics() {
        latestHeartRate = 101.78
        let dispatchGroup = DispatchGroup()

        dispatchGroup.enter()
        fetchLatestHeartRate { dispatchGroup.leave() }

        dispatchGroup.enter()
        fetchLatestHRV { dispatchGroup.leave() }

        dispatchGroup.enter()
        fetchRestingHeartRate { dispatchGroup.leave() }

        dispatchGroup.enter()
        fetchHighHeartRateEvents { dispatchGroup.leave() }

        dispatchGroup.notify(queue: .main) {
            let payload: [String: Any] = [
                // "hr": latestHeartRate
                // "hrv": latestHRV
                // "rhr": latestRHR
                // "hhr": latestHHR
                
                // For Demo purposes, inputting values that guarantee a possible risk
                "hr": 101.78,
                "hrv": 29.28,
                "rhr": 113.83,
                "hhr": 2
            ]
            
//            self.sendJSON(to: "http://10.0.0.141:8000/send_all_metrics", payload: payload)
            self.sendJSON(to: "http://192.168.2.125:8000/send_all_metrics", payload: payload)
            

            self.updateLastUpdated()
        }
    }

    // MARK: - Fetch Methods

    func fetchLatestHeartRate(completion: @escaping () -> Void) {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            completion(); return
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: heartRateType,
                                  predicate: nil,
                                  limit: 1,
                                  sortDescriptors: [sort]) { _, samples, _ in
            if let sample = samples?.first as? HKQuantitySample {
                let bpm = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                DispatchQueue.main.async {
                    self.latestHeartRate = bpm
                    print("HR: \(bpm) bpm")
                }
            }
            completion()
        }
        healthStore.execute(query)
    }

    func fetchLatestHRV(completion: @escaping () -> Void) {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            completion(); return
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: hrvType,
                                  predicate: nil,
                                  limit: 1,
                                  sortDescriptors: [sort]) { _, samples, _ in
            if let sample = samples?.first as? HKQuantitySample {
                let ms = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
                DispatchQueue.main.async {
                    self.latestHRV = ms
                    print("HRV: \(ms) ms")
                }
            }
            completion()
        }
        healthStore.execute(query)
    }

    func fetchRestingHeartRate(completion: @escaping () -> Void) {
        guard let rhrType = HKQuantityType.quantityType(forIdentifier: .restingHeartRate) else {
            completion(); return
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: rhrType,
                                  predicate: nil,
                                  limit: 1,
                                  sortDescriptors: [sort]) { _, samples, _ in
            if let sample = samples?.first as? HKQuantitySample {
                let rhr = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                DispatchQueue.main.async {
                    self.latestRHR = rhr
                    print("RHR: \(rhr) bpm")
                }
            }
            completion()
        }
        healthStore.execute(query)
    }

    func fetchHighHeartRateEvents(completion: @escaping () -> Void) {
        guard let hrType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            completion(); return
        }

        let oneHourAgo = Calendar.current.date(byAdding: .hour, value: -1, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: oneHourAgo, end: Date(), options: [])

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: hrType,
                                  predicate: predicate,
                                  limit: HKObjectQueryNoLimit,
                                  sortDescriptors: [sort]) { _, samples, _ in
            let highHRCount = samples?.compactMap { $0 as? HKQuantitySample }
                .filter { $0.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute())) > 100 }
                .count ?? 0

            DispatchQueue.main.async {
                self.latestHHR = highHRCount
                print("HHR Events: \(highHRCount)")
            }
            completion()
        }
        healthStore.execute(query)
    }

    // MARK: - Fetch ECG Sample

    func fetchECGSample() {
        print("Fetching latest ECG Sample...")
        guard HKHealthStore.isHealthDataAvailable() else {
            print("Health data not available")
            return
        }
        let ecgType = HKObjectType.electrocardiogramType()
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: ecgType,
                                  predicate: nil,
                                  limit: 1,
                                  sortDescriptors: [sort]) { [weak self] _, samples, error in
            guard let self = self else { return }
            if let error = error {
                print("Error fetching ECG: \(error.localizedDescription)")
                return
            }
            guard let ecgSample = samples?.first as? HKElectrocardiogram else {
                print("No ECG sample found")
                return
            }
            var values: [Double] = []
            let ecgQuery = HKElectrocardiogramQuery(ecgSample) { _, result in
                switch result {
                case .measurement(let m):
                    if let v = m.quantity(for: .appleWatchSimilarToLeadI)?.doubleValue(for: .volt()) {
                        values.append(v)
                    }
                case .done:
                    DispatchQueue.main.async {
                        self.latestECGValues = values
                        print("Retrieved ECG Sample: \(values.count) values")
                    }
                case .error(let err):
                    print("Error processing ECG: \(err.localizedDescription)")
                @unknown default:
                    print("Unknown ECG state")
                }
            }
            self.healthStore.execute(ecgQuery)
        }
        healthStore.execute(query)
    }
    
    func sendTestECGSample() {
        print("Sending test ECG JSON to server…")
        guard let fileURL = Bundle.main.url(forResource: "204885", withExtension: "json") else {
            print("File not found in bundle")
            return
        }
        do {
            let jsonData = try Data(contentsOf: fileURL)
            if let obj = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
                print("Loaded test JSON keys: \(obj.keys)")
            }
            guard let url = URL(string: "http://192.168.2.125:8000/send_ecg") else {
                print("Invalid send_ecg URL")
                return
            }
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = jsonData

            URLSession.shared.dataTask(with: req) { data, resp, error in
                if let e = error {
                    print("Test ECG POST failed: \(e.localizedDescription)")
                    return
                }
                if let code = (resp as? HTTPURLResponse)?.statusCode {
                    print("[send_ecg] status: \(code)")
                }
                if let data = data,
                   let s = String(data: data, encoding: .utf8) {
                    print("[send_ecg] response: \(s)")
                    // parse and extract finalPrediction
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let final = json["finalPrediction"] as? String {
                        DispatchQueue.main.async {
                            self.finalPrediction = final
                        }
                        print("finalPrediction: \(final)")
                    }
                }
            }.resume()

        } catch {
            print("Failed to read 204885.json: \(error)")
        }
    }

    // MARK: - Send JSON to Server

    private func sendJSON(to urlString: String, payload: [String: Any]) {
        print("Sending to \(urlString) with payload: \(payload)")

        guard let url = URL(string: urlString),
              let jsonData = try? JSONSerialization.data(withJSONObject: payload) else {
            print("Invalid URL or JSON")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Request error: \(error.localizedDescription)")
                return
            }
            if let httpResponse = response as? HTTPURLResponse {
                print("[\(urlString)] Status: \(httpResponse.statusCode)")
            }
            if let data = data,
               let responseStr = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    self.predictionResult = responseStr
                }
                print("[\(urlString)] Response: \(responseStr)")
            }
        }.resume()
    }

    // MARK: - Last Updated

    private func updateLastUpdated() {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, h:mm a"
        self.lastUpdated = formatter.string(from: Date())
        print("Last Updated: \(self.lastUpdated)")
    }
}
