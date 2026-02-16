import SwiftUI
import CoreLocation
import Network

struct ContentView: View {
    @StateObject private var locationManager = LocationManager()
    @State private var status: String = "Ready"

    var body: some View {
        VStack(spacing: 16) {
            Text("Network Collector")
                .font(.title2)
            Button("Send Telemetry") {
                Task {
                    await sendTelemetry()
                }
            }
            Text(status)
                .font(.footnote)
                .foregroundColor(.gray)
                .padding()
        }
        .padding()
    }

    func sendTelemetry() async {
        let monitor = NWPathMonitor()
        let queue = DispatchQueue(label: "net.monitor")
        monitor.start(queue: queue)

        let location = locationManager.lastLocation
        let payload: [String: Any] = [
            "device_id": UIDevice.current.name,
            "network_type": "WiFi",
            "latitude": location?.coordinate.latitude as Any,
            "longitude": location?.coordinate.longitude as Any,
            "store_telemetry": true
        ]

        guard let url = URL(string: "http://YOUR_SERVER_IP:8000/predict") else {
            status = "Invalid URL"
            return
        }

        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = data

            let (responseData, _) = try await URLSession.shared.data(for: request)
            status = String(data: responseData, encoding: .utf8) ?? "No response"
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}

final class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    @Published var lastLocation: CLLocation?

    override init() {
        super.init()
        manager.delegate = self
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        lastLocation = locations.last
    }
}
