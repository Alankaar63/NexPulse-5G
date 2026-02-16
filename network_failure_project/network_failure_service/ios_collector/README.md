# iOS Collector Starter

Starter SwiftUI collector for GPS + basic network context.

## Setup
1. Create a new SwiftUI iOS app in Xcode.
2. Replace `ContentView.swift` with:
   - `/Users/vivektripathi/Desktop/machineLearning/network_failure_service/ios_collector/ContentView.swift`
3. In `Info.plist`, add location usage description:
   - `NSLocationWhenInUseUsageDescription`
4. Replace endpoint URL:
   - `http://YOUR_SERVER_IP:8000/predict`

## Note
iOS limits direct low-level Wi-Fi telemetry in public APIs. For richer RF data, integrate router-side telemetry or MDM-controlled enterprise APIs.
