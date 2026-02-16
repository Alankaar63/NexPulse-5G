# Android Collector Starter

This folder contains a starter `MainActivity.kt` for a native Android telemetry collector.

## What it captures
- GPS location (if permission granted)
- Wi-Fi RSSI (`signal_strength_dbm`)
- Downlink bandwidth hint from network capabilities

## Setup steps
1. Create a new Android Studio project (Kotlin + Empty Activity).
2. Replace `MainActivity.kt` with `/Users/vivektripathi/Desktop/machineLearning/network_failure_service/android_collector/MainActivity.kt`.
3. Add permissions in `AndroidManifest.xml`:
   - `ACCESS_FINE_LOCATION`
   - `ACCESS_COARSE_LOCATION`
   - `ACCESS_WIFI_STATE`
   - `ACCESS_NETWORK_STATE`
   - `INTERNET`
4. Update endpoint URL in code:
   - `http://YOUR_SERVER_IP:8000/predict`
5. Add Google Play Services location dependency in `build.gradle`:
   - `implementation("com.google.android.gms:play-services-location:21.3.0")`

## Notes
- This is a starter implementation to provide raw telemetry missing from browsers.
- For production, replace blocking location logic with callback/Flow and add robust error handling.
