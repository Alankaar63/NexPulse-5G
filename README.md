# NexPulse-5G (Network Failure Prediction Software)

This folder contains the complete network failure prediction system that was upgraded from a hackathon model to a deployable platform.

## Contents
- `network_failure_service/` API, website, LAN health scanner, ticket automation, retraining scripts
- `NetworkFailurePred.py` original training script
- `signal_metrics.csv` training dataset
- `model.pkl`, `ohe.pkl`, `features.pkl` model artifacts
- `docker-compose.network-failure.yml` container orchestration
- `render.yaml` cloud deployment template

## Quick start

```bash
cd /Users/vivektripathi/Desktop/machineLearning
./network_failure_service/start_local.sh
```

Open `http://127.0.0.1:8000/`.

## Quick Links
- Android Studio app folder: [network_failure_service/android_studio_app](https://github.com/Alankaar63/AI-Driven-Network-Failure-Prediction/tree/main/network_failure_service/android_studio_app)
- Download full source ZIP: [main.zip](https://github.com/Alankaar63/AI-Driven-Network-Failure-Prediction/archive/refs/heads/main.zip)
