import requests

BASE = "http://127.0.0.1:8000"

payload = {
    "device_id": "demo-device-01",
    "network_type": "4G",
    "latitude": 28.6139,
    "longitude": 77.209,
    "downlink_mbps": 10.2,
    "rtt_ms": 92,
    "packet_loss_pct": 4.8,
    "store_telemetry": True,
}

resp = requests.post(f"{BASE}/predict", json=payload, timeout=30)
print(resp.status_code)
print(resp.json())
