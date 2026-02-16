import requests

url = "http://127.0.0.1:5000/predict"
sample_data = {
    "Signal Strength (dBm)": -95,
    "BB60C Measurement (dBm)": -96,
    "srsRAN Measurement (dBm)": -97,
    "BladeRFxA9 Measurement (dBm)": -98,
    "Latency (ms)": 95,
    "Network Type_4G": 1,
    "Network Type_5G": 0
}

response = requests.post(url, json=sample_data)

if response.status_code == 200:
    print("Prediction Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
