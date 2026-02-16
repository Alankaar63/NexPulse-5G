from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "network_failure_service"
DB_PATH = SERVICE_DIR / "network_failure.db"
DATASET_PATH = ROOT / "signal_metrics.csv"

BASE_MODEL = ROOT / "model.pkl"
BASE_OHE = ROOT / "ohe.pkl"
BASE_FEATURES = ROOT / "features.pkl"

ARTIFACTS_DIR = SERVICE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def map_network_type(raw: str | None, effective: str | None) -> str:
    if raw:
        return raw
    if not effective:
        return "Unknown"
    normalized = effective.strip().lower()
    mapping = {"2g": "2G", "3g": "3G", "4g": "4G", "5g": "5G", "slow-2g": "2G"}
    return mapping.get(normalized, "Unknown")


def estimate_radio_features(row: dict) -> dict[str, float]:
    downlink = row.get("speed_test_download_mbps") or row.get("downlink_mbps") or 25.0
    rtt = row.get("latency_ms") or row.get("rtt_ms") or 50.0

    base_signal = -68.0
    base_signal -= max(0.0, 40.0 - downlink) * 0.55
    base_signal -= max(0.0, rtt - 20.0) * 0.30
    base_signal = clamp(base_signal, -120.0, -45.0)

    signal = row.get("signal_strength_dbm")
    if signal is None:
        signal = base_signal

    bb60c = row.get("bb60c_dbm")
    if bb60c is None:
        bb60c = clamp(signal - 1.5, -120.0, -40.0)

    srsran = row.get("srsran_dbm")
    if srsran is None:
        srsran = clamp(signal - 2.0, -120.0, -40.0)

    bladerf = row.get("bladerfxa9_dbm")
    if bladerf is None:
        bladerf = clamp(signal - 2.5, -120.0, -40.0)

    latency = row.get("latency_ms") or row.get("rtt_ms")
    if latency is None:
        latency = clamp(120.0 - downlink * 1.5, 10.0, 300.0)

    return {
        "Signal Strength (dBm)": float(signal),
        "BB60C Measurement (dBm)": float(bb60c),
        "srsRAN Measurement (dBm)": float(srsran),
        "BladeRFxA9 Measurement (dBm)": float(bladerf),
        "Latency (ms)": float(latency),
    }


def load_base_training_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing base dataset: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    data["Failure"] = (
        (data["Signal Strength (dBm)"] < -93)
        | (
            (data["BB60C Measurement (dBm)"] < -93)
            & (data["srsRAN Measurement (dBm)"] < -93)
            & (data["BladeRFxA9 Measurement (dBm)"] < -93)
            & (data["Latency (ms)"] > 90)
        )
    ).astype(int)

    data["Network Type"] = data["Network Type"].fillna("Unknown")
    cols = [
        "Signal Strength (dBm)",
        "BB60C Measurement (dBm)",
        "srsRAN Measurement (dBm)",
        "BladeRFxA9 Measurement (dBm)",
        "Latency (ms)",
        "Network Type",
        "Failure",
    ]
    return data[cols]


def load_feedback_training_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=[
            "Signal Strength (dBm)",
            "BB60C Measurement (dBm)",
            "srsRAN Measurement (dBm)",
            "BladeRFxA9 Measurement (dBm)",
            "Latency (ms)",
            "Network Type",
            "Failure",
        ])

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
          t.network_type, t.effective_type,
          t.downlink_mbps, t.rtt_ms, t.speed_test_download_mbps, t.latency_ms,
          t.signal_strength_dbm, t.bb60c_dbm, t.srsran_dbm, t.bladerfxa9_dbm,
          f.actual_failure
        FROM feedback f
        JOIN predictions p ON p.id = f.prediction_id
        JOIN telemetry t ON t.id = p.telemetry_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    mapped_rows = []
    for r in rows:
        row = dict(r)
        features = estimate_radio_features(row)
        features["Network Type"] = map_network_type(row.get("network_type"), row.get("effective_type"))
        features["Failure"] = int(row.get("actual_failure") or 0)
        mapped_rows.append(features)

    if not mapped_rows:
        return pd.DataFrame(columns=[
            "Signal Strength (dBm)",
            "BB60C Measurement (dBm)",
            "srsRAN Measurement (dBm)",
            "BladeRFxA9 Measurement (dBm)",
            "Latency (ms)",
            "Network Type",
            "Failure",
        ])

    return pd.DataFrame(mapped_rows)


def train_and_save(promote: bool) -> None:
    base_df = load_base_training_data()
    feedback_df = load_feedback_training_data()

    combined = pd.concat([base_df, feedback_df], ignore_index=True)
    combined["Network Type"] = combined["Network Type"].fillna("Unknown")

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(combined[["Network Type"]])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(["Network Type"]))

    X = pd.concat(
        [
            combined[
                [
                    "Signal Strength (dBm)",
                    "BB60C Measurement (dBm)",
                    "srsRAN Measurement (dBm)",
                    "BladeRFxA9 Measurement (dBm)",
                    "Latency (ms)",
                ]
            ].reset_index(drop=True),
            encoded_df.reset_index(drop=True),
        ],
        axis=1,
    )
    y = combined["Failure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=None, class_weight="balanced")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    report = classification_report(y_test, pred, output_dict=True)

    stamp = str(int(time.time()))
    model_path = ARTIFACTS_DIR / f"model_{stamp}.pkl"
    ohe_path = ARTIFACTS_DIR / f"ohe_{stamp}.pkl"
    features_path = ARTIFACTS_DIR / f"features_{stamp}.pkl"
    metrics_path = ARTIFACTS_DIR / f"metrics_{stamp}.json"

    joblib.dump(model, model_path)
    joblib.dump(ohe, ohe_path)
    joblib.dump(list(X.columns), features_path)

    metrics_payload = {
        "created_at": int(stamp),
        "accuracy": acc,
        "num_rows_total": int(len(combined)),
        "num_rows_feedback": int(len(feedback_df)),
        "report": report,
        "artifacts": {
            "model": str(model_path),
            "ohe": str(ohe_path),
            "features": str(features_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if promote:
        joblib.dump(model, BASE_MODEL)
        joblib.dump(ohe, BASE_OHE)
        joblib.dump(list(X.columns), BASE_FEATURES)

    print(json.dumps(metrics_payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain network failure model using feedback + base dataset")
    parser.add_argument("--promote", action="store_true", help="Replace root model artifacts after retraining")
    args = parser.parse_args()
    train_and_save(promote=args.promote)


if __name__ == "__main__":
    main()
