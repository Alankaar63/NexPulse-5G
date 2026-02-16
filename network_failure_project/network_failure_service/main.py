from __future__ import annotations

import json
import ipaddress
import os
import re
import socket
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
import traceback
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"
ENCODER_PATH = PROJECT_ROOT / "ohe.pkl"
FEATURES_PATH = PROJECT_ROOT / "features.pkl"
DB_PATH = BASE_DIR / "network_failure.db"

TICKETING_PROVIDER = os.getenv("TICKETING_PROVIDER", "none").strip().lower()
TICKET_WEBHOOK_URL = os.getenv("TICKET_WEBHOOK_URL", "").strip()
ZAMMAD_URL = os.getenv("ZAMMAD_URL", "").strip().rstrip("/")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "").strip()
ZAMMAD_GROUP = os.getenv("ZAMMAD_GROUP", "Users").strip()

REQUIRED_NUMERIC_FEATURES = [
    "Signal Strength (dBm)",
    "BB60C Measurement (dBm)",
    "srsRAN Measurement (dBm)",
    "BladeRFxA9 Measurement (dBm)",
    "Latency (ms)",
]


class TelemetryRequest(BaseModel):
    device_id: Optional[str] = Field(default=None, max_length=120)
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    network_type: Optional[str] = Field(default=None, max_length=32)
    effective_type: Optional[str] = Field(default=None, max_length=32)

    downlink_mbps: Optional[float] = None
    rtt_ms: Optional[float] = None
    speed_test_download_mbps: Optional[float] = None
    speed_test_upload_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    packet_loss_pct: Optional[float] = None

    signal_strength_dbm: Optional[float] = None
    bb60c_dbm: Optional[float] = None
    srsran_dbm: Optional[float] = None
    bladerfxa9_dbm: Optional[float] = None


class PredictRequest(TelemetryRequest):
    store_telemetry: bool = True
    telemetry_id: Optional[int] = None


class FeedbackRequest(BaseModel):
    prediction_id: int
    actual_failure: bool
    maintenance_performed: bool
    notes: Optional[str] = Field(default=None, max_length=1000)


class LanHealthRequest(BaseModel):
    subnet: Optional[str] = Field(default=None, max_length=32)
    alert_device_threshold: int = Field(default=45, ge=1, le=10000)
    target_host: str = Field(default="1.1.1.1", max_length=64)
    device_id: Optional[str] = Field(default=None, max_length=120)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def model_to_json(model_obj: BaseModel) -> str:
    # Support both Pydantic v1 and v2 environments.
    if hasattr(model_obj, "model_dump_json"):
        return model_obj.model_dump_json()
    return model_obj.json()


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            device_id TEXT,
            latitude REAL,
            longitude REAL,
            network_type TEXT,
            effective_type TEXT,
            downlink_mbps REAL,
            rtt_ms REAL,
            speed_test_download_mbps REAL,
            speed_test_upload_mbps REAL,
            latency_ms REAL,
            jitter_ms REAL,
            packet_loss_pct REAL,
            signal_strength_dbm REAL,
            bb60c_dbm REAL,
            srsran_dbm REAL,
            bladerfxa9_dbm REAL,
            raw_payload TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            telemetry_id INTEGER,
            failure_probability REAL NOT NULL,
            failure_prediction INTEGER NOT NULL,
            maintenance_required INTEGER NOT NULL,
            confidence REAL NOT NULL,
            recommended_action TEXT NOT NULL,
            model_version TEXT NOT NULL,
            FOREIGN KEY(telemetry_id) REFERENCES telemetry(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            prediction_id INTEGER NOT NULL,
            actual_failure INTEGER NOT NULL,
            maintenance_performed INTEGER NOT NULL,
            notes TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS maintenance_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            prediction_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            external_ticket_id TEXT,
            status TEXT NOT NULL,
            payload_json TEXT,
            response_json TEXT,
            error_message TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )
        """
    )
    conn.commit()
    conn.close()


def insert_telemetry(payload: TelemetryRequest) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO telemetry (
            created_at, device_id, latitude, longitude, network_type, effective_type,
            downlink_mbps, rtt_ms, speed_test_download_mbps, speed_test_upload_mbps,
            latency_ms, jitter_ms, packet_loss_pct,
            signal_strength_dbm, bb60c_dbm, srsran_dbm, bladerfxa9_dbm, raw_payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(time.time()),
            payload.device_id,
            payload.latitude,
            payload.longitude,
            payload.network_type,
            payload.effective_type,
            payload.downlink_mbps,
            payload.rtt_ms,
            payload.speed_test_download_mbps,
            payload.speed_test_upload_mbps,
            payload.latency_ms,
            payload.jitter_ms,
            payload.packet_loss_pct,
            payload.signal_strength_dbm,
            payload.bb60c_dbm,
            payload.srsran_dbm,
            payload.bladerfxa9_dbm,
            model_to_json(payload),
        ),
    )
    telemetry_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(telemetry_id)


def insert_prediction(
    telemetry_id: Optional[int],
    failure_probability: float,
    failure_prediction: int,
    maintenance_required: bool,
    confidence: float,
    recommended_action: str,
    model_version: str,
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (
            created_at, telemetry_id, failure_probability, failure_prediction,
            maintenance_required, confidence, recommended_action, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(time.time()),
            telemetry_id,
            failure_probability,
            failure_prediction,
            int(maintenance_required),
            confidence,
            recommended_action,
            model_version,
        ),
    )
    prediction_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(prediction_id)


def insert_feedback(payload: FeedbackRequest) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (
            created_at, prediction_id, actual_failure, maintenance_performed, notes
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(time.time()),
            payload.prediction_id,
            int(payload.actual_failure),
            int(payload.maintenance_performed),
            payload.notes,
        ),
    )
    feedback_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(feedback_id)


def insert_ticket(
    prediction_id: int,
    provider: str,
    status: str,
    payload: dict[str, Any],
    response: Optional[dict[str, Any]] = None,
    external_ticket_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO maintenance_tickets (
            created_at, prediction_id, provider, external_ticket_id,
            status, payload_json, response_json, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(time.time()),
            prediction_id,
            provider,
            external_ticket_id,
            status,
            json.dumps(payload),
            json.dumps(response) if response is not None else None,
            error_message,
        ),
    )
    ticket_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(ticket_id)


def list_tickets(limit: int = 25) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, created_at, prediction_id, provider, external_ticket_id, status, error_message
        FROM maintenance_tickets
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def map_network_type(payload: TelemetryRequest) -> str:
    if payload.network_type:
        return payload.network_type
    if not payload.effective_type:
        return "Unknown"

    normalized = payload.effective_type.strip().lower()
    mapping = {
        "slow-2g": "2G",
        "2g": "2G",
        "3g": "3G",
        "4g": "4G",
        "5g": "5G",
    }
    return mapping.get(normalized, "Unknown")


def estimate_radio_features(payload: TelemetryRequest) -> dict[str, float]:
    downlink = payload.speed_test_download_mbps or payload.downlink_mbps or 25.0
    rtt = payload.latency_ms or payload.rtt_ms or 50.0

    # Browser APIs cannot access raw Wi-Fi radio values.
    # This maps available web diagnostics into proxy levels.
    base_signal = -68.0
    base_signal -= max(0.0, 40.0 - downlink) * 0.55
    base_signal -= max(0.0, rtt - 20.0) * 0.30
    base_signal = clamp(base_signal, -120.0, -45.0)

    signal_strength = payload.signal_strength_dbm
    if signal_strength is None:
        signal_strength = base_signal

    bb60c = payload.bb60c_dbm
    if bb60c is None:
        bb60c = clamp(signal_strength - 1.5, -120.0, -40.0)

    srsran = payload.srsran_dbm
    if srsran is None:
        srsran = clamp(signal_strength - 2.0, -120.0, -40.0)

    bladerf = payload.bladerfxa9_dbm
    if bladerf is None:
        bladerf = clamp(signal_strength - 2.5, -120.0, -40.0)

    latency = payload.latency_ms or payload.rtt_ms
    if latency is None:
        latency = clamp(120.0 - downlink * 1.5, 10.0, 300.0)

    return {
        "Signal Strength (dBm)": float(signal_strength),
        "BB60C Measurement (dBm)": float(bb60c),
        "srsRAN Measurement (dBm)": float(srsran),
        "BladeRFxA9 Measurement (dBm)": float(bladerf),
        "Latency (ms)": float(latency),
    }


def get_encoder_feature_names(ohe: Any) -> list[str]:
    if hasattr(ohe, "get_feature_names_out"):
        return list(ohe.get_feature_names_out(["Network Type"]))
    # Compatibility fallback for older sklearn variants.
    categories = getattr(ohe, "categories_", [["Unknown"]])[0]
    return [f"Network Type_{c}" for c in categories]


def build_model_frame(
    payload: TelemetryRequest,
    ohe: Any,
    expected_features: list[str],
) -> pd.DataFrame:
    radio_features = estimate_radio_features(payload)
    network_type = map_network_type(payload)

    base = pd.DataFrame([radio_features])
    encoded = ohe.transform(pd.DataFrame([{"Network Type": network_type}]))
    encoded_df = pd.DataFrame(encoded, columns=get_encoder_feature_names(ohe))
    full_df = pd.concat([base, encoded_df], axis=1)

    aligned = full_df.reindex(columns=expected_features, fill_value=0.0)
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)
    for col in REQUIRED_NUMERIC_FEATURES:
        if col not in aligned.columns:
            raise HTTPException(status_code=500, detail=f"Model feature missing: {col}")
    return aligned


def infer(payload: TelemetryRequest, model: Any, ohe: Any, trained_features: list[str]) -> tuple[int, float, bool, float, str]:
    expected_features = list(getattr(model, "feature_names_in_", trained_features))
    model_input = build_model_frame(payload, ohe=ohe, expected_features=expected_features)
    prediction = int(model.predict(model_input)[0])

    if hasattr(model, "predict_proba"):
        failure_probability = float(model.predict_proba(model_input)[0][1])
    else:
        failure_probability = float(prediction)

    failure_probability = clamp(failure_probability, 0.0, 1.0)

    radio = estimate_radio_features(payload)
    packet_loss = payload.packet_loss_pct or 0.0
    maintenance_required = (
        failure_probability >= 0.60
        or packet_loss >= 5.0
        or (
            radio["Signal Strength (dBm)"] <= -93.0
            and radio["Latency (ms)"] >= 90.0
        )
    )

    if maintenance_required and failure_probability >= 0.75:
        recommended_action = "Urgent field check: inspect access point and backhaul."
    elif maintenance_required:
        recommended_action = "Schedule maintenance: validate router load, interference, and RF channel."
    else:
        recommended_action = "No immediate maintenance required; continue monitoring."

    confidence = float(abs(failure_probability - 0.5) * 2.0)
    return prediction, failure_probability, maintenance_required, confidence, recommended_action


def run_command(args: list[str], timeout: int = 20) -> str:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.stdout or ""


def count_devices_via_nmap(subnet: str, timeout: int) -> int:
    # Use tighter scan settings to avoid long hangs on managed/campus Wi-Fi.
    out = run_command(
        [
            "nmap",
            "-sn",
            "-n",
            "-T4",
            "--max-retries",
            "1",
            "--host-timeout",
            "700ms",
            subnet,
            "-oG",
            "-",
        ],
        timeout=timeout,
    )
    return len(re.findall(r"Status:\s+Up", out))


def count_devices_via_arp() -> int:
    # Use numeric output to avoid reverse-DNS stalls.
    out = run_command(["arp", "-an"], timeout=4)
    ips = set(re.findall(r"\((\d+\.\d+\.\d+\.\d+)\)", out))
    return len(ips)


def get_wifi_rssi_dbm() -> Optional[float]:
    mac_airport_paths = [
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/A/Resources/airport",
    ]
    for airport_bin in mac_airport_paths:
        if os.path.exists(airport_bin):
            output = run_command([airport_bin, "-I"], timeout=8)
            match = re.search(r"agrCtlRSSI:\s*(-?\d+)", output)
            if match:
                return float(match.group(1))

    linux_output = run_command(["sh", "-lc", "iwconfig 2>/dev/null"], timeout=8)
    match = re.search(r"Signal level=(-?\d+)\s*dBm", linux_output)
    if match:
        return float(match.group(1))
    return None


def parse_ping_stats(ping_output: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    packet_loss = None
    latency_avg = None
    jitter = None

    loss_match = re.search(r"([0-9.]+)%\s*packet loss", ping_output)
    if loss_match:
        packet_loss = float(loss_match.group(1))

    rtt_match = re.search(
        r"(?:round-trip|rtt)\s+min/avg/max/(?:stddev|mdev)\s*=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)",
        ping_output,
    )
    if rtt_match:
        latency_avg = float(rtt_match.group(2))
        jitter = float(rtt_match.group(4))

    return latency_avg, jitter, packet_loss


def default_local_subnet() -> str:
    # Auto-detect local IP and default to /24 scan for responsiveness.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
    finally:
        sock.close()
    net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
    return str(net)


def validate_private_subnet(subnet: str) -> str:
    try:
        network = ipaddress.ip_network(subnet, strict=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid subnet: {exc}") from exc
    if not network.is_private:
        raise HTTPException(status_code=400, detail="Only private LAN ranges are allowed.")
    return str(network)


def collect_lan_health(subnet: str, target_host: str) -> dict[str, Any]:
    device_count = 0
    scan_method = "nmap"
    try:
        device_count = count_devices_via_nmap(subnet=subnet, timeout=25)
    except subprocess.TimeoutExpired:
        try:
            # Fallback to even shorter timeout before dropping to ARP table.
            device_count = count_devices_via_nmap(subnet=subnet, timeout=12)
        except subprocess.TimeoutExpired:
            try:
                device_count = count_devices_via_arp()
                scan_method = "arp-fallback"
            except subprocess.TimeoutExpired:
                # Do not fail LAN health end-to-end if host scan methods stall.
                device_count = 0
                scan_method = "degraded-timeout"

    ping_out = run_command(["ping", "-c", "8", target_host], timeout=20)
    latency_avg, jitter, packet_loss = parse_ping_stats(ping_out)

    rssi = get_wifi_rssi_dbm()

    return {
        "device_count": float(device_count),
        "scan_method": scan_method,
        "signal_strength_dbm": rssi,
        "latency_ms": latency_avg,
        "jitter_ms": jitter,
        "packet_loss_pct": packet_loss,
    }


def run_prediction_pipeline(payload: PredictRequest) -> dict[str, Any]:
    telemetry_id = payload.telemetry_id
    if payload.store_telemetry and telemetry_id is None:
        telemetry_id = insert_telemetry(payload)

    prediction, failure_probability, maintenance_required, confidence, recommended_action = infer(
        payload,
        model=model,
        ohe=ohe,
        trained_features=trained_features,
    )

    prediction_id = insert_prediction(
        telemetry_id=telemetry_id,
        failure_probability=failure_probability,
        failure_prediction=prediction,
        maintenance_required=maintenance_required,
        confidence=confidence,
        recommended_action=recommended_action,
        model_version=model_version,
    )

    ticket_info: dict[str, Any] = {
        "ticket_created": False,
        "ticket_provider": None,
        "ticket_reference": None,
        "ticket_error": None,
    }
    if maintenance_required:
        ticket_payload = create_ticket_payload(
            payload,
            prediction_id=prediction_id,
            failure_probability=failure_probability,
            confidence=confidence,
            recommended_action=recommended_action,
        )
        provider, external_id, response_json, error_message = create_external_ticket(ticket_payload)
        status = "created" if external_id or (provider == "webhook" and error_message is None) else "failed"
        insert_ticket(
            prediction_id=prediction_id,
            provider=provider,
            status=status,
            payload=ticket_payload,
            response=response_json,
            external_ticket_id=external_id,
            error_message=error_message,
        )
        ticket_info = {
            "ticket_created": status == "created",
            "ticket_provider": provider,
            "ticket_reference": external_id,
            "ticket_error": error_message,
        }

    return {
        "prediction_id": prediction_id,
        "telemetry_id": telemetry_id,
        "failure_prediction": prediction,
        "failure_probability": round(failure_probability, 4),
        "maintenance_required": maintenance_required,
        "confidence": round(confidence, 4),
        "recommended_action": recommended_action,
        "model_version": model_version,
        **ticket_info,
    }


def post_json(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        raw = resp.read().decode("utf-8")
        if not raw:
            return resp.status, {}
        return resp.status, json.loads(raw)


def create_ticket_payload(
    payload: PredictRequest,
    prediction_id: int,
    failure_probability: float,
    confidence: float,
    recommended_action: str,
) -> dict[str, Any]:
    return {
        "title": f"Network maintenance alert: prediction #{prediction_id}",
        "severity": "high" if failure_probability >= 0.75 else "medium",
        "prediction_id": prediction_id,
        "device_id": payload.device_id,
        "location": {"latitude": payload.latitude, "longitude": payload.longitude},
        "network_type": payload.network_type,
        "effective_type": payload.effective_type,
        "failure_probability": round(failure_probability, 4),
        "confidence": round(confidence, 4),
        "recommended_action": recommended_action,
        "telemetry": {
            "downlink_mbps": payload.downlink_mbps,
            "rtt_ms": payload.rtt_ms,
            "latency_ms": payload.latency_ms,
            "jitter_ms": payload.jitter_ms,
            "packet_loss_pct": payload.packet_loss_pct,
            "speed_test_download_mbps": payload.speed_test_download_mbps,
        },
    }


def create_external_ticket(ticket_payload: dict[str, Any]) -> tuple[str, Optional[str], Optional[dict[str, Any]], Optional[str]]:
    provider = TICKETING_PROVIDER

    if provider == "none":
        return "none", None, None, "Ticketing provider disabled"

    try:
        if provider == "webhook":
            if not TICKET_WEBHOOK_URL:
                return "webhook", None, None, "TICKET_WEBHOOK_URL is not configured"
            status_code, response_json = post_json(TICKET_WEBHOOK_URL, ticket_payload)
            external_id = str(response_json.get("id") or response_json.get("ticket_id") or "") or None
            return "webhook", external_id, {"status_code": status_code, "body": response_json}, None

        if provider == "zammad":
            if not ZAMMAD_URL or not ZAMMAD_TOKEN:
                return "zammad", None, None, "ZAMMAD_URL or ZAMMAD_TOKEN is not configured"

            z_payload = {
                "title": ticket_payload["title"],
                "group": ZAMMAD_GROUP,
                "customer": ticket_payload.get("device_id") or "network.monitor@local",
                "article": {
                    "subject": ticket_payload["title"],
                    "body": json.dumps(ticket_payload, indent=2),
                    "type": "note",
                    "internal": False,
                },
            }
            headers = {"Authorization": f"Token token={ZAMMAD_TOKEN}"}
            status_code, response_json = post_json(f"{ZAMMAD_URL}/api/v1/tickets", z_payload, headers=headers)
            external_id = str(response_json.get("id") or "") or None
            return "zammad", external_id, {"status_code": status_code, "body": response_json}, None

        return provider, None, None, f"Unsupported TICKETING_PROVIDER: {provider}"
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        return provider, None, None, str(exc)


if not MODEL_PATH.exists() or not ENCODER_PATH.exists() or not FEATURES_PATH.exists():
    raise RuntimeError("Missing model artifacts: model.pkl, ohe.pkl, or features.pkl")

model = joblib.load(MODEL_PATH)
ohe = joblib.load(ENCODER_PATH)
trained_features = joblib.load(FEATURES_PATH)
model_version = f"rf:{MODEL_PATH.stat().st_mtime_ns}"

init_db()

app = FastAPI(title="Network Failure Predictor", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": model_version,
        "ticketing_provider": TICKETING_PROVIDER,
    }


@app.get("/ping")
def ping() -> dict[str, int]:
    return {"server_time": int(time.time() * 1000)}


@app.get("/download-test")
def download_test(size_kb: int = 1024) -> Response:
    if size_kb < 64 or size_kb > 10240:
        raise HTTPException(status_code=400, detail="size_kb must be between 64 and 10240")
    payload = b"0" * (size_kb * 1024)
    return Response(content=payload, media_type="application/octet-stream")


@app.post("/upload-test")
async def upload_test(request: Request) -> dict[str, int]:
    body = await request.body()
    return {"received_bytes": len(body)}


@app.post("/telemetry")
def telemetry(payload: TelemetryRequest) -> dict[str, int]:
    telemetry_id = insert_telemetry(payload)
    return {"telemetry_id": telemetry_id}


@app.get("/tickets")
def tickets(limit: int = 25) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be in range [1, 200]")
    return {"items": list_tickets(limit=limit)}


@app.post("/predict")
def predict(payload: PredictRequest) -> JSONResponse:
    try:
        return JSONResponse(run_prediction_pipeline(payload))
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {"detail": f"{type(exc).__name__}: {exc}"},
            status_code=500,
        )


@app.post("/lan-health")
def lan_health(payload: LanHealthRequest) -> JSONResponse:
    try:
        subnet = validate_private_subnet(payload.subnet) if payload.subnet else default_local_subnet()
        lan_metrics = collect_lan_health(subnet=subnet, target_host=payload.target_host)

        predict_payload = PredictRequest(
            device_id=payload.device_id or "lan-health-agent",
            network_type="WiFi",
            effective_type="wifi",
            downlink_mbps=None,
            rtt_ms=lan_metrics["latency_ms"],
            speed_test_download_mbps=None,
            speed_test_upload_mbps=None,
            latency_ms=lan_metrics["latency_ms"],
            jitter_ms=lan_metrics["jitter_ms"],
            packet_loss_pct=lan_metrics["packet_loss_pct"],
            signal_strength_dbm=lan_metrics["signal_strength_dbm"],
            bb60c_dbm=None,
            srsran_dbm=None,
            bladerfxa9_dbm=None,
            store_telemetry=True,
        )

        prediction_result = run_prediction_pipeline(predict_payload)
        device_count = int(lan_metrics["device_count"] or 0)
        failure_probability = float(prediction_result["failure_probability"])
        congestion_risk_triggered = (
            device_count > payload.alert_device_threshold and failure_probability > 0.7
        )

        return JSONResponse(
            {
                "subnet_scanned": subnet,
                "device_count": device_count,
                "scan_method": lan_metrics.get("scan_method"),
                "alert_device_threshold": payload.alert_device_threshold,
                "congestion_risk_triggered": congestion_risk_triggered,
                "signal_strength_dbm": lan_metrics["signal_strength_dbm"],
                "latency_ms": lan_metrics["latency_ms"],
                "jitter_ms": lan_metrics["jitter_ms"],
                "packet_loss_pct": lan_metrics["packet_loss_pct"],
                **prediction_result,
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {"detail": f"{type(exc).__name__}: {exc}"},
            status_code=500,
        )


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, int]:
    feedback_id = insert_feedback(payload)
    return {"feedback_id": feedback_id}


static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="app")
