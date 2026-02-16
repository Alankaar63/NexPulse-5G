const apiBase = window.location.origin;

const byId = (id) => document.getElementById(id);
const statusEl = byId("status");
const resultEl = byId("result");
const lanStatusEl = byId("lanStatus");
const lanResultEl = byId("lanResult");

const telemetry = {
  device_id: null,
  latitude: null,
  longitude: null,
  network_type: null,
  effective_type: null,
  downlink_mbps: null,
  rtt_ms: null,
  speed_test_download_mbps: null,
  speed_test_upload_mbps: null,
  latency_ms: null,
  jitter_ms: null,
  packet_loss_pct: null,
  signal_strength_dbm: null,
  bb60c_dbm: null,
  srsran_dbm: null,
  bladerfxa9_dbm: null,
  store_telemetry: true,
};

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#f88c8c" : "#8ef1c1";
}

function setLanStatus(message, isError = false) {
  lanStatusEl.textContent = message;
  lanStatusEl.style.color = isError ? "#f88c8c" : "#8ef1c1";
}

function stableDeviceId() {
  const key = "nexuspulse_device_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const generated = `web-${Math.random().toString(36).slice(2, 10)}-${Date.now().toString(36)}`;
  localStorage.setItem(key, generated);
  return generated;
}

function formatValue(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

function updateTelemetryCards() {
  byId("tDeviceId").textContent = formatValue(telemetry.device_id, 0);
  byId("tNetworkType").textContent = formatValue(telemetry.network_type, 0);
  byId("tEffectiveType").textContent = formatValue(telemetry.effective_type, 0);
  byId("tLatitude").textContent = formatValue(telemetry.latitude, 6);
  byId("tLongitude").textContent = formatValue(telemetry.longitude, 6);
  byId("tDownlink").textContent = formatValue(telemetry.downlink_mbps);
  byId("tRtt").textContent = formatValue(telemetry.rtt_ms);
  byId("tLatency").textContent = formatValue(telemetry.latency_ms);
  byId("tJitter").textContent = formatValue(telemetry.jitter_ms);
  byId("tPacketLoss").textContent = formatValue(telemetry.packet_loss_pct);
  byId("tDownload").textContent = formatValue(telemetry.speed_test_download_mbps);

  byId("metricLatency").textContent = formatValue(telemetry.latency_ms);
  byId("metricJitter").textContent = formatValue(telemetry.jitter_ms);
  byId("metricLoss").textContent = formatValue(telemetry.packet_loss_pct);
}

function resolveNetworkType() {
  const effective = telemetry.effective_type;
  if (!effective) return "Unknown";
  const mapping = {
    "slow-2g": "2G",
    "2g": "2G",
    "3g": "3G",
    "4g": "4G",
    "5g": "5G",
  };
  return mapping[effective.toLowerCase()] || "Unknown";
}

async function captureLocation() {
  if (!navigator.geolocation) {
    setStatus("Geolocation unavailable; continuing without GPS.");
    return;
  }

  await new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        telemetry.latitude = pos.coords.latitude;
        telemetry.longitude = pos.coords.longitude;
        resolve();
      },
      () => resolve(),
      { enableHighAccuracy: true, timeout: 10000 }
    );
  });
}

function captureBrowserNetworkHints() {
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (!connection) return;

  telemetry.downlink_mbps = Number.isFinite(connection.downlink) ? connection.downlink : telemetry.downlink_mbps;
  telemetry.rtt_ms = Number.isFinite(connection.rtt) ? connection.rtt : telemetry.rtt_ms;
  telemetry.effective_type = connection.effectiveType || telemetry.effective_type;
  telemetry.network_type = resolveNetworkType();
}

async function runDiagnostics() {
  setStatus("Running active diagnostics...");
  const pingCount = 6;
  const times = [];
  let failures = 0;

  for (let i = 0; i < pingCount; i += 1) {
    const start = performance.now();
    try {
      const response = await fetch(`${apiBase}/ping`, { cache: "no-store" });
      if (!response.ok) throw new Error("ping failed");
      times.push(performance.now() - start);
    } catch (_error) {
      failures += 1;
    }
  }

  if (times.length > 0) {
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const variance = times.reduce((sum, t) => sum + (t - avg) ** 2, 0) / times.length;
    telemetry.latency_ms = avg;
    telemetry.jitter_ms = Math.sqrt(variance);
  }

  telemetry.packet_loss_pct = (failures / pingCount) * 100;

  const dlStart = performance.now();
  const dlResponse = await fetch(`${apiBase}/download-test?size_kb=1024`, { cache: "no-store" });
  if (dlResponse.ok) {
    const bytes = 1024 * 1024;
    await dlResponse.arrayBuffer();
    const seconds = (performance.now() - dlStart) / 1000;
    telemetry.speed_test_download_mbps = ((bytes * 8) / seconds) / 1_000_000;
  }

  updateTelemetryCards();
  setStatus("Diagnostics complete.");
}

async function runPrediction() {
  setStatus("Running model inference...");
  const response = await fetch(`${apiBase}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(telemetry),
  });

  const rawText = await response.text();
  let data;
  try {
    data = rawText ? JSON.parse(rawText) : {};
  } catch (_err) {
    data = { detail: rawText || "Unknown server response" };
  }
  if (!response.ok) {
    throw new Error(data.detail || JSON.stringify(data));
  }

  resultEl.textContent = JSON.stringify(data, null, 2);
  if (data.maintenance_required) {
    if (data.ticket_created) {
      setStatus(`Maintenance required. Ticket created (${data.ticket_provider}${data.ticket_reference ? ` #${data.ticket_reference}` : ""}).`);
    } else {
      setStatus(`Maintenance required. Ticket not created${data.ticket_error ? `: ${data.ticket_error}` : "."}`, true);
    }
  } else {
    setStatus("No immediate maintenance required.");
  }
}

async function runAutoScanAndPredict() {
  try {
    setStatus("Starting autonomous telemetry capture...");
    telemetry.device_id = stableDeviceId();
    captureBrowserNetworkHints();
    updateTelemetryCards();

    await captureLocation();
    captureBrowserNetworkHints();
    telemetry.network_type = resolveNetworkType();
    updateTelemetryCards();

    await runDiagnostics();
    await runPrediction();
  } catch (error) {
    setStatus(`Auto scan failed: ${error.message}`, true);
  }
}

function setLanCard(id, value) {
  const el = byId(id);
  if (!el) return;
  el.textContent = value === null || value === undefined ? "--" : String(value);
}

async function runLanHealthCheck() {
  try {
    setLanStatus("Scanning LAN and computing risk...");
    const subnet = (byId("lanSubnet").value || "").trim() || null;
    const threshold = Number(byId("lanThreshold").value || "45");

    const response = await fetch(`${apiBase}/lan-health`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        subnet,
        alert_device_threshold: Number.isFinite(threshold) ? threshold : 45,
        target_host: "1.1.1.1",
        device_id: telemetry.device_id,
      }),
    });

    const rawText = await response.text();
    let data;
    try {
      data = rawText ? JSON.parse(rawText) : {};
    } catch (_err) {
      data = { detail: rawText || "Unknown server response" };
    }
    if (!response.ok) {
      throw new Error(data.detail || JSON.stringify(data));
    }

    lanResultEl.textContent = JSON.stringify(data, null, 2);
    setLanCard("lanSubnetOut", data.subnet_scanned);
    setLanCard("lanDeviceCount", data.device_count);
    setLanCard("lanScanMethod", data.scan_method);
    setLanCard("lanFailureProb", data.failure_probability);
    setLanCard("lanMaintenance", data.maintenance_required ? "Required" : "Not required");
    setLanCard(
      "lanTicket",
      data.ticket_created
        ? `Created (${data.ticket_provider}${data.ticket_reference ? ` #${data.ticket_reference}` : ""})`
        : (data.ticket_error ? `Failed (${data.ticket_error})` : "Not created")
    );
    setLanCard("lanAlert", data.congestion_risk_triggered ? "TRIGGERED" : "Normal");

    if (data.congestion_risk_triggered) {
      setLanStatus("Alert triggered: high active devices and high failure probability.", true);
    } else {
      setLanStatus("LAN health check complete.");
    }
  } catch (error) {
    setLanStatus(`LAN health check failed: ${error.message}`, true);
  }
}

function setupRevealAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("in");
      });
    },
    { threshold: 0.18 }
  );

  document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
}

function setupParallax() {
  const hero = byId("hero");
  const orb1 = document.querySelector(".orb-1");
  const orb2 = document.querySelector(".orb-2");
  if (!hero || !orb1 || !orb2) return;

  hero.addEventListener("mousemove", (event) => {
    const x = (event.clientX / window.innerWidth - 0.5) * 18;
    const y = (event.clientY / window.innerHeight - 0.5) * 18;
    orb1.style.transform = `translate(${x}px, ${y}px)`;
    orb2.style.transform = `translate(${-x}px, ${-y}px)`;
  });
}

function setupNavScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      const targetId = link.getAttribute("href");
      const target = document.querySelector(targetId);
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

byId("autoScanBtn").addEventListener("click", runAutoScanAndPredict);
byId("runDiagnosticsBtn").addEventListener("click", runDiagnostics);
byId("runDiagnosticsHero").addEventListener("click", runDiagnostics);
byId("lanHealthBtn").addEventListener("click", runLanHealthCheck);

setupRevealAnimations();
setupParallax();
setupNavScroll();
telemetry.device_id = stableDeviceId();
captureBrowserNetworkHints();
updateTelemetryCards();
