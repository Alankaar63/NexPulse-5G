from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "network_failure_service" / "artifacts"

if not ARTIFACTS.exists():
    raise SystemExit("No artifacts directory found. Run retrain_model.py first.")

model_files = sorted(ARTIFACTS.glob("model_*.pkl"))
ohe_files = sorted(ARTIFACTS.glob("ohe_*.pkl"))
features_files = sorted(ARTIFACTS.glob("features_*.pkl"))

if not model_files or not ohe_files or not features_files:
    raise SystemExit("Missing one or more latest artifacts.")

latest_model = model_files[-1]
latest_ohe = ohe_files[-1]
latest_features = features_files[-1]

shutil.copy2(latest_model, ROOT / "model.pkl")
shutil.copy2(latest_ohe, ROOT / "ohe.pkl")
shutil.copy2(latest_features, ROOT / "features.pkl")

print("Promoted latest artifacts:")
print(latest_model)
print(latest_ohe)
print(latest_features)
