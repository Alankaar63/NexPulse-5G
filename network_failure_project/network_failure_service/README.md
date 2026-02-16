# Network Failure Predictor Service

Production-style network failure platform with:
- automated telemetry capture
- prediction + maintenance recommendation
- LAN health scanning
- ticket automation
- feedback + retraining pipeline

## API endpoints
- `GET /` redirect to website
- `GET /app` website UI
- `GET /docs` Swagger docs
- `GET /health`
- `GET /ping`
- `GET /download-test`
- `POST /telemetry`
- `POST /predict`
- `POST /lan-health`
- `POST /feedback`
- `GET /tickets`

## Run locally
From `network_failure_project`:

```bash
./network_failure_service/start_local.sh
```

Open:
- Website: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Environment config

```bash
cp network_failure_service/.env.example network_failure_service/.env
```

Ticketing options in `.env`:
- `TICKETING_PROVIDER=none` (default)
- `TICKETING_PROVIDER=webhook` + `TICKET_WEBHOOK_URL`
- `TICKETING_PROVIDER=zammad` + `ZAMMAD_URL`, `ZAMMAD_TOKEN`, `ZAMMAD_GROUP`

## Retraining

```bash
./.nfp-venv/bin/python network_failure_service/scripts/retrain_model.py
./.nfp-venv/bin/python network_failure_service/scripts/retrain_model.py --promote
```

## Docker

```bash
docker compose -f docker-compose.network-failure.yml up --build
```

## Notes
- Browser clients cannot access all low-level Wi-Fi metrics.
- `POST /lan-health` requires host tools (`nmap`, `ping`) installed.
