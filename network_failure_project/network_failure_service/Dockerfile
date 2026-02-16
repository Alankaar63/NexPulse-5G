FROM python:3.11-slim

WORKDIR /app

COPY network_failure_service/requirements.txt /app/network_failure_service/requirements.txt
RUN pip install --no-cache-dir -r /app/network_failure_service/requirements.txt

COPY model.pkl /app/model.pkl
COPY ohe.pkl /app/ohe.pkl
COPY features.pkl /app/features.pkl
COPY network_failure_service /app/network_failure_service

EXPOSE 8000

CMD ["uvicorn", "network_failure_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
