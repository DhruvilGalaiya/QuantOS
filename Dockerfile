FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

COPY requirements-mlops.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-mlops.txt

COPY src/quantos_hmm_v10.py src/quantos_hmm_v10.py
COPY mlops/ mlops/
COPY data/regime/features/nifty_50.parquet data/regime/features/nifty_50.parquet

EXPOSE 8000

CMD ["uvicorn", "mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
