FROM python:3.14-slim

WORKDIR /api

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/lgbm_model.pkl .
COPY api/main.py .

ENV PORT=8080 

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]