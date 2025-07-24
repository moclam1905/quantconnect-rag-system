FROM python:3.11-slim
WORKDIR /app

# requirements ở root dự án → copy từ build‑context
COPY ../../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY data/bm25_index.pkl.gz  /app/data/bm25_index.pkl.gz
COPY data/bm25_meta.json     /app/data/bm25_meta.json

# copy mã hybrid_api
COPY ../../src/data_processing/hybrid_api.py /app/src/data_processing/hybrid_api.py

CMD ["uvicorn", "src.data_processing.hybrid_api:app", "--host", "0.0.0.0", "--port", "8000"]
