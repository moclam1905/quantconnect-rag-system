FROM python:3.11-slim
WORKDIR /app

COPY ../../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy toàn bộ phase4
COPY ../../src/data_processing/phase4 /app/src/data_processing/phase4

CMD ["uvicorn", "src.data_processing.phase4.answer_api:app", "--host", "0.0.0.0", "--port", "8001"]
