FROM python:3.8.0-slim-buster

LABEL version="0.1.0"

WORKDIR /app

COPY . .

# Install dependencies and set permissions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    chown -R 777 /app
    
WORKDIR /app/src_review

CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]
