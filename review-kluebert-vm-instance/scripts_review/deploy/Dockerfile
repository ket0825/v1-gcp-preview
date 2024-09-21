FROM python:3.8.0-slim-buster

LABEL version="0.1.0"

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

EXPOSE 8080

ENTRYPOINT [ "python", "./src_review/model_server.py" ]
