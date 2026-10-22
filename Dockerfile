FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY class_quality ./class_quality
COPY pytest.ini .

ENV CLASS_QUALITY_HOST=0.0.0.0
ENV PORT=7860
ENV OLLAMA_CLOUD_PERMITTED=0

EXPOSE 7860

CMD ["python", "-m", "class_quality.app"]
