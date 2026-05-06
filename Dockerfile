FROM python:3.11-slim

WORKDIR /app

# System deps for sounddevice (PortAudio) — needed if talk_tee runs inside container
# For API-only usage these are not required but included for completeness
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY backend.py        .
COPY datag_bridge.py   .
COPY chat_tee.py       .
COPY learn.py          .
COPY ingest_doc.py     .
COPY reindex_substrate.py .
COPY talk_tee.py       .
COPY talk_voxi.py      .
COPY src/              ./src/
COPY models/           ./models/
COPY data_store/       ./data_store/

# Sentence-transformers caches models to ~/.cache by default — keep it in /app
ENV SENTENCE_TRANSFORMERS_HOME=/app/.st_cache
ENV HF_HOME=/app/.hf_cache

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
