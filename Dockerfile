# CPU-only image. Models are downloaded and cached at build time; the semantic cache is pre-warmed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY pytest.ini ./
COPY data data
COPY app app
COPY scripts scripts
COPY bench bench
COPY tests tests

# 1) download + cache the embedding model and cross-encoder inside the image
RUN python -c "from app import models; models.warm(); print('models cached')"
# 2) pre-warm the semantic cache from queries.json + gold samples (no LLM is reachable during a build,
#    so these plans come from the deterministic grounded extractor; new queries use the SLM at runtime)
RUN SGTE_LLM_PROVIDER=none python scripts/prewarm.py --fresh

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
