# Serving image for the FastAPI PHA-classification API (src/otda/api.py).
#
# Expects models/champion/ to already exist locally before `docker build`
# — run scripts/export_champion_model.py first. The model is baked into
# the image at build time (not loaded from a live MLflow connection at
# runtime), which is also the artifact shape Stage 9's SageMaker
# /invocations adapter will expect.
FROM python:3.13-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY models/champion/ models/champion/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "otda.api:app", "--host", "0.0.0.0", "--port", "8000"]
