FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
