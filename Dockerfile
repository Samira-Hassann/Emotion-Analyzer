FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image smaller. Use a CUDA base image for GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app.py index.html ./

# The model is downloaded from the Hugging Face Hub on first start.
ENV MODEL_ID=Samira-Hassan/modernbert-goemotions-6class
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
