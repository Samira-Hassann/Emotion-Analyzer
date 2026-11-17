import html
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import emoji
import torch
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Hugging Face model id (or a local folder path). Override with the MODEL_ID env var.
MODEL_ID = os.getenv("MODEL_ID", "Samira-Hassan/modernbert-goemotions-6class")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BASE_DIR = Path(__file__).parent


def load_label_classes():
    """Fallback: read label_classes.json if config.json only has LABEL_0..LABEL_n."""
    try:
        local = Path(MODEL_ID) / "label_classes.json"
        path = local if local.exists() else Path(hf_hub_download(MODEL_ID, "label_classes.json"))
        with open(path, encoding="utf-8") as f:
            classes = json.load(f)
    except Exception:
        return None
    if isinstance(classes, dict):
        first = str(next(iter(classes)))
        if first.isdigit():
            return {int(k): v for k, v in classes.items()}
        return {int(v): k for k, v in classes.items()}
    return {i: c for i, c in enumerate(classes)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    if all(str(v).startswith("LABEL_") for v in id2label.values()):
        id2label = load_label_classes() or id2label

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.id2label = id2label
    yield
    del model, tokenizer


app = FastAPI(
    title="GoEmotions ModernBERT API",
    description="Six-class emotion classification with a fine-tuned ModernBERT-base.",
    version="1.0.0",
    lifespan=lifespan,
)


def clean_text(text: str) -> str:
    """Same preprocessing used during training."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"[‘’´`′]", "'", text)
    text = re.sub(r"\b(\w+)\s+'\s+t\b", r"\1n't", text)
    text = re.sub(r"\b(\w+)\s+'\s+re\b", r"\1're", text)
    text = re.sub(r"\b(\w+)\s+'\s+ve\b", r"\1've", text)
    text = re.sub(r"\b(\w+)\s+'\s+ll\b", r"\1'll", text)
    text = re.sub(r"\b(\w+)\s+'\s+d\b", r"\1'd", text)
    text = re.sub(r"\b(\w+)\s+'\s+m\b", r"\1'm", text)
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " emailtoken ", text)
    text = re.sub(r"(?<!\w)@\w+", " usertoken ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EmotionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        json_schema_extra={"example": "By far the coolest thing I've seen on this thread yet!"},
    )


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


def run_inference(request: Request, texts: list[str]) -> list[dict]:
    tokenizer = request.app.state.tokenizer
    model = request.app.state.model
    id2label = request.app.state.id2label

    cleaned = [clean_text(t) for t in texts]
    if any(not c for c in cleaned):
        raise HTTPException(status_code=400, detail="One or more texts are empty after cleaning.")

    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1).cpu()

    results = []
    for original, clean, p in zip(texts, cleaned, probs):
        pred_id = int(torch.argmax(p))
        results.append({
            "original_text": original,
            "cleaned_text": clean,
            "predicted_emotion": id2label[pred_id],
            "confidence": round(float(p[pred_id]), 4),
            "probabilities": {id2label[i]: round(float(p[i]), 4) for i in range(len(p))},
        })
    return results


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health_check(request: Request):
    return {
        "status": "running",
        "model": MODEL_ID,
        "device": str(DEVICE),
        "labels": list(request.app.state.id2label.values()),
    }


@app.post("/predict")
def predict_emotion(body: EmotionRequest, request: Request):
    return run_inference(request, [body.text])[0]


@app.post("/predict/batch")
def predict_batch(body: BatchRequest, request: Request):
    return {"results": run_inference(request, body.texts)}
