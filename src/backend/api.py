import os
import json
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import mlflow
from mlflow import MlflowClient
from mlflow import sklearn as mlflow_sklearn

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from scipy.special import softmax
import torch

# =========================
# Configuración base / MLflow
# =========================

load_dotenv(override=True)

# Tracking (Databricks o local, como en tu training)
def _resolve_tracking_uri() -> str:
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")

    if env_uri:
        return env_uri
    if profile:
        return f"databricks://{profile}"
    if host and token:
        return "databricks"

    # Fallback local
    local_store = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(local_store, exist_ok=True)
    return f"file://{local_store}"


mlflow.set_tracking_uri(_resolve_tracking_uri())
mlflow.set_registry_uri("databricks-uc")

DEEP_MODEL_NAME = "workspace.default.EMI_imedia_sentiment_deep_model_NTBK"
DEEP_MODEL_ALIAS = "champion"

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BERT_MODEL_MAP = {
    "bert_nlptown": "nlptown/bert-base-multilingual-uncased-sentiment",
    "bert_distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
}


# =========================
# Carga perezosa de modelos
# =========================

@lru_cache(maxsize=1)
def get_mlp_model_and_embedder():
    """
    Carga el modelo MLP + scaler desde MLflow Model Registry (alias champion)
    y el SentenceTransformer para generar embeddings.
    """
    model_uri = f"models:/{DEEP_MODEL_NAME}@{DEEP_MODEL_ALIAS}"
    print(f"[MLP] Cargando modelo desde {model_uri}")
    clf = mlflow_sklearn.load_model(model_uri=model_uri)

    print(f"[MLP] Cargando SentenceTransformer: {ST_MODEL_NAME}")
    st_model = SentenceTransformer(ST_MODEL_NAME)

    return clf, st_model


@lru_cache(maxsize=4)
def get_bert_model(model_key: str):
    """
    Carga tokenizer + modelo BERT según la key.
    """
    if model_key not in BERT_MODEL_MAP:
        raise ValueError(f"Modelo BERT desconocido: {model_key}")

    hf_name = BERT_MODEL_MAP[model_key]
    print(f"[BERT] Cargando modelo: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)
    return tokenizer, model


# =========================
# Lógica de predicción
# =========================

def _predict_with_bert(model_key: str, text: str) -> dict:
    tokenizer, model = get_bert_model(model_key)
    hf_name = BERT_MODEL_MAP[model_key]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits.numpy()[0]

    probs = softmax(logits)
    idx = int(np.argmax(probs))
    score = float(probs[idx])

    # Map a etiqueta textual
    if "nlptown" in hf_name:
        # 5 clases: 1-5 estrellas
        if idx <= 1:
            label = "negative"
        elif idx == 2:
            label = "neutral"
        else:
            label = "positive"
    else:
        # Binario: label_0 = NEG, label_1 = POS
        label = "positive" if idx == 1 else "negative"

    return {
        "label": label,
        "score": score,
        "model_key": model_key,
        "backend_model_name": hf_name,
    }


def _embed_texts(st_model: SentenceTransformer, texts, batch_size: int = 32) -> np.ndarray:
    texts = list(texts)
    emb = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return emb


def _predict_with_mlp(text: str) -> dict:
    clf, st_model = get_mlp_model_and_embedder()

    X_emb = _embed_texts(st_model, [text])  # (1, dim)
    # Asumimos clf es Pipeline(scaler + MLPClassifier) con predict_proba
    probas = clf.predict_proba(X_emb)[0]
    score_pos = float(probas[1])
    label = "positive" if score_pos >= 0.5 else "negative"

    return {
        "label": label,
        "score": score_pos,
        "model_key": "mlp_transformer",
        "backend_model_name": f"{DEEP_MODEL_NAME}@{DEEP_MODEL_ALIAS}",
    }


def predict_sentiment(text: str, model_key: str) -> dict:
    """
    Router de modelos: elige entre BERT o MLP+embeddings.
    """
    if model_key == "mlp_transformer":
        return _predict_with_mlp(text)
    elif model_key in BERT_MODEL_MAP:
        return _predict_with_bert(model_key, text)
    else:
        # fallback: usar MLP si el model_key no es reconocido
        result = _predict_with_mlp(text)
        result["warning"] = f"model_key desconocido '{model_key}', se usó mlp_transformer."
        return result


# =========================
# API FastAPI
# =========================

app = FastAPI(
    title="IMEDIA Sentiment API",
    description="API de inferencia para análisis de sentimientos en comentarios de Reddit.",
    version="1.0.0",
)


class SentimentRequest(BaseModel):
    text: str
    model_key: Literal["mlp_transformer", "bert_nlptown", "bert_distilbert"] = "mlp_transformer"


class SentimentResponse(BaseModel):
    label: str
    score: float
    model_key: str
    backend_model_name: str
    warning: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/predict-sentiment", response_model=SentimentResponse)
def predict_sentiment_endpoint(payload: SentimentRequest):
    """
    Endpoint principal de predicción de sentimiento.
    """
    result = predict_sentiment(text=payload.text, model_key=payload.model_key)
    return result
