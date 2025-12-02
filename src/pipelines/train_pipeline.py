# ============================================
# Sentiment Flow (Prefect + BERT + MLP + HPO + MLflow)
# ============================================

import os
import json
import warnings
from datetime import datetime
from typing import Tuple, Dict, Any, List

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow import sklearn as mlflow_sklearn

from dotenv import load_dotenv

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from scipy.special import softmax
import torch

from prefect import flow, task

# HPO (Hyperopt)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# =========================
# Constantes / Config
# =========================

SEED = 42
np.random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
TRAIN_PATH = DATA_DIR / "sentiment_train.parquet"
VAL_PATH   = DATA_DIR / "sentiment_val.parquet"
TEST_PATH  = DATA_DIR / "sentiment_test.parquet"

EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_LOCAL_DIR = ROOT_DIR / "preprocesador"
PREPROC_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "/Users/marianasgg19@gmail.com/EMI/imedia/Sentiment_BERT_MLP"
DEEP_MODEL_NAME = "workspace.default.EMI_imedia_sentiment_deep_model_NTBK"

BERT_MODELS = [
    "nlptown/bert-base-multilingual-uncased-sentiment",      # 5 clases (1-5 stars)
    "distilbert-base-uncased-finetuned-sst-2-english",       # 2 clases (neg/pos)
]

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Espacio de búsqueda HPO para el MLP
HLS_CHOICES      = [(128,), (256,), (512,)]
BS_CHOICES       = [64, 128, 256]
MAXITER_CHOICES  = [5, 10, 20]
MAX_EVALS        = 20  # nº de trials de Hyperopt


# =========================
# Utils generales
# =========================

def _resolve_tracking_uri() -> str:
    load_dotenv(override=True)
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

    local_store = Path.cwd() / "mlruns"
    local_store.mkdir(parents=True, exist_ok=True)
    return f"file://{local_store}"


def save_embeddings_artifact(model_name: str,
                             embeddings_matrix: np.ndarray,
                             split: str,
                             extra_metadata: Dict[str, Any] | None = None) -> Dict[str, str]:
    """
    Guarda embeddings + metadata en ./embeddings con sufijo NTBK.
    """
    import datetime as dt

    safe_model_name = model_name.replace("/", "_")
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    base_name = f"{safe_model_name}_{split}_{timestamp}_NTBK"

    emb_path = EMBEDDINGS_DIR / f"{base_name}_embeddings.npy"
    meta_path = EMBEDDINGS_DIR / f"{base_name}_metadata.json"

    np.save(emb_path, embeddings_matrix)

    metadata = {
        "model_name": model_name,
        "split": split,
        "embedding_shape": embeddings_matrix.shape,
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    pd.Series(metadata).to_json(meta_path)

    return {
        "embeddings_path": str(emb_path),
        "metadata_path": str(meta_path),
    }


def predict_binary_from_logits(logits: np.ndarray, model_name: str) -> int:
    """
    Convierte logits de cada modelo a etiqueta binaria 0/1.
    Implementación simple basada en convenciones típicas de estos checkpoints.
    """
    probs = softmax(logits)[0]
    if "nlptown" in model_name:
        # 5 clases -> 1,2 = negativo; 3 = neutro; 4,5 = positivo
        idx = np.argmax(probs)
        return 1 if idx >= 3 else 0
    else:
        # Modelos binarios: índice 1 = positivo
        idx = np.argmax(probs)
        return 1 if idx == 1 else 0


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    texts = list(texts)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb


# =========================
# Tareas Prefect
# =========================

@task(name="Sentiment-Setup-MLflow")
def setup_mlflow_task() -> None:
    """
    Carga variables de entorno y configura MLflow (tracking y experimento).
    """
    tracking_uri = _resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"Experimento creado: {EXPERIMENT_NAME} (id={exp_id})")
    else:
        print(f"Experimento encontrado: {EXPERIMENT_NAME} (id={exp.experiment_id})")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("Tracking URI:", mlflow.get_tracking_uri())


@task(name="Sentiment-Load-Data")
def load_data_task(sample_frac: float = 1.0, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga train/val/test preprocesados desde data/processed.
    Opcional: hace muestreo por fracción para entrenar más rápido.
    """
    assert TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists(), "Faltan splits en data/processed"

    rng = np.random.RandomState(random_state)

    def _load_and_sample(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if sample_frac < 1.0:
            n = int(len(df) * sample_frac)
            idx = rng.choice(len(df), size=n, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        return df

    train_df = _load_and_sample(TRAIN_PATH)
    val_df   = _load_and_sample(VAL_PATH)
    test_df  = _load_and_sample(TEST_PATH)

    print("Shapes after sampling:")
    print("  train:", train_df.shape)
    print("  val  :", val_df.shape)
    print("  test :", test_df.shape)

    return train_df, val_df, test_df


@task(name="Sentiment-Preprocess-Text")
def preprocess_text_task(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae clean_text y labels de cada split.
    """
    X_train_text = train_df["clean_text"].astype(str).tolist()
    X_val_text   = val_df["clean_text"].astype(str).tolist()
    X_test_text  = test_df["clean_text"].astype(str).tolist()

    y_train = train_df["sentiment"].astype(int).values
    y_val   = val_df["sentiment"].astype(int).values
    y_test  = test_df["sentiment"].astype(int).values

    return X_train_text, X_val_text, X_test_text, y_train, y_val, y_test


@task(name="Sentiment-Train-BERT-Models")
def train_bert_models_task(
    X_val_text: List[str],
    X_test_text: List[str],
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Evalúa los dos modelos BERT (inferencia) y loggea métricas en MLflow.
    Devuelve lista de dicts con resultados para tabla de comparación.
    """
    results: List[Dict[str, Any]] = []

    for model_name in BERT_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        def predict_batch(texts):
            preds = []
            for t in texts:
                inputs = tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256,
                )
                with torch.no_grad():
                    logits = model(**inputs).logits.numpy()
                preds.append(predict_binary_from_logits(logits, model_name))
            return np.array(preds, dtype=int)

        run_name = f"BERT_{model_name.split('/')[-1]}"
        with mlflow.start_run(run_name=run_name):
            y_val_pred = predict_batch(X_val_text)
            y_test_pred = predict_batch(X_test_text)

            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1  = f1_score(y_val, y_val_pred, average="binary")
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1  = f1_score(y_test, y_test_pred, average="binary")

            mlflow.log_param("model_family", "bert_sequence_classifier")
            mlflow.log_param("hf_model_name", model_name)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_f1",       val_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_f1",       test_f1)

            print(f"\n=== {model_name} ===")
            print("Validation accuracy:", val_acc)
            print("Validation F1      :", val_f1)
            print("Test accuracy      :", test_acc)
            print("Test F1            :", test_f1)

            # Guardar tokenizer como "preprocesador" con sufijo NTBK
            safe_name = model_name.replace("/", "_")
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            tok_dir = PREPROC_LOCAL_DIR / f"{safe_name}_tokenizer_NTBK_{ts}"
            tok_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(tok_dir)

            # Loggear tokenizer como artifact en MLflow
            mlflow.log_artifacts(str(tok_dir), artifact_path="preprocessor_tokenizer")

            results.append({
                "model_key": f"bert_{model_name.split('/')[-1]}",
                "run_id": mlflow.active_run().info.run_id,
                "val_accuracy": float(val_acc),
                "val_f1": float(val_f1),
                "test_accuracy": float(test_acc),
                "test_f1": float(test_f1),
            })

    return results


@task(name="Sentiment-Train-MLP-Embeddings-HPO")
def train_mlp_embeddings_hpo_task(
    X_train_text: List[str],
    X_val_text: List[str],
    X_test_text: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Genera embeddings con SentenceTransformer + ejecuta HPO (Hyperopt)
    sobre un MLPClassifier. Registra el mejor modelo en UC y devuelve
    resultados para tabla de comparación.
    """
    # 1) Embeddings con SentenceTransformer (una sola vez)
    st_model = SentenceTransformer(ST_MODEL_NAME)

    X_train_emb = embed_texts(st_model, X_train_text)
    X_val_emb   = embed_texts(st_model, X_val_text)
    X_test_emb  = embed_texts(st_model, X_test_text)

    print("Emb shapes:", X_train_emb.shape, X_val_emb.shape, X_test_emb.shape)

    # Guardar embeddings como artefactos reproducibles
    emb_info_train = save_embeddings_artifact(ST_MODEL_NAME, X_train_emb, split="train")
    emb_info_val   = save_embeddings_artifact(ST_MODEL_NAME, X_val_emb,   split="val")
    emb_info_test  = save_embeddings_artifact(ST_MODEL_NAME, X_test_emb,  split="test")

    # 2) Definir espacio de búsqueda HPO
    search_space = {
        "hidden_layer_sizes": hp.choice("hidden_layer_sizes", HLS_CHOICES),
        "alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e-1)),
        "learning_rate_init": hp.loguniform("learning_rate_init", np.log(1e-4), np.log(5e-3)),
        "batch_size": hp.choice("batch_size", BS_CHOICES),
        "max_iter": hp.choice("max_iter", MAXITER_CHOICES),
    }

    # 3) Parent run para agrupar todos los trials
    with mlflow.start_run(run_name="mlp_transformer_embeddings_HPO_FLOW_NTBK") as parent_run:

        def objective(params):
            """
            Función objetivo para Hyperopt.
            Entrena pipeline scaler + MLP con los hiperparámetros propuestos
            y devuelve -val_f1 (porque Hyperopt MINIMIZA).
            """
            mlp_clf = MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                activation="relu",
                solver="adam",
                batch_size=int(params["batch_size"]),
                learning_rate_init=float(params["learning_rate_init"]),
                alpha=float(params["alpha"]),
                max_iter=int(params["max_iter"]),
                random_state=SEED,
                verbose=False,
            )

            pipe = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("mlp", mlp_clf),
            ])

            with mlflow.start_run(run_name="mlp_hpo_trial", nested=True):
                pipe.fit(X_train_emb, y_train)

                y_val_pred = pipe.predict(X_val_emb)
                val_acc = accuracy_score(y_val, y_val_pred)
                val_f1  = f1_score(y_val, y_val_pred, average="binary")

                mlflow.log_param("model_family", "mlp_classifier")
                mlflow.log_param("features", "sentence_transformers_embeddings")
                mlflow.log_params({
                    "hidden_layer_sizes": params["hidden_layer_sizes"],
                    "alpha": params["alpha"],
                    "learning_rate_init": params["learning_rate_init"],
                    "batch_size": params["batch_size"],
                    "max_iter": params["max_iter"],
                })
                mlflow.log_metric("val_accuracy", val_acc)
                mlflow.log_metric("val_f1", val_f1)

                return {
                    "loss": -val_f1,   # queremos maximizar F1 → minimizamos -F1
                    "status": STATUS_OK,
                }

        # 4) Ejecutar Hyperopt
        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
            # sin rstate explícito para evitar problemas de versión
        )

        # Mapear índices de hp.choice a valores reales
        idx_hls      = best["hidden_layer_sizes"]
        idx_batch    = best["batch_size"]
        idx_max_iter = best["max_iter"]

        best_hls      = HLS_CHOICES[idx_hls]
        best_batch    = BS_CHOICES[idx_batch]
        best_max_iter = MAXITER_CHOICES[idx_max_iter]

        best_alpha = best["alpha"]
        best_lr    = best["learning_rate_init"]

        # Log de mejores hiperparámetros en el parent run
        mlflow.log_param("hpo_max_evals", MAX_EVALS)
        mlflow.log_params({
            "best_hidden_layer_sizes": best_hls,
            "best_alpha": best_alpha,
            "best_learning_rate_init": best_lr,
            "best_batch_size": best_batch,
            "best_max_iter": best_max_iter,
        })

        # 5) Entrenamiento final con hiperparámetros óptimos
        best_mlp = MLPClassifier(
            hidden_layer_sizes=best_hls,
            activation="relu",
            solver="adam",
            batch_size=int(best_batch),
            learning_rate_init=float(best_lr),
            alpha=float(best_alpha),
            max_iter=int(best_max_iter),
            random_state=SEED,
            verbose=True,
        )

        best_pipe = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("mlp", best_mlp),
        ])

        best_pipe.fit(X_train_emb, y_train)

        # Evaluación final en val y test
        y_val_pred  = best_pipe.predict(X_val_emb)
        y_test_pred = best_pipe.predict(X_test_emb)

        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1  = f1_score(y_val, y_val_pred, average="binary")
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1  = f1_score(y_test, y_test_pred, average="binary")

        print("\n[FINAL MLP + Transformer después de HPO]")
        print("Validation accuracy:", val_acc)
        print("Validation F1      :", val_f1)
        print("Test accuracy      :", test_acc)
        print("Test F1            :", test_f1)

        mlflow.log_metric("final_val_accuracy",  val_acc)
        mlflow.log_metric("final_val_f1",        val_f1)
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_f1",       test_f1)

        # Log de embeddings como artifacts del parent run
        mlflow.log_artifact(emb_info_train["embeddings_path"], artifact_path="embeddings/train")
        mlflow.log_artifact(emb_info_val["embeddings_path"],   artifact_path="embeddings/val")
        mlflow.log_artifact(emb_info_test["embeddings_path"],  artifact_path="embeddings/test")

        # Firma + registro en UC
        example_input = X_train_emb[:100]
        example_output = best_pipe.predict(example_input)
        signature = infer_signature(example_input, example_output)

        mlflow.set_registry_uri("databricks-uc")
        mlflow_sklearn.log_model(
            sk_model=best_pipe,
            artifact_path="model",
            registered_model_name=DEEP_MODEL_NAME,
            signature=signature,
            input_example=example_input[:5],
        )

        # Guardar preprocesador (scaler) localmente + artifact
        fitted_scaler = best_pipe.named_steps["scaler"]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        local_preproc_dir = PREPROC_LOCAL_DIR / f"mlp_scaler_NTBK_{ts}"
        mlflow_sklearn.save_model(sk_model=fitted_scaler, path=str(local_preproc_dir))

        mlflow_sklearn.log_model(
            sk_model=fitted_scaler,
            artifact_path="preprocessor_scaler_NTBK",
        )

        return {
            "model_key": "mlp_transformer",
            "run_id": parent_run.info.run_id,
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
        }


@task(name="Sentiment-Compare-Register-Notify")
def compare_register_notify_task(
    bert_results: List[Dict[str, Any]],
    mlp_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Une resultados, selecciona mejor modelo por val_f1, asigna alias 'champion'
    al MLP en UC si es el mejor, e imprime resumen (notificación).
    """
    results = bert_results + [mlp_result]
    comp_df = pd.DataFrame(results).sort_values("val_f1", ascending=False).reset_index(drop=True)

    print("\n=== Comparación de modelos por val_f1 (desc) ===")
    print(comp_df)

    best_row = comp_df.iloc[0]
    best_model_key = best_row["model_key"]
    print(f"\n🏆 Mejor modelo según val_f1: {best_model_key} (val_f1={best_row['val_f1']:.4f})")

    champion_version = None
    if best_model_key == "mlp_transformer":
        mlflow.set_registry_uri("databricks-uc")
        client = MlflowClient()

        versions = client.search_model_versions(f"name = '{DEEP_MODEL_NAME}'")
        if versions:
            champion_version = max(int(v.version) for v in versions)
            client.set_registered_model_alias(
                name=DEEP_MODEL_NAME,
                alias="champion",
                version=champion_version,
            )
            print(f"\n✅ Alias 'champion' asignado a {DEEP_MODEL_NAME} (v{champion_version})")
        else:
            print(f"\n⚠ No se encontraron versiones para el modelo {DEEP_MODEL_NAME}")
    else:
        print("\n⚠ El mejor modelo es un BERT (no registrado en UC en este flow).")
        print("   El MLP se registró en el Model Registry, pero SIN alias 'champion'.")

    summary = {
        "best_model_key": best_model_key,
        "best_val_f1": float(best_row["val_f1"]),
        "best_test_f1": float(best_row["test_f1"]),
        "champion_version": champion_version,
        "comparison_table": comp_df.to_dict(orient="records"),
    }
    print("\nResumen final del flow:")
    print(json.dumps(summary, indent=2))
    return summary


# =========================
# FLOW PRINCIPAL
# =========================

@flow(name="Sentiment-BERT-MLP-HPO-Training-Flow")
def sentiment_training_flow(sample_frac: float = 1.0) -> Dict[str, Any]:
    """
    Flow de entrenamiento:
      1) Setup MLflow
      2) Carga / muestreo de datos
      3) Preprocesamiento (texto + labels)
      4) Evaluación de BERTs (inferencia)
      5) HPO + entrenamiento final de MLP + embeddings
      6) Comparación, registro (alias) y notificación
    """
    setup_mlflow_task()

    train_df, val_df, test_df = load_data_task(sample_frac=sample_frac)

    X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = preprocess_text_task(
        train_df, val_df, test_df
    )

    bert_results = train_bert_models_task(
        X_val_text=X_val_text,
        X_test_text=X_test_text,
        y_val=y_val,
        y_test=y_test,
    )

    mlp_result = train_mlp_embeddings_hpo_task(
        X_train_text=X_train_text,
        X_val_text=X_val_text,
        X_test_text=X_test_text,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    summary = compare_register_notify_task(
        bert_results=bert_results,
        mlp_result=mlp_result,
    )

    return summary


if __name__ == "__main__":
    result = sentiment_training_flow(sample_frac=1.0)
    print("\nResultado devuelto por el flow:")
    print(json.dumps(result, indent=2))

'''
Para ejecutar en local:
prefect server start
uv run src/pipelines/train_pipeline.py

para abrir la ui:
uv run uvicorn src.backend.api:app --reload --port 8000
uv run streamlit run src/frontend/main.py
'''