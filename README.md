# IMEDIA Project

> **Estado**: MVP funcional (Reddit) · Arquitectura escalable a Facebook/Threads/X · Medallion (`raw → bronze → silver → gold`)

---

## 1) Introducción
En un entorno donde la información cambia a velocidad absurda, **IMEDIA Project** nace para ayudarte a **explorar, analizar y entender** lo que pasa en redes sociales y comunidades online. Empezamos por **Reddit** (API abierta, estable), pero la arquitectura ya está pensada para sumar **Facebook, Threads, X (Twitter)** u otras fuentes.

IMEDIA implementa un pipeline de datos reproducible (con **uv**, **Polars**, **SQLite**) que extrae publicaciones/comentarios, los procesa por capas (**medallion**), y deja la puerta abierta a **NLP** y **dashboards**.

---

## 2) Problema y justificación
- Las personas y equipos necesitan **monitorizar tendencias**, **temas** y **sentimientos** sin perderse en scroll infinito.
- Las APIs y formatos cambian; necesitas una **arquitectura modular**, **idempotente** y **trazable**.
- Herramientas low-code/BI ayudan a visualizar, pero el valor nace en **datos limpios** y **modelados**.

**IMEDIA** justifica su existencia al:  
(1) estandarizar ingestion y almacenamiento,  
(2) dejar un **track** auditable por capas,  
(3) facilitar análisis avanzados (NLP, LLMs, dashboards) sobre bases sólidas.

---

## 3) Objetivos
**General**: Automatizar la **recolección y procesamiento** de contenido social para habilitar análisis y visualizaciones confiables.

**Específicos**:
- Ingerir contenido de Reddit vía API oficial (PRAW).
- Normalizar y persistir en formato analítico (**Parquet**) y en una base **SQLite** para consultas rápidas.
- Preparar **dimensiones** y **hechos** (SILVER) que soporten KPIs y análisis posteriores.
- Diseñar una CLI reproducible con **uv** para orquestar corridas.
- Roadmap: sentiment/topic modeling, features para LLMs, dashboards (Power BI/DuckDB, etc.).

---

## 4) Alcance (MVP) y estado actual
- **Fuente**: Reddit ✅  
- **Pipeline**: `raw → bronze → silver` ✅ (gold en diseño)  
- **Persistencia**: Parquet + SQLite ✅  
- **CLI**: modo *subreddit único* y modo *descubrimiento de N subreddits “hot”* ✅  
- **Comentarios**: descarga del **primer post** por subreddit (flag `--fetch-comments`) ✅  
- **GOLD**: KPIs/ML/LLMs (pendiente) 🔜  

---

## 5) Arquitectura del pipeline (Medallion)
```
Reddit API → [RAW] NDJSON (as-is) → [BRONZE] Parquet (tipado/flatten) → [SILVER] Parquet+SQLite (dims/facts) → [GOLD] KPIs/Features/ML/LLMs
```

**Capas**:
- **RAW**: dumps sin transformar (NDJSON por origen/lote). No se borra ni se pisa.
- **BRONZE**: tipado suave, separar campos, **sin perder columnas** (nulos permitidos). Particionado por fecha (`created_utc`).
- **SILVER**: normalización (dimensiones y tablas de hechos), claves coherentes, **upsert** en SQLite.
- **GOLD**: métricas, agregados, features ML y vistas para BI (en construcción).

**Modelado (SILVER)**:
- `dim_subreddit(subreddit, subscribers, description, created_utc, over18)`
- `dim_author(author_name)`
- `fact_posts(post_id, subreddit, author, title, selftext, url, score, num_comments, over_18, created_utc, …)`
- `fact_comments(comment_id, post_id, author, body, created_utc, score, …)`

---

## 6) Estructura del repositorio
```
IMEDIA_PROJECT_V2/
├─ pyproject.toml
├─ README.md
├─ uv.lock
├─ LICENSE
├─ .gitignore
├─ .env
├─ .env.example
├─ .venv/                          
├─ imedia.egg-info/
├─ src/
│  └─ imedia/                          
├─ notebooks/
│  ├─ 00_informe_final.ipynb
│  ├─ 01_eda_inicial.ipynb
│  ├─ 02_data_wrangling.ipynb
│  └─ 03-training_model.ipynb
├─ db/
│  └─ imedia.sqlite                    
├─ data/
│  ├─ raw/
│  ├─ bronze/
│  ├─ silver/
│  └─ processed/                       
└─ preprocesador/                      
```

---

# Flujo (resumen)
```
Reddit API
  → data/raw/reddit
  → data/bronze/reddit
  → data/silver/reddit + SQLite
  → data/processed/*.csv
  → preprocesador/<modelo>_*
```

---

# Notas
- La carpeta **preprocesador/** guarda *pipelines* de preprocesamiento para modelos entrenados.
- **processed/** contiene datasets finales para entrenamiento y validación.

---

### Código fuente (src/)
- `config.py`  
- `reddit_client.py`  
- `raw_extractor.py`  
- `bronze_transformer.py`  
- `silver_normalizer.py`  
- `gold_products.py`  
- `repo_sqlite.py`  
- `utils.py`  
- `__main__.py`  

---

## 7) Requisitos
- **Python ≥ 3.11**
- **uv** — https://docs.astral.sh/uv/
- API Reddit funcional

---

## 8) Instalación
```bash
git clone <URL-del-repo>
cd imedia
uv sync
cp .env.example .env
```

### Variables de entorno
```env
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=imedia/0.1 by <user>
IMEDIA_DB_PATH=db/imedia.sqlite
IMEDIA_DATA_ROOT=data
REDDIT_REQUEST_TIMEOUT=30
```

---

## 9) Uso (CLI)
Ejecutar:

```bash
uv run python -m imedia --subreddit python --limit 50 --fetch-comments
```

Modo “discover-hot”:

```bash
uv run python -m imedia --discover-hot 10 --hot-strategy all_top_day --limit 30
```

---

## 10) Salidas esperadas
- RAW → NDJSON  
- BRONZE → Parquet  
- SILVER → dims + facts + SQLite  

---

## 11) Verificación rápida
Consultas a SQLite + revisión de archivos Parquet.

---

## 12) Solución de problemas comunes
Errores de `.env`, OAuth, timeouts, etc.

---

## 13) Roadmap
- full comments  
- capa GOLD  
- LLM Q&A  
- Integración X/Threads/Facebook  
- DuckDB/ADBC  
- tests y ruff  

---

# ⭐ **14) Reproducción completa del proyecto (entorno, pipeline, inferencia)**  
*(Sección agregada para cumplir con requisitos académicos)*  

Esta sección explica cómo **correr IMEDIA desde cero**, **ejecutar el pipeline**, y **realizar inferencias** con los modelos de sentimiento.

---

## **14.1 Crear el entorno**
```bash
git clone <repo>
cd IMEDIA_PROJECT_V2

# crea el entorno y sincroniza dependencias
uv sync

# activa uv (no requiere venv manual)
uv run python --version
```

---

## **14.2 Configurar credenciales**
En `.env`:

```env
REDDIT_CLIENT_ID=xxxx
REDDIT_CLIENT_SECRET=xxxx
REDDIT_USER_AGENT=imedia/0.1 by <usuario>
IMEDIA_DATA_ROOT=data
IMEDIA_DB_PATH=db/imedia.sqlite
```

Validar autenticación:

```bash
uv run python - <<'PY'
from imedia.reddit_client import RedditClient
r = RedditClient().reddit
s = r.subreddit("python")
print("OK Reddit:", s.subscribers)
PY
```

---

## **14.3 Ejecutar todo el pipeline RAW → BRONZE → SILVER**
### Ingesta de ejemplo:
```bash
uv run python -m imedia \
    --subreddit python \
    --limit 50 \
    --time-filter day \
    --fetch-comments
```

### Discover-hot:
```bash
uv run python -m imedia \
    --discover-hot 10 \
    --hot-strategy all_top_day \
    --limit 40
```

Esto poblará:
```
data/raw/
data/bronze/
data/silver/
db/imedia.sqlite
```

---

## **14.4 Generar datasets limpios para NLP (processed/)**
Abre el notebook:

```
notebooks/02_data_wrangling.ipynb
```

Ejecuta todas las celdas y se generarán:

```
data/processed/sentiment_train.parquet
data/processed/sentiment_val.parquet
data/processed/sentiment_test.parquet
```

---

## **14.5 Entrenar modelos y registrar en MLflow**
### Ejecutar el flow Prefect completo
```bash
# (opcional)
prefect server start

# flow de entrenamiento
uv run src/pipelines/train_pipeline.py
```

El pipeline ejecuta:

1. BERT zero-shot eval  
2. SentenceTransformer embeddings  
3. HPO (Hyperopt) para MLP  
4. Entrenamiento final  
5. Registro en Databricks UC (alias `champion`)  
6. Tabla comparativa impresa al final  

---

## **14.6 Levantar el backend de inferencia (FastAPI)**
Desde `src/backend`:

```bash
uv run uvicorn api:app --reload --port 8000
```

Probar:

```bash
curl http://127.0.0.1:8000/health
```

Inferencia:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict-sentiment \
    -H "Content-Type: application/json" \
    -d '{"text":"I love this project!", "model_key":"mlp_transformer"}'
```

---

## **14.7 Ejecutar la UI (Streamlit)**

Desde `src/frontend`:

```bash
uv run streamlit run main.py
```

Interfaz disponible en:

```
http://localhost:8501
```

La UI permite:
- Escribir texto
- Elegir modelo (MLP o BERT)
- Enviar al backend
- Ver etiqueta, score y JSON completo

---

## **14.8 Contenerización y despliegue**
### Backend
```bash
docker build -t imedia-api -f src/backend/Dockerfile .
docker run -p 8000:8000 imedia-api
```

### Frontend
```bash
docker build -t imedia-ui -f src/frontend/Dockerfile .
docker run -p 8501:8501 imedia-ui
```

### Compose
(`docker-compose.yaml`)
```bash
docker compose up --build
```

---

## **14.9 Inferencia desde cualquier cliente**
Ejemplo en Python:

```python
import requests

payload = {
    "text": "This tool is extremely helpful!",
    "model_key": "mlp_transformer"
}

res = requests.post(
    "http://localhost:8000/api/v1/predict-sentiment",
    json=payload
)
print(res.json())
```

---

# ✔️ FIN DEL README (COMPLETO Y ACTUALIZADO)
