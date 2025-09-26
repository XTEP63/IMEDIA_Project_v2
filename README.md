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

**IMEDIA** justifica su existencia al: (1) estandarizar ingestion y almacenamiento, (2) dejar un **track** auditable por capas, (3) facilitar análisis avanzados (NLP, LLMs, dashboards) sobre bases sólidas.

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
Reddit API → [RAW] NDJSON (as-is) → [BRONZE] Parquet (tipado/flatten) → [SILVER] Parquet+SQLite (dims/facts) → [GOLD] KPIs/Features/LLMs
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
imedia/
├─ pyproject.toml
├─ .env.example
├─ README.md
├─ db/
│  └─ imedia.sqlite
└─ data/
   ├─ raw/reddit/
   │  ├─ posts/part-<batch>-[subreddit].ndjson
   │  ├─ comments/part-<batch>-[post_id].ndjson
   │  ├─ subreddits/part-<batch>-[subreddit].ndjson
   │  └─ hot_sublists/part-<batch>-<estrategia>.ndjson
   ├─ bronze/reddit/
   │  ├─ posts/YYY=…/MM=…/DD=…/posts__<batch>__<subreddit>.parquet
   │  ├─ comments/YYY=…/MM=…/DD=…/comments__<batch>__<post_id>.parquet
   │  ├─ subreddits/subreddits-<batch>-<subreddit>.parquet
   │  └─ hot_sublists/hot_sublists-<batch>-<uid>.parquet
   └─ silver/reddit/
      ├─ dim_subreddit.parquet
      ├─ dim_author.parquet
      ├─ fact_posts.parquet
      └─ fact_comments.parquet
```

### Código fuente (src/)
- `config.py` — rutas, batch id, env vars
- `reddit_client.py` — autenticación PRAW (read-only)
- `raw_extractor.py` — descarga **as-is** a RAW
- `bronze_transformer.py` — tipado/flatten → BRONZE
- `silver_normalizer.py` — normalización + upsert a SQLite
- `gold_products.py` — placeholder para KPIs/ML
- `repo_sqlite.py` — DDL + upserts
- `utils.py` — helpers (slugify, casts robustos)
- `__main__.py` — CLI orquestador

---

## 7) Requisitos
- **Python** ≥ 3.11
- **uv** (gestión de entornos ultra-rápida) → https://docs.astral.sh/uv/
- Conexión a internet (para API Reddit)

---

## 8) Instalación
```bash
# 1) clona el repo
git clone <URL-del-repo>
cd imedia

# 2) instala dependencias
uv sync

# 3) copia variables de entorno
type .env.example > .env   # (Windows: cp .env.example .env)
```

### Variables de entorno (`.env`)
```env
REDDIT_CLIENT_ID=tu_client_id
REDDIT_CLIENT_SECRET=tu_client_secret
REDDIT_USER_AGENT=imedia/0.1 by <tu_usuario>
IMEDIA_DB_PATH=db/imedia.sqlite
IMEDIA_DATA_ROOT=data
REDDIT_REQUEST_TIMEOUT=30
# opcional para etiquetar corridas manualmente
# IMEDIA_BATCH_TS=20250101_1200
```

### Test de autenticación
```bash
uv run python -c "from imedia.reddit_client import RedditClient; r=RedditClient().reddit; s=r.subreddit('python'); print('OK Reddit! subs:', getattr(s,'subscribers',None))"
```

---

## 9) Uso (CLI)
La CLI vive en `__main__.py`. Ejecuta con `uv run python -m imedia [opciones]`.

### Modos (exclusivos)
1. **Subreddit único**
```bash
uv run python -m imedia \
  --subreddit python \
  --limit 50 \
  --time-filter day \
  --fetch-comments
```
2. **Descubrir N subreddits “hot”** (y descargar posts de cada uno)
```bash
uv run python -m imedia \
  --discover-hot 10 \
  --hot-strategy all_top_day \
  --limit 30 \
  --include-nsfw    # opcional
```

### Parámetros
| Parámetro | Tipo | Obligatorio | Default | Descripción |
|---|---:|:---:|---:|---|
| `--subreddit <nombre>` | str | **Mutuamente excluyente** con `--discover-hot` | — | Modo 1: ingestión de un subreddit específico. |
| `--discover-hot <N>` | int | **Mutuamente excluyente** con `--subreddit` | — | Modo 2: descubre N subreddits “calientes” y descarga posts de cada uno. |
| `--hot-strategy {popular,all_hot,all_top_day}` | str | No (solo aplica con `--discover-hot`) | `popular` | Cómo descubrir subreddits: `popular` (rápido), `all_hot` (zeitgeist), `all_top_day` (mejores del día). |
| `--include-nsfw` | flag | No | `false` | Incluir subreddits NSFW en el descubrimiento. |
| `--limit <N>` | int | No | `100` | Posts a descargar **por subreddit**. |
| `--time-filter {hour,day,week,month,year,all}` | str | No | `day` | Ventana temporal para `top`. |
| `--fetch-comments` | flag | No | `false` | Descarga comentarios del **primer post** en cada subreddit del lote. |

> **Nota**: `--fetch-comments` actualmente trae **solo** el primer post de cada subreddit. Un flag `--all-comments` puede añadirse en el roadmap.

### Ejemplos útiles
- Top 20 `machinelearning` última semana con comentarios del primer post:
```bash
uv run python -m imedia --subreddit machinelearning --limit 20 --time-filter week --fetch-comments
```
- Descubrir 15 subreddits por popularidad e ingerir 40 posts por cada uno:
```bash
uv run python -m imedia --discover-hot 15 --hot-strategy popular --limit 40
```

---

## 10) Salidas esperadas
- **RAW**: NDJSON por origen (no se pisa). Ej: `data/raw/reddit/posts/part-<batch>-python.ndjson`.
- **BRONZE**: Parquet particionado por `YYYY/MM/DD` (posts/comments) + archivos únicos por sub/post.
- **SILVER**: `dim_*.parquet`, `fact_*.parquet` y **SQLite** poblado (`db/imedia.sqlite`).

---

## 11) Verificación rápida (post-run)
```bash
# conteos en SQLite
uv run python - <<'PY'
import sqlite3
con = sqlite3.connect('db/imedia.sqlite')
for t in ('subreddits','authors','posts','comments'):
    try:
        n = con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
        print(t, n)
    except Exception as e:
        print(t, 'no existe:', e)
PY
```
```bash
# inspeccionar SILVER
uv run python - <<'PY'
import polars as pl
p = pl.read_parquet('data/silver/reddit/fact_posts.parquet')
print('subs distintos:', p.select('subreddit').n_unique())
print('total posts:', p.height)
print(p.select('subreddit').unique().head(15))
PY
```

---

## 12) Solución de problemas comunes
- **`ValueError: Faltan variables en .env`** → Completa `REDDIT_CLIENT_ID/SECRET/USER_AGENT`.
- **`OAuthException`** (PRAW) → Verifica que tu app de Reddit sea de tipo **script** y que el secret sea correcto.
- **Timeouts** → Aumenta `REDDIT_REQUEST_TIMEOUT` (ej. 60) o reduce `--limit` y la cantidad de subreddits.

---

## 13) Roadmap (sujeto a cambios)
- `--all-comments` (comentarios de todos los posts del lote)
- Capa **GOLD**: KPIs (7d/24h), engagement por hora, features para modelos
- Integración **LLMs**: Q&A sobre corpus, resúmenes temáticos
- Más fuentes: X/Threads/Facebook (cuando políticas y APIs lo permitan)
- Export a **DuckDB/ADBC** y/o formatos **Delta/Iceberg** para datasets grandes
- Tests `pytest` + `ruff` 

---

### Créditos
- **PRAW**, **Polars**, **uv** y comunidad OSS ❤️

