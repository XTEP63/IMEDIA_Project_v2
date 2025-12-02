import os
import json
import requests
import streamlit as st

# =========================
# Configuración básica de la app
# =========================
st.set_page_config(
    page_title="Reddit Sentiment · IMEDIA",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Análisis de Sentimientos en Comentarios de Reddit")
st.write(
    "Front-end para consumir el servicio de inferencia FastAPI "
    "entrenado en el MLOps del proyecto IMEDIA."
)

# =========================
# Configuración del backend
# =========================
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{BACKEND_URL}/api/v1/predict-sentiment"

st.sidebar.header("⚙️ Configuración")

st.sidebar.markdown(f"**Backend URL:** `{BACKEND_URL}`")

# Selección de modelo (debe alinear con las keys que uses en FastAPI)
MODEL_OPTIONS = {
    "MLP + SentenceTransformer (all-MiniLM-L6-v2)": "mlp_transformer",
    "BERT nlptown (multilingual-uncased-sentiment)": "bert_nlptown",
    "DistilBERT (sst-2, inglés)": "bert_distilbert",
}

model_label = st.sidebar.selectbox(
    "Modelo de sentimiento",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
)
model_key = MODEL_OPTIONS[model_label]

threshold = st.sidebar.slider(
    "Umbral de confianza para destacar la predicción",
    min_value=0.5,
    max_value=0.99,
    value=0.7,
    step=0.01,
)

st.sidebar.info(
    "El umbral solo se usa para resaltar visualmente la confianza; "
    "la predicción siempre se mostrará."
)

# =========================
# Entrada de usuario
# =========================
st.subheader("✍️ Escribe o pega un comentario de Reddit")

default_text = (
    "I really enjoyed this post, the discussion was super insightful and helpful!"
)

user_text = st.text_area(
    "Comentario",
    value=default_text,
    height=200,
    help="Este texto se enviará al backend FastAPI para obtener el sentimiento.",
)

analyze_btn = st.button("🔍 Analizar sentimiento")

# =========================
# Llamada a la API
# =========================
if analyze_btn:
    if not user_text.strip():
        st.warning("Por favor escribe un comentario antes de analizar.")
    else:
        payload = {
            "text": user_text,
            "model_key": model_key,
        }

        with st.spinner("Llamando al servicio de predicción..."):
            try:
                response = requests.post(
                    PREDICT_ENDPOINT,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Error al comunicarse con el backend: {e}")
            else:
                if response.status_code != 200:
                    st.error(
                        f"Respuesta no exitosa del backend "
                        f"({response.status_code}): {response.text}"
                    )
                else:
                    result = response.json()
                    label = result.get("label", "unknown")
                    score = float(result.get("score", 0.0))
                    backend_model_key = result.get("model_key", model_key)

                    # =========================
                    # Mostrar resultado
                    # =========================
                    st.subheader("✅ Resultado de la predicción")

                    # Estilo simple según polaridad
                    label_lower = label.lower()
                    if "neg" in label_lower:
                        sentiment_emoji = "😡"
                    elif "pos" in label_lower:
                        sentiment_emoji = "😄"
                    elif "neu" in label_lower:
                        sentiment_emoji = "😐"
                    else:
                        sentiment_emoji = "🤔"

                    st.markdown(
                        f"### {sentiment_emoji} Sentimiento predicho: "
                        f"**`{label}`**"
                    )

                    st.metric(
                        label="Confianza del modelo",
                        value=f"{score:.3f}",
                    )

                    if score >= threshold:
                        st.success(
                            f"La confianza ({score:.3f}) está por encima del umbral "
                            f"configurado ({threshold:.2f})."
                        )
                    else:
                        st.warning(
                            f"La confianza ({score:.3f}) es inferior al umbral "
                            f"configurado ({threshold:.2f})."
                        )

                    st.caption(
                        f"Modelo usado (backend): `{backend_model_key}` "
                        f"· Seleccionado en UI: `{model_key}`"
                    )

                    # Mostrar JSON crudo (útil para debug)
                    with st.expander("Ver respuesta completa de la API (debug)"):
                        st.json(result)
