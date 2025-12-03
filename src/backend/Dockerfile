# Imagen base ligera
FROM python:3.11-slim

# Variables para evitar problemas en tiempo de ejecución
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/huggingface \
    TRANSFORMERS_CACHE=/huggingface

# Crear los directorios necesarios para HuggingFace cache
RUN mkdir -p /huggingface && chmod -R 777 /huggingface

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias básicas del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements del backend
COPY ./requirements.txt /app/requirements.txt

# Instalamos dependencias del backend (FastAPI, Transformers, MLflow, etc.)
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copiar código del backend
COPY . /app

# Exponer puerto de FastAPI
EXPOSE 8000

# CMD para correr el servidor
# uvicorn main:app — ajusta si tu archivo se llama distinto
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
