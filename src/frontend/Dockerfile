FROM python:3.11-slim

# Evita que python bufee logs y que pip deje cache
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# Paquetes de sistema mínimos
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalación de dependencias
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /code/requirements.txt

# Copiar todo el frontend (o todo el repo si es pequeño)
COPY . /app

EXPOSE 8501

# Ejecutar Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
