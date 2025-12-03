FROM python:3.11-slim

# Logs sin buffer y sin cache de pip
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copiar requirements ligeros de la UI
COPY requirements.txt /app/requirements.txt

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copiar el código de la app
COPY . /app

# Puerto de Streamlit
EXPOSE 8501

# Ejecutar Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
