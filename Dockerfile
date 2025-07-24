# Dockerfile pour votre projet MLOps
FROM python:3.11-slim

LABEL maintainer="maramatad@gmail.com" \
      description="MLOps House Price Prediction Project" \
      version="1.0.0"

WORKDIR /app

# Variables d'environnement pour Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie de TOUS les dossiers du projet (IMPORTANT!)
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY setup.py .

# Installation du package
RUN pip install -e .

# Création des répertoires nécessaires
RUN mkdir -p logs models data/raw data/processed

# Port pour l'API REST
EXPOSE 8000

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut: serveur HTTP
CMD ["python", "-m", "http.server", "8000"]
