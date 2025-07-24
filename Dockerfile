# Dockerfile pour {{cookiecutter.project_name}}
# Base: python:{{cookiecutter.python_version}}-slim (selon exigences)
FROM python:{{cookiecutter.python_version}}-slim

LABEL maintainer="{{cookiecutter.author_email}}" \
      description="{{cookiecutter.description}}" \
      version="{{cookiecutter.version}}"

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

# Copie du code source
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY setup.py .

# Installation du package
RUN pip install -e .

# Création des répertoires nécessaires
RUN mkdir -p logs data/raw data/processed

# Port pour l'API REST
EXPOSE 8000

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut: API REST
CMD ["uvicorn", "src.{{cookiecutter.package_name}}.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
