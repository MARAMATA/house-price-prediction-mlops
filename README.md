# {{cookiecutter.project_name}}

{{cookiecutter.description}}

## 📊 Dataset

- **{{cookiecutter.dataset_rows}} maisons** avec **{{cookiecutter.dataset_features}} features**
- **Source**: [Kaggle House Data](https://www.kaggle.com/datasets/shree1992/housedata)
- **Target**: {{cookiecutter.target_variable}} (prix des maisons)

### Features disponibles:
- **Temporelles**: date
- **Structurelles**: bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
- **Qualité**: waterfront, view, condition
- **Historiques**: yr_built, yr_renovated  
- **Géographiques**: street, city, statezip, country

## 🏗️ Architecture MLOps

- ✅ **API REST** FastAPI avec documentation
- ✅ **Docker** containerisation (python:{{cookiecutter.python_version}}-slim)
- ✅ **CI/CD** GitHub Actions
- ✅ **Logging structuré** (timestamp, features, prédiction, durée)
- ✅ **Monitoring** et audit
- ✅ **Tests** automatisés

## 🚀 Démarrage rapide

```bash
# Configuration automatique
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Ajouter les données
cp data.csv data/raw/
cp output.csv data/raw/

# Exploration
jupyter notebook notebooks/01-data-exploration.ipynb

# Entraînement
python scripts/train.py

# API
make run
```

## 📋 Pipeline ML

1. **Préparation des données**
   - Nettoyage (valeurs manquantes, outliers)
   - Ingénierie des features (variables dérivées)
   
2. **Sélection de modèles**
   - Exploration de différents algorithmes
   - Validation croisée
   
3. **Déploiement**
   - API REST pour prédictions
   - Monitoring en temps réel

## 🐳 Docker

```bash
# Build
docker build -t {{cookiecutter.project_slug}}:latest .

# Run
docker run -p 8000:8000 {{cookiecutter.project_slug}}:latest
```

## 📊 API Endpoints

- `GET /` - Interface web
- `GET /health` - Health check
- `POST /predict` - Prédiction prix maison
- `GET /metrics` - Métriques Prometheus

## 👤 Auteur

**{{cookiecutter.author_name}}** - [{{cookiecutter.author_email}}](mailto:{{cookiecutter.author_email}})

## �� License

{{cookiecutter.license}}
