#!/bin/bash

set -e

echo "🏠 Configuration de l'environnement {{cookiecutter.project_name}}"
echo "Dataset: {{cookiecutter.dataset_rows}} maisons avec {{cookiecutter.dataset_features}} features"
echo "================================================================="

# Vérification Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 requis"
    exit 1
fi

# Environnement virtuel
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Environnement virtuel créé"
else
    echo "⚠️  Environnement virtuel existe déjà"
fi

# Activation
source venv/bin/activate
echo "✅ Environnement virtuel activé"

# Mise à jour pip
pip install --upgrade pip

# Installation des dépendances
pip install -r requirements.txt
echo "✅ Dépendances installées"

# Installation du package
pip install -e .
echo "✅ Package {{cookiecutter.package_name}} installé"

# Création des répertoires
mkdir -p logs models data/processed data/external monitoring figures
touch data/processed/.gitkeep data/external/.gitkeep logs/.gitkeep models/.gitkeep monitoring/.gitkeep
echo "✅ Structure de répertoires créée"

# Configuration environnement
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Fichier .env créé"
fi

# Git
if command -v git &> /dev/null && [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "Initial commit - {{cookiecutter.project_name}}" || true
    echo "✅ Repository Git initialisé"
fi

# Test d'import
if python -c "import {{cookiecutter.package_name}}" 2>/dev/null; then
    echo "✅ Package importé avec succès"
else
    echo "⚠️  Import du package échoué (normal sans données)"
fi

echo ""
echo "🎉 Configuration terminée avec succès!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. source venv/bin/activate"
echo "2. Copiez vos fichiers CSV dans data/raw/:"
echo "   - cp data.csv data/raw/"
echo "   - cp output.csv data/raw/"
echo "3. jupyter notebook notebooks/01-data-exploration.ipynb"
echo "4. python scripts/train.py"
echo "5. make run"
echo "6. Testez l'API: curl http://localhost:8000/health"
echo ""
echo "🆘 Aide: make help"
echo "📚 Documentation: README.md"
echo "🌐 API Docs: http://localhost:8000/docs (après make run)"
