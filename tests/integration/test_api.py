"""
Tests d'intégration pour l'API House Price Prediction
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ajouter le src au path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from {{cookiecutter.package_name}}.api.main import app

client = TestClient(app)


class TestHousePriceAPI:
    """Tests d'intégration pour l'API"""
    
    def test_root_endpoint(self):
        """Test de la page d'accueil"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "House Price Prediction" in response.text
    
    def test_health_endpoint(self):
        """Test du health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "dataset_info" in data
        assert data["dataset_info"]["rows"] == {{cookiecutter.dataset_rows}}
        assert data["dataset_info"]["features"] == {{cookiecutter.dataset_features}}
    
    def test_metrics_endpoint(self):
        """Test des métriques"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions_total" in data
        assert "uptime_seconds" in data
        assert "dataset_info" in data
        assert data["dataset_info"]["houses"] == {{cookiecutter.dataset_rows}}
    
    def test_predict_endpoint_valid_data(self):
        """Test de prédiction avec données valides"""
        house_data = {
            "features": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft_living": 2000,
                "sqft_lot": 8000,
                "floors": 2,
                "sqft_above": 1500,
                "sqft_basement": 500,
                "yr_built": 1990,
                "yr_renovated": 2010,
                "waterfront": 0,
                "view": 2,
                "condition": 3,
                "street": "123 Main St",
                "city": "Seattle",
                "statezip": "WA 98101",
                "country": "USA"
            },
            "return_confidence": False,
            "return_explanation": False
        }
        
        response = client.post("/predict", json=house_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "timestamp" in data
        assert "prediction_time_seconds" in data
        assert "request_id" in data
        assert "dataset_info" in data
        
        # Vérifier que la prédiction est raisonnable
        assert data["prediction"] > 50000  # Prix minimum
        assert data["prediction"] < 5000000  # Prix maximum raisonnable
    
    def test_predict_endpoint_invalid_data(self):
        """Test avec données invalides"""
        invalid_data = {
            "features": {
                "bedrooms": -1,  # Invalide
                "bathrooms": 2.5,
                "sqft_living": 2000,
                "sqft_lot": 8000,
                "floors": 2,
                "sqft_above": 1500,
                "sqft_basement": 500,
                "yr_built": 1990,
                "yr_renovated": 2010,
                "waterfront": 0,
                "view": 2,
                "condition": 3,
                "street": "123 Main St",
                "city": "Seattle",
                "statezip": "WA 98101",
                "country": "USA"
            }
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self):
        """Test avec champs manquants"""
        incomplete_data = {
            "features": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                # Champs manquants...
            }
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error
    
    def test_logging_structured(self):
        """Test que le logging structuré fonctionne"""
        house_data = {
            "features": {
                "bedrooms": 4,
                "bathrooms": 3,
                "sqft_living": 2500,
                "sqft_lot": 10000,
                "floors": 2,
                "sqft_above": 2000,
                "sqft_basement": 500,
                "yr_built": 2000,
                "yr_renovated": 0,
                "waterfront": 1,
                "view": 4,
                "condition": 4,
                "street": "456 Oak Ave",
                "city": "Bellevue",
                "statezip": "WA 98004",
                "country": "USA"
            }
        }
        
        response = client.post("/predict", json=house_data)
        assert response.status_code == 200
        
        # Le logging est testé en vérifiant qu'aucune exception n'est levée
        # et que la réponse contient un request_id pour le tracking
        data = response.json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0
