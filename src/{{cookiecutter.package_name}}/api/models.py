"""
Modèles Pydantic pour l'API House Price Prediction
Basé sur le dataset réel: 4600 maisons, 18 features
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime


class HouseFeatures(BaseModel):
    """
    Modèle pour les caractéristiques d'une maison
    Basé sur les 18 colonnes du dataset réel
    """
    
    # Features numériques principales
    bedrooms: float = Field(..., ge=0, le=20, description="Nombre de chambres")
    bathrooms: float = Field(..., ge=0, le=15, description="Nombre de salles de bain")
    sqft_living: int = Field(..., ge=0, le=50000, description="Surface habitable (sqft)")
    sqft_lot: int = Field(..., ge=0, le=2000000, description="Surface du terrain (sqft)")
    floors: float = Field(..., ge=1, le=5, description="Nombre d'étages")
    sqft_above: int = Field(..., ge=0, le=50000, description="Surface au-dessus du sol (sqft)")
    sqft_basement: int = Field(..., ge=0, le=10000, description="Surface du sous-sol (sqft)")
    yr_built: int = Field(..., ge=1800, le=2025, description="Année de construction")
    yr_renovated: int = Field(..., ge=0, le=2025, description="Année de rénovation (0 si jamais)")
    
    # Features catégorielles
    waterfront: int = Field(..., ge=0, le=1, description="Vue sur l'eau (0=Non, 1=Oui)")
    view: int = Field(..., ge=0, le=4, description="Qualité de la vue (0-4)")
    condition: int = Field(..., ge=1, le=5, description="État de la maison (1-5)")
    
    # Features géographiques (du dataset réel)
    street: str = Field(..., min_length=1, max_length=200, description="Adresse rue")
    city: str = Field(..., min_length=1, max_length=100, description="Ville")
    statezip: str = Field(..., min_length=5, max_length=20, description="État et code postal")
    country: str = Field(default="USA", description="Pays")
    
    @validator('sqft_above')
    def validate_sqft_above(cls, v, values):
        """Validation: sqft_above <= sqft_living"""
        if 'sqft_living' in values and v > values['sqft_living']:
            raise ValueError('sqft_above ne peut pas être supérieur à sqft_living')
        return v
    
    @validator('yr_renovated')
    def validate_renovation(cls, v, values):
        """Validation: année rénovation >= année construction"""
        if v > 0 and 'yr_built' in values and v < values['yr_built']:
            raise ValueError('Année de rénovation ne peut pas être antérieure à la construction')
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
                "street": "123 Main Street",
                "city": "Seattle",
                "statezip": "WA 98101",
                "country": "USA"
            }
        }


class HousePredictionRequest(BaseModel):
    """Requête de prédiction"""
    
    features: HouseFeatures
    return_confidence: bool = Field(default=False, description="Retourner intervalle de confiance")
    return_explanation: bool = Field(default=False, description="Retourner explication")
    
    class Config:
        schema_extra = {
            "example": {
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
                    "street": "123 Main Street",
                    "city": "Seattle",
                    "statezip": "WA 98101",
                    "country": "USA"
                },
                "return_confidence": False,
                "return_explanation": False
            }
        }


class HousePredictionResponse(BaseModel):
    """Réponse de prédiction"""
    
    prediction: float = Field(..., description="Prix prédit en dollars")
    timestamp: datetime = Field(..., description="Timestamp de la prédiction")
    prediction_time_seconds: float = Field(..., description="Durée du calcul")
    model_version: str = Field(default="v1.0.0", description="Version du modèle")
    request_id: str = Field(..., description="ID unique de la requête")
    dataset_info: Dict[str, Any] = Field(..., description="Infos sur le dataset d'entraînement")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 425000.50,
                "timestamp": "2025-01-15T10:30:00.000Z",
                "prediction_time_seconds": 0.0156,
                "model_version": "v1.0.0",
                "request_id": "req-123e4567-e89b-12d3-a456-426614174000",
                "dataset_info": {
                    "trained_on_houses": 4600,
                    "features_used": 18
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Statut de l'API")
    timestamp: datetime = Field(..., description="Timestamp du check")
    version: str = Field(..., description="Version de l'API")
    dataset_info: Dict[str, Any] = Field(..., description="Informations dataset")
    uptime_seconds: float = Field(..., description="Uptime en secondes")
    predictions_count: int = Field(..., description="Nombre total de prédictions")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-15T10:30:00.000Z",
                "version": "1.0.0",
                "dataset_info": {
                    "rows": 4600,
                    "features": 18,
                    "target": "price"
                },
                "uptime_seconds": 3600.5,
                "predictions_count": 42
            }
        }


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction en batch"""
    
    houses: list[HouseFeatures] = Field(..., min_items=1, max_items=100, 
                                       description="Liste des maisons (max 100)")
    
    class Config:
        schema_extra = {
            "example": {
                "houses": [
                    {
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
                        "street": "123 Main Street",
                        "city": "Seattle",
                        "statezip": "WA 98101",
                        "country": "USA"
                    }
                ]
            }
        }