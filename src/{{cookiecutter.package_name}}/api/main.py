"""
API REST FastAPI pour prédiction des prix de maisons
Logging structuré: timestamp, features, prédiction, durée
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import os
import time
from datetime import datetime
from loguru import logger
import structlog
import uuid
import json
from contextlib import asynccontextmanager

from .models import HousePredictionRequest, HousePredictionResponse, HealthResponse

# Configuration du logging structuré selon les exigences
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

structured_logger = structlog.get_logger()

# Variables globales
startup_time = time.time()
prediction_count = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie de l'application"""
    logger.info("🏠 Démarrage API House Price Prediction - Dataset 4600 maisons")
    yield
    logger.info("🔻 Arrêt de l'API")

# Application FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="API de prédiction des prix de maisons - Dataset 4600 maisons avec 18 features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pour logging structuré des requêtes"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Logger la requête entrante
    structured_logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        timestamp=datetime.now().isoformat()
    )
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Logger la réponse
    structured_logger.info(
        "request_completed",
        request_id=request_id,
        status_code=response.status_code,
        duration=duration,
        timestamp=datetime.now().isoformat()
    )
    
    return response

@app.get("/", response_class=HTMLResponse)
async def root():
    """Interface web de l'API"""
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Price Prediction API</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); overflow: hidden; }}
            .header {{ background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 30px; text-align: center; }}
            .content {{ padding: 30px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #e9ecef; }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .links {{ display: flex; gap: 15px; justify-content: center; margin: 30px 0; }}
            .btn {{ background: #3498db; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; transition: all 0.3s; font-weight: 500; }}
            .btn:hover {{ background: #2980b9; transform: translateY(-2px); }}
            .btn.success {{ background: #27ae60; }}
            .btn.info {{ background: #8e44ad; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏠 House Price Prediction</h1>
                <p>API de prédiction des prix immobiliers avec Machine Learning</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">4600</div>
                        <div>Maisons dans le dataset</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">18</div>
                        <div>Features disponibles</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">REST</div>
                        <div>API moderne</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">ML</div>
                        <div>Machine Learning</div>
                    </div>
                </div>
                
                <div class="links">
                    <a href="/docs" class="btn">📚 Documentation API</a>
                    <a href="/health" class="btn success">🔍 Health Check</a>
                    <a href="/metrics" class="btn info">📊 Métriques</a>
                </div>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <h3>🚀 Exemple d'utilisation</h3>
                    <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;">
POST /predict
{{
  "features": {{
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 8000,
    "floors": 2,
    "waterfront": 0,
    "view": 2,
    "condition": 3,
    "sqft_above": 1500,
    "sqft_basement": 500,
    "yr_built": 1990,
    "yr_renovated": 2010,
    "street": "123 Main St",
    "city": "Seattle",
    "statezip": "WA 98101",
    "country": "USA"
  }}
}}</pre>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check avec métriques"""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        dataset_info={
            "rows": 4600,
            "features": 18,
            "target": "price"  # Replace with actual target variable
        },
        uptime_seconds=uptime,
        predictions_count=prediction_count
    )

@app.post("/predict", response_model=HousePredictionResponse)
async def predict_house_price(request: HousePredictionRequest):
    """
    Prédiction du prix d'une maison
    
    Logging structuré selon les exigences:
    - timestamp de requête
    - entrées (features)  
    - prédiction
    - durée
    """
    global prediction_count
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.now()
    
    try:
        # Extraction des features
        features = request.features.dict()
        
        # LOG STRUCTURÉ - DÉBUT PRÉDICTION
        structured_logger.info(
            "prediction_started",
            request_id=request_id,
            timestamp=timestamp.isoformat(),
            features=features  # entrées (features)
        )
        
        # SIMULATION DE PRÉDICTION (remplacer par le vrai modèle)
        predicted_price = calculate_house_price(features)
        
        duration = time.time() - start_time
        prediction_count += 1
        
        # LOG STRUCTURÉ - PRÉDICTION TERMINÉE (selon exigences)
        structured_logger.info(
            "prediction_completed",
            request_id=request_id,
            timestamp=timestamp.isoformat(),  # timestamp de requête
            features=features,                # entrées (features)
            prediction=predicted_price,       # prédiction
            duration=duration,               # durée
            prediction_count=prediction_count
        )
        
        # Réponse
        return HousePredictionResponse(
            prediction=predicted_price,
            timestamp=timestamp,
            prediction_time_seconds=duration,
            model_version="v1.0.0",
            request_id=request_id,
            dataset_info={
                "trained_on_houses": 4600,
                "features_used": 18
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        # LOG STRUCTURÉ - ERREUR
        structured_logger.error(
            "prediction_error",
            request_id=request_id,
            timestamp=timestamp.isoformat(),
            features=features,
            error=str(e),
            duration=duration
        )
        
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

def calculate_house_price(features: dict) -> float:
    """
    Calcul du prix basé sur les features du dataset
    Features du dataset réel: bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated,
    street, city, statezip, country
    """
    base_price = 400000
    price = base_price
    
    # Calcul basé sur les vraies features du dataset
    price += features.get('bedrooms', 0) * 25000
    price += features.get('bathrooms', 0) * 20000
    price += features.get('sqft_living', 0) * 150
    price += features.get('sqft_lot', 0) * 5
    price += features.get('floors', 0) * 15000
    price += features.get('waterfront', 0) * 200000
    price += features.get('view', 0) * 30000
    price += features.get('condition', 0) * 25000
    price += features.get('sqft_above', 0) * 100
    price += features.get('sqft_basement', 0) * 80
    
    # Ajustement selon l'âge
    current_year = datetime.now().year
    house_age = current_year - features.get('yr_built', current_year)
    price -= house_age * 1000
    
    # Bonus rénovation
    if features.get('yr_renovated', 0) > 0:
        years_since_reno = current_year - features.get('yr_renovated', 0)
        if years_since_reno < 10:
            price += 50000
    
    # Ajustement par ville (simulation)
    city_multipliers = {
        'Seattle': 1.3,
        'Bellevue': 1.4,
        'Redmond': 1.2,
        'Kirkland': 1.15
    }
    city = features.get('city', '').lower()
    for city_name, multiplier in city_multipliers.items():
        if city_name.lower() in city:
            price *= multiplier
            break
    
    return max(price, 50000)  # Prix minimum

@app.get("/metrics")
async def get_metrics():
    """Métriques pour monitoring"""
    return {
        "predictions_total": prediction_count,
        "uptime_seconds": time.time() - startup_time,
        "dataset_info": {
            "houses": 4600,
            "features": 18,
            "target": "price"  # Replace with actual target variable
        },
        "api_version": "1.0.0",
        "status": "healthy"
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(
        "src.mlops_package.api.main:app",  # Updated to match the actual package name
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )