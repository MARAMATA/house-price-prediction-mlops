#!/usr/bin/env python3
"""
Script d'entraînement pour le dataset House Prices
Dataset: 21613 maisons avec 18 features
Colonnes: date, price, bedrooms, bathrooms, sqft_living, sqft_lot, floors,
         waterfront, view, condition, sqft_above, sqft_basement, yr_built,
         yr_renovated, street, city, statezip, country
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import argparse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost non installé. pip install xgboost pour de meilleures performances")

# Essayer d'importer LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM non installé. pip install lightgbm pour de meilleures performances")


def setup_logging():
    """Configuration du logging"""
    logger.add(
        "logs/training_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )


def load_real_dataset(file_path: str) -> pd.DataFrame:
    """
    Charge le vrai dataset si disponible
    
    Args:
        file_path: Chemin vers data.csv ou output.csv
        
    Returns:
        DataFrame avec les données
    """
    if Path(file_path).exists():
        logger.info(f"Chargement du dataset réel: {file_path}")
        data = pd.read_csv(file_path)
        
        # Vérification des colonnes attendues
        expected_cols = [
            'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
            'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'street', 'city', 'statezip', 'country'
        ]
        
        missing_cols = set(expected_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Colonnes manquantes dans le dataset: {missing_cols}")
        
        logger.info(f"Dataset chargé: {data.shape[0]} maisons, {data.shape[1]} colonnes")
        
        # Afficher les statistiques de prix
        logger.info(f"Prix moyen: ${data['price'].mean():,.0f}")
        logger.info(f"Prix médian: ${data['price'].median():,.0f}")
        logger.info(f"Prix min: ${data['price'].min():,.0f}")
        logger.info(f"Prix max: ${data['price'].max():,.0f}")
        
        return data
    else:
        raise FileNotFoundError(f"Dataset non trouvé: {file_path}")


def generate_sample_dataset() -> pd.DataFrame:
    """
    Génère un dataset d'exemple avec la même structure que le vrai
    """
    logger.info("Génération d'un dataset d'exemple (21613 maisons)")
    
    np.random.seed(42)
    n_samples = 21613
    
    # Dates sur 2 ans
    start_date = pd.Timestamp('2022-01-01')
    end_date = pd.Timestamp('2023-12-31')
    dates = pd.date_range(start_date, end_date, periods=n_samples)
    
    # Villes réalistes de la région Seattle
    cities = ['Seattle', 'Bellevue', 'Redmond', 'Kirkland', 'Bothell', 'Issaquah', 'Renton', 'Kent']
    states = ['WA 98101', 'WA 98004', 'WA 98052', 'WA 98033', 'WA 98027', 'WA 98059', 'WA 98055', 'WA 98031']
    
    data = pd.DataFrame({
        'date': np.random.choice(dates, n_samples),
        'bedrooms': np.random.randint(1, 8, n_samples),
        'bathrooms': np.round(np.random.uniform(1, 5, n_samples) * 2) / 2,  # 1.0, 1.5, 2.0, etc.
        'sqft_living': np.random.randint(500, 5000, n_samples),
        'sqft_lot': np.random.randint(3000, 50000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.3, 0.1, 0.4, 0.1, 0.1]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'sqft_above': np.random.randint(500, 4000, n_samples),
        'sqft_basement': np.random.randint(0, 2000, n_samples),
        'yr_built': np.random.randint(1900, 2020, n_samples),
        'yr_renovated': np.random.choice([0] + list(range(1950, 2024)), n_samples, p=[0.7] + [0.3/74]*74),
        'street': [f"{np.random.randint(100, 9999)} {np.random.choice(['Main St', 'Oak Ave', 'Pine Rd', 'Cedar Ln', 'Maple Dr'])}" for _ in range(n_samples)],
        'city': np.random.choice(cities, n_samples),
        'statezip': np.random.choice(states, n_samples),
        'country': 'USA'
    })
    
    # Correction: sqft_above <= sqft_living
    data['sqft_above'] = np.minimum(data['sqft_above'], data['sqft_living'])
    
    # Génération des prix réalistes basés sur les features
    base_price = 300000
    data['price'] = (
        base_price +
        data['bedrooms'] * 30000 +
        data['bathrooms'] * 25000 +
        data['sqft_living'] * 200 +
        data['sqft_lot'] * 8 +
        data['floors'] * 20000 +
        data['waterfront'] * 300000 +
        data['view'] * 40000 +
        data['condition'] * 30000 +
        data['sqft_basement'] * 100 +
        (2024 - data['yr_built']) * -1500 +  # Dépréciation
        np.where(data['yr_renovated'] > 0, 75000, 0) +  # Bonus rénovation
        np.random.normal(0, 80000, n_samples)  # Bruit
    )
    
    # Ajustement par ville (prix plus élevés à Seattle/Bellevue)
    city_multipliers = {'Seattle': 1.4, 'Bellevue': 1.5, 'Redmond': 1.3, 'Kirkland': 1.2}
    for city, multiplier in city_multipliers.items():
        mask = data['city'] == city
        data.loc[mask, 'price'] *= multiplier
    
    # Prix minimum
    data['price'] = np.maximum(data['price'], 100000)
    data['price'] = data['price'].round(0)
    
    logger.info(f"Dataset généré: {data.shape[0]} maisons")
    logger.info(f"Prix moyen: ${data['price'].mean():,.0f}")
    logger.info(f"Prix médian: ${data['price'].median():,.0f}")
    
    return data


def train_models(X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue plusieurs modèles avec hyperparamètres optimisés
    """
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,      # Augmenté de 100 à 200
            max_depth=20,          # Augmenté pour capturer plus de complexité
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',   # Meilleure généralisation
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,      # Augmenté de 100 à 200
            learning_rate=0.05,    # Réduit pour meilleure généralisation
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,         # Évite overfitting
            max_features='sqrt',
            random_state=42
        )
    }
    
    # Ajouter XGBoost si disponible
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,            # Régularisation
            reg_alpha=0.1,        # L1 regularization
            reg_lambda=1,         # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    # Ajouter LightGBM si disponible
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    
    results = {}
    trained_models = {}
    
    logger.info("Entraînement des modèles de prédiction des prix")
    logger.info(f"Taille du dataset d'entraînement: {X_train.shape}")
    
    for name, model in models.items():
        logger.info(f"Entraînement {name}...")
        
        # Entraînement
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        
        # Calcul du MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }
        
        trained_models[name] = model
        
        logger.info(f"{name} - R²: {r2:.4f}, RMSE: ${rmse:,.0f}, MAPE: {mape:.1f}%, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return results, trained_models


def optimize_best_model(X_train, y_train, model_name: str):
    """
    Optimise les hyperparamètres du meilleur modèle avec GridSearchCV
    """
    logger.info(f"Optimisation des hyperparamètres pour {model_name}...")
    
    if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9]
        }
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    
    elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1],
            'num_leaves': [31, 50, 70]
        }
        model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
    
    elif model_name == 'GradientBoosting':
        param_grid = {
            'n_estimators': [150, 200, 250],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.03, 0.05, 0.08],
            'subsample': [0.7, 0.8, 0.9]
        }
        model = GradientBoostingRegressor(random_state=42)
    
    elif model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [150, 200, 300],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6]
        }
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    else:
        logger.info(f"Pas d'optimisation disponible pour {model_name}")
        return None
    
    # GridSearchCV
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3,  # 3-fold pour gagner du temps
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Meilleurs paramètres: {grid_search.best_params_}")
    logger.info(f"Meilleur score CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def main():
    """Fonction principale d'entraînement"""
    parser = argparse.ArgumentParser(description="Entraînement modèle House Price Prediction")
    parser.add_argument("--data", default="data/raw/data.csv", help="Chemin vers les données")
    parser.add_argument("--output", default="data/raw/output.csv", help="Chemin vers données de test")
    parser.add_argument("--save-sample", action="store_true", help="Sauvegarder dataset d'exemple")
    parser.add_argument("--optimize", action="store_true", help="Optimiser les hyperparamètres du meilleur modèle")
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("🏠 Début entraînement House Price Prediction")
    logger.info(f"Dataset cible: {args.data}")
    
    try:
        # Tentative de chargement du vrai dataset
        try:
            data = load_real_dataset(args.data)
        except FileNotFoundError:
            logger.warning(f"Dataset réel non trouvé: {args.data}")
            logger.info("Génération d'un dataset d'exemple...")
            
            data = generate_sample_dataset()
            
            # Sauvegarde optionnelle
            if args.save_sample:
                Path(args.data).parent.mkdir(parents=True, exist_ok=True)
                data.to_csv(args.data, index=False)
                logger.info(f"Dataset d'exemple sauvegardé: {args.data}")
                
                # Créer aussi le fichier de test
                test_data = data.sample(frac=0.3, random_state=42)
                test_data.to_csv(args.output, index=False)
                logger.info(f"Dataset de test sauvegardé: {args.output}")
        
        # Préparation des données selon les exigences du projet
        logger.info("Préparation des données...")
        
        # Import du processeur spécifique
        from src.mlops_package.data.make_dataset import HousePriceDataProcessor
        
        processor = HousePriceDataProcessor()
        processed_data = processor.prepare_features(data, fit=True)
        
        # Division train/test
        X_train, X_test, y_train, y_test = processor.split_data(processed_data)
        
        logger.info(f"Données préparées - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        logger.info(f"Features finales: {X_train.shape[1]}")
        
        # Sauvegarde des préprocesseurs
        processor.save_preprocessors()
        
        # Entraînement des modèles
        results, trained_models = train_models(X_train, X_test, y_train, y_test)
        
        # Sélection du meilleur modèle
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model = trained_models[best_model_name]
        
        logger.info(f"🏆 Meilleur modèle: {best_model_name}")
        logger.info(f"   R²: {results[best_model_name]['r2']:.4f}")
        logger.info(f"   RMSE: ${results[best_model_name]['rmse']:,.0f}")
        logger.info(f"   MAPE: {results[best_model_name]['mape']:.1f}%")
        
        # Optimisation optionnelle
        if args.optimize and best_model_name in ['XGBoost', 'LightGBM', 'GradientBoosting', 'RandomForest']:
            optimized_model = optimize_best_model(X_train, y_train, best_model_name)
            if optimized_model:
                # Réévaluer le modèle optimisé
                y_pred_opt = optimized_model.predict(X_test)
                r2_opt = r2_score(y_test, y_pred_opt)
                rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
                
                logger.info(f"🎯 Modèle optimisé - R²: {r2_opt:.4f}, RMSE: ${rmse_opt:,.0f}")
                
                if r2_opt > results[best_model_name]['r2']:
                    best_model = optimized_model
                    results[best_model_name]['r2_optimized'] = r2_opt
                    results[best_model_name]['rmse_optimized'] = rmse_opt
        
        # Sauvegarde du meilleur modèle
        Path("models").mkdir(exist_ok=True)
        model_path = "models/best_model.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"Modèle sauvegardé: {model_path}")
        
        # Rapport d'entraînement
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'rows': len(data),
                'rows_after_filtering': len(processed_data),
                'features': len(X_train.columns),
                'target': 'price'
            },
            'model_results': results,
            'best_model': best_model_name,
            'data_preprocessing': {
                'missing_values': 'interpolation',
                'outliers': 'clip',
                'price_filtering': 'applied',
                'feature_engineering': 'advanced'
            }
        }
        
        with open("models/training_report.json", "w") as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📊 Rapport d'entraînement sauvegardé: models/training_report.json")
        
        # Résumé final
        logger.info("=" * 60)
        logger.info("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info(f"📊 Dataset: {len(data)} maisons → {len(processed_data)} après filtrage")
        logger.info(f"📊 Features: {len(X_train.columns)} (incluant feature engineering)")
        logger.info(f"🏆 Meilleur modèle: {best_model_name}")
        logger.info(f"📈 Performance R²: {results[best_model_name]['r2']:.4f}")
        logger.info(f"💰 Erreur RMSE: ${results[best_model_name]['rmse']:,.0f}")
        logger.info(f"📊 Erreur MAPE: {results[best_model_name]['mape']:.1f}%")
        logger.info(f"💾 Modèle sauvé: {model_path}")
        logger.info("🚀 Prêt pour déploiement API!")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        raise


if __name__ == "__main__":
    main()