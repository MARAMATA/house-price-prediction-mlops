"""
Tests unitaires pour le processeur de données House Price
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le src au path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from {{cookiecutter.package_name}}.data.make_dataset import HousePriceDataProcessor


class TestHousePriceDataProcessor:
    """Tests pour le processeur de données spécifique au dataset"""
    
    @pytest.fixture
    def sample_data(self):
        """Données d'exemple pour les tests"""
        return pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=100),
            'price': np.random.uniform(200000, 800000, 100),
            'bedrooms': np.random.randint(1, 6, 100),
            'bathrooms': np.random.uniform(1, 4, 100),
            'sqft_living': np.random.randint(800, 4000, 100),
            'sqft_lot': np.random.randint(5000, 20000, 100),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], 100),
            'waterfront': np.random.choice([0, 1], 100),
            'view': np.random.randint(0, 5, 100),
            'condition': np.random.randint(1, 6, 100),
            'sqft_above': np.random.randint(800, 3500, 100),
            'sqft_basement': np.random.randint(0, 1500, 100),
            'yr_built': np.random.randint(1950, 2020, 100),
            'yr_renovated': np.random.choice([0] + list(range(1980, 2023)), 100),
            'street': [f"{i} Main St" for i in range(100)],
            'city': np.random.choice(['Seattle', 'Bellevue', 'Redmond'], 100),
            'statezip': np.random.choice(['WA 98101', 'WA 98004'], 100),
            'country': 'USA'
        })
    
    @pytest.fixture
    def processor(self):
        """Instance du processeur"""
        return HousePriceDataProcessor()
    
    def test_initialization(self, processor):
        """Test de l'initialisation"""
        assert processor is not None
        assert len(processor.numerical_cols) == 9
        assert len(processor.categorical_cols) == 7
        assert processor.target_col == 'price'
    
    def test_clean_data(self, processor, sample_data):
        """Test du nettoyage des données"""
        # Ajouter quelques valeurs manquantes
        sample_data.loc[0:5, 'bedrooms'] = np.nan
        sample_data.loc[10:15, 'city'] = np.nan
        
        cleaned = processor.clean_data(sample_data)
        
        # Vérifier qu'il n'y a plus de valeurs manquantes
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) <= len(sample_data)  # Peut être réduit par outlier removal
    
    def test_feature_engineering(self, processor, sample_data):
        """Test de l'ingénierie des features"""
        enhanced = processor.feature_engineering(sample_data)
        
        # Vérifier que de nouvelles features ont été créées
        assert 'year' in enhanced.columns
        assert 'month' in enhanced.columns
        assert 'house_age' in enhanced.columns
        assert 'is_renovated' in enhanced.columns
        assert 'living_lot_ratio' in enhanced.columns
        
        # Vérifier que les dimensions ont augmenté
        assert enhanced.shape[1] > sample_data.shape[1]
    
    def test_encode_categorical_features(self, processor, sample_data):
        """Test de l'encodage des features catégorielles"""
        encoded = processor.encode_categorical_features(sample_data, fit=True)
        
        # Vérifier que les encodeurs ont été créés
        assert len(processor.label_encoders) > 0
        assert 'city' in processor.label_encoders
        
        # Vérifier que les valeurs sont numériques
        assert encoded['city'].dtype in [np.int32, np.int64]
    
    def test_scale_features(self, processor, sample_data):
        """Test de la normalisation"""
        # D'abord l'ingénierie pour avoir toutes les features
        enhanced = processor.feature_engineering(sample_data)
        scaled = processor.scale_features(enhanced, fit=True)
        
        # Vérifier que le scaler a été ajusté
        assert processor.scaler is not None
        
        # Vérifier que les features numériques sont normalisées
        numeric_cols = [col for col in processor.numerical_cols if col in scaled.columns]
        if numeric_cols:
            # Les valeurs doivent être approximativement centrées et réduites
            for col in numeric_cols[:3]:  # Tester quelques colonnes
                assert abs(scaled[col].mean()) < 0.1  # Proche de 0
                assert abs(scaled[col].std() - 1) < 0.1  # Proche de 1
    
    def test_prepare_features_pipeline(self, processor, sample_data):
        """Test du pipeline complet"""
        processed = processor.prepare_features(sample_data, fit=True)
        
        # Vérifier que le processeur est configuré
        assert processor.is_fitted
        
        # Vérifier les dimensions
        assert processed.shape[0] <= sample_data.shape[0]  # Peut perdre des lignes (outliers)
        assert processed.shape[1] > sample_data.shape[1]   # Doit gagner des colonnes
        
        # Vérifier qu'il n'y a pas de valeurs manquantes
        assert processed.isnull().sum().sum() == 0
    
    def test_split_data(self, processor, sample_data):
        """Test de la division des données"""
        processed = processor.prepare_features(sample_data, fit=True)
        X_train, X_test, y_train, y_test = processor.split_data(processed)
        
        # Vérifier les dimensions
        total_samples = len(processed)
        assert len(X_train) + len(X_test) == total_samples
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Vérifier que la cible n'est pas dans X
        assert 'price' not in X_train.columns
        assert 'price' not in X_test.columns
