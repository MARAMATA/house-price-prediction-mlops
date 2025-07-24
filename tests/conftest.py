"""
Configuration pytest pour les tests du projet House Price Prediction
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def sample_house_dataset():
    """
    Dataset d'exemple pour les tests
    Reproduit la structure exacte du vrai dataset
    """
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n_samples),
        'price': np.random.uniform(200000, 800000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples),
        'sqft_living': np.random.randint(800, 4000, n_samples),
        'sqft_lot': np.random.randint(5000, 20000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
        'waterfront': np.random.choice([0, 1], n_samples),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'sqft_above': np.random.randint(800, 3500, n_samples),
        'sqft_basement': np.random.randint(0, 1500, n_samples),
        'yr_built': np.random.randint(1950, 2020, n_samples),
        'yr_renovated': np.random.choice([0] + list(range(1980, 2023)), n_samples),
        'street': [f"{i} Test St" for i in range(n_samples)],
        'city': np.random.choice(['Seattle', 'Bellevue', 'Redmond'], n_samples),
        'statezip': np.random.choice(['WA 98101', 'WA 98004'], n_samples),
        'country': 'USA'
    })


@pytest.fixture
def valid_house_features():
    """Features valides pour une maison"""
    return {
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
    }


@pytest.fixture
def project_paths():
    """Chemins importants du projet"""
    root = Path(__file__).parent.parent
    return {
        'root': root,
        'data_raw': root / 'data' / 'raw',
        'data_processed': root / 'data' / 'processed',
        'models': root / 'models',
        'logs': root / 'logs'
    }
