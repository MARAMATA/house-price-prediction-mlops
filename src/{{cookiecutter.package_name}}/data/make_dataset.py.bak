"""
Module de traitement des données pour le dataset House Prices
Dataset: {{cookiecutter.dataset_rows}} maisons avec {{cookiecutter.dataset_features}} features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from loguru import logger
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib


class HousePriceDataProcessor:
    """
    Processeur de données spécifique au dataset House Prices
    Gère les 18 colonnes : date, price, bedrooms, bathrooms, sqft_living, sqft_lot, 
    floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, 
    yr_renovated, street, city, statezip, country
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
        # Colonnes spécifiques du dataset
        self.numerical_cols = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
        ]
        self.categorical_cols = [
            'waterfront', 'view', 'condition', 'street', 'city', 'statezip', 'country'
        ]
        self.target_col = 'price'
        self.date_col = 'date'
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut pour le dataset"""
        return {
            'data': {
                'raw_path': 'data/raw/',
                'processed_path': 'data/processed/',
                'train_file': 'data.csv',
                'test_file': 'output.csv'
            },
            'model': {
                'preprocessing': {
                    'missing_values': 'interpolation',
                    'outliers': 'clip',
                    'price_filters': {
                        'min_price': 50000,
                        'max_price': 3000000
                    }
                }
            }
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Charge les données CSV
        
        Args:
            file_path: Chemin vers le fichier CSV
            
        Returns:
            DataFrame avec les {{cookiecutter.dataset_rows}} maisons
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Données chargées: {data.shape[0]} maisons, {data.shape[1]} features")
            
            # Vérification des colonnes attendues
            expected_cols = set(self.numerical_cols + self.categorical_cols + [self.target_col, self.date_col])
            actual_cols = set(data.columns)
            
            if not expected_cols.issubset(actual_cols):
                missing = expected_cols - actual_cols
                logger.warning(f"Colonnes manquantes: {missing}")
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyage avec filtrage des prix aberrants
        """
        logger.info("Nettoyage des données - Filtrage et interpolation")
        
        cleaned_data = data.copy()
        
        # NOUVEAU : Filtrer les prix aberrants
        if self.target_col in cleaned_data.columns:
            initial_count = len(cleaned_data)
            
            # Récupérer les seuils de la config ou utiliser les valeurs par défaut
            min_price = self.config.get('model', {}).get('preprocessing', {}).get('price_filters', {}).get('min_price', 50000)
            max_price = self.config.get('model', {}).get('preprocessing', {}).get('price_filters', {}).get('max_price', 3000000)
            
            # Filtrer prix = 0 ou prix extrêmes
            cleaned_data = cleaned_data[
                (cleaned_data[self.target_col] > min_price) & 
                (cleaned_data[self.target_col] < max_price)
            ]
            
            filtered_count = initial_count - len(cleaned_data)
            logger.info(f"Maisons filtrées (prix < ${min_price:,} ou > ${max_price:,}): {filtered_count}")
            logger.info(f"Dataset après filtrage: {len(cleaned_data)} maisons")
        
        # 1. Gestion valeurs manquantes par interpolation (comme demandé)
        initial_missing = cleaned_data.isnull().sum().sum()
        logger.info(f"Valeurs manquantes initiales: {initial_missing}")
        
        # Interpolation pour les variables numériques
        for col in self.numerical_cols:
            if col in cleaned_data.columns and cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col] = cleaned_data[col].interpolate(method='linear')
                logger.info(f"Interpolation appliquée à {col}")
        
        # Mode pour les variables catégorielles
        for col in self.categorical_cols:
            if col in cleaned_data.columns and cleaned_data[col].isnull().sum() > 0:
                mode_val = cleaned_data[col].mode()[0] if len(cleaned_data[col].mode()) > 0 else 'Unknown'
                cleaned_data[col].fillna(mode_val, inplace=True)
                logger.info(f"Mode appliqué à {col}: {mode_val}")
        
        final_missing = cleaned_data.isnull().sum().sum()
        logger.info(f"Valeurs manquantes après nettoyage: {final_missing}")
        
        # 2. Gestion des outliers (clipping) pour les features numériques
        for col in self.numerical_cols:
            if col in cleaned_data.columns:
                cleaned_data = self._handle_outliers(cleaned_data, col)
        
        logger.info(f"Données nettoyées: {cleaned_data.shape[0]} maisons")
        return cleaned_data
    
    def _handle_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Gestion des outliers par clipping"""
        Q1 = data[column].quantile(0.01)  # 1er percentile
        Q3 = data[column].quantile(0.99)  # 99e percentile
        
        outliers_count = ((data[column] < Q1) | (data[column] > Q3)).sum()
        if outliers_count > 0:
            logger.info(f"Outliers détectés dans {column}: {outliers_count}")
            data[column] = data[column].clip(lower=Q1, upper=Q3)
            logger.info(f"Outliers clippés pour {column}")
        
        return data
    
    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ingénierie des features avancée pour améliorer le R²
        """
        logger.info("Ingénierie des features - Ajout de variables dérivées avancées")
        
        enhanced_data = data.copy()
        
        # 1. Features temporelles depuis 'date'
        if self.date_col in enhanced_data.columns:
            enhanced_data[self.date_col] = pd.to_datetime(enhanced_data[self.date_col])
            enhanced_data['year'] = enhanced_data[self.date_col].dt.year
            enhanced_data['month'] = enhanced_data[self.date_col].dt.month
            enhanced_data['quarter'] = enhanced_data[self.date_col].dt.quarter
            enhanced_data['day_of_year'] = enhanced_data[self.date_col].dt.dayofyear
            enhanced_data['season'] = (enhanced_data['month'] % 12 + 3) // 3  # 1=Winter, 2=Spring, etc.
            logger.info("Features temporelles créées: year, month, quarter, day_of_year, season")
        
        # 2. Features de ratio et combinaisons
        if all(col in enhanced_data.columns for col in ['sqft_living', 'sqft_lot']):
            enhanced_data['living_lot_ratio'] = enhanced_data['sqft_living'] / (enhanced_data['sqft_lot'] + 1)
            logger.info("Feature créée: living_lot_ratio")
        
        if all(col in enhanced_data.columns for col in ['sqft_above', 'sqft_living']):
            enhanced_data['above_living_ratio'] = enhanced_data['sqft_above'] / (enhanced_data['sqft_living'] + 1)
            logger.info("Feature créée: above_living_ratio")
        
        if all(col in enhanced_data.columns for col in ['bedrooms', 'bathrooms']):
            enhanced_data['bed_bath_ratio'] = enhanced_data['bedrooms'] / (enhanced_data['bathrooms'] + 0.5)
            enhanced_data['total_rooms'] = enhanced_data['bedrooms'] + enhanced_data['bathrooms']
            enhanced_data['rooms_per_sqft'] = enhanced_data['total_rooms'] / (enhanced_data['sqft_living'] + 1)
            logger.info("Features créées: bed_bath_ratio, total_rooms, rooms_per_sqft")
        
        # 3. Age de la maison
        if 'yr_built' in enhanced_data.columns:
            current_year = pd.Timestamp.now().year
            enhanced_data['house_age'] = current_year - enhanced_data['yr_built']
            enhanced_data['age_squared'] = enhanced_data['house_age'] ** 2  # Relation non-linéaire
            logger.info("Features créées: house_age, age_squared")
        
        # 4. Indicateur de rénovation
        if 'yr_renovated' in enhanced_data.columns:
            enhanced_data['is_renovated'] = (enhanced_data['yr_renovated'] > 0).astype(int)
            enhanced_data['years_since_renovation'] = np.where(
                enhanced_data['yr_renovated'] > 0,
                pd.Timestamp.now().year - enhanced_data['yr_renovated'],
                0
            )
            enhanced_data['effective_age'] = np.where(
                enhanced_data['yr_renovated'] > 0,
                pd.Timestamp.now().year - enhanced_data['yr_renovated'],
                enhanced_data['house_age'] if 'house_age' in enhanced_data.columns else 0
            )
            logger.info("Features créées: is_renovated, years_since_renovation, effective_age")
        
        # 5. Features de qualité
        if all(col in enhanced_data.columns for col in ['condition', 'view']):
            enhanced_data['quality_score'] = enhanced_data['condition'] * 2 + enhanced_data['view']
            enhanced_data['high_quality'] = ((enhanced_data['condition'] >= 4) & (enhanced_data['view'] >= 2)).astype(int)
            logger.info("Features créées: quality_score, high_quality")
        
        # 6. Features de surface
        if all(col in enhanced_data.columns for col in ['sqft_above', 'sqft_basement']):
            enhanced_data['total_sqft'] = enhanced_data['sqft_above'] + enhanced_data['sqft_basement']
            enhanced_data['basement_ratio'] = enhanced_data['sqft_basement'] / (enhanced_data['total_sqft'] + 1)
            enhanced_data['has_basement'] = (enhanced_data['sqft_basement'] > 0).astype(int)
            logger.info("Features créées: total_sqft, basement_ratio, has_basement")
        
        # 7. NOUVEAU : Log transformation des surfaces (linéarise la relation avec le prix)
        if 'sqft_living' in enhanced_data.columns:
            enhanced_data['log_sqft_living'] = np.log1p(enhanced_data['sqft_living'])
            enhanced_data['log_sqft_lot'] = np.log1p(enhanced_data['sqft_lot'])
            enhanced_data['log_total_sqft'] = np.log1p(enhanced_data.get('total_sqft', enhanced_data['sqft_living']))
            logger.info("Features créées: log_sqft_living, log_sqft_lot, log_total_sqft")
        
        # 8. NOUVEAU : Features de localisation avancées
        if 'city' in enhanced_data.columns and self.target_col in enhanced_data.columns:
            # Prix médian par ville (très prédictif!)
            city_price_median = enhanced_data.groupby('city')[self.target_col].transform('median')
            enhanced_data['city_price_median'] = city_price_median
            
            # Nombre de maisons par ville (indicateur de popularité)
            city_count = enhanced_data.groupby('city')['city'].transform('count')
            enhanced_data['city_popularity'] = city_count
            logger.info("Features créées: city_price_median, city_popularity")
        
        # 9. NOUVEAU : Interactions importantes
        if all(col in enhanced_data.columns for col in ['sqft_living', 'condition']):
            enhanced_data['sqft_living_condition'] = enhanced_data['sqft_living'] * enhanced_data['condition']
            logger.info("Feature créée: sqft_living_condition (interaction)")
        
        if all(col in enhanced_data.columns for col in ['sqft_living', 'house_age']):
            enhanced_data['sqft_age_interaction'] = enhanced_data['sqft_living'] * np.log1p(enhanced_data['house_age'])
            logger.info("Feature créée: sqft_age_interaction")
        
        # 10. NOUVEAU : Score de luxe composite
        if all(col in enhanced_data.columns for col in ['waterfront', 'view', 'condition']):
            enhanced_data['luxury_score'] = (
                enhanced_data['waterfront'] * 5 + 
                enhanced_data['view'] * 2 + 
                (enhanced_data['condition'] >= 4).astype(int) * 3 +
                (enhanced_data.get('sqft_living', 0) > enhanced_data.get('sqft_living', 0).quantile(0.75)).astype(int) * 2
            )
            logger.info("Feature créée: luxury_score")
        
        # 11. NOUVEAU : Ratios de prix au m²
        if 'sqft_living' in enhanced_data.columns:
            enhanced_data['price_per_sqft_potential'] = enhanced_data.get('city_price_median', 0) / (enhanced_data['sqft_living'] + 1)
            logger.info("Feature créée: price_per_sqft_potential")
        
        logger.info(f"Ingénierie terminée. Nouvelles dimensions: {enhanced_data.shape}")
        return enhanced_data
    
    def encode_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encodage des variables catégorielles avec gestion améliorée"""
        logger.info("Encodage des variables catégorielles")
        
        encoded_data = data.copy()
        
        # Pour 'street', utiliser un encodage de fréquence plutôt que LabelEncoder (trop de catégories)
        if 'street' in encoded_data.columns:
            if fit:
                street_freq = encoded_data['street'].value_counts().to_dict()
                self.label_encoders['street_freq'] = street_freq
                encoded_data['street_frequency'] = encoded_data['street'].map(street_freq)
            else:
                street_freq = self.label_encoders.get('street_freq', {})
                encoded_data['street_frequency'] = encoded_data['street'].map(street_freq).fillna(1)
            
            # Supprimer la colonne originale street (trop de catégories)
            encoded_data = encoded_data.drop('street', axis=1)
            logger.info(f"Street encodé par fréquence: {len(street_freq)} adresses uniques")
        
        # Encoder les autres catégorielles normalement
        for feature in [col for col in self.categorical_cols if col != 'street']:
            if feature in encoded_data.columns:
                if fit:
                    le = LabelEncoder()
                    encoded_data[feature] = le.fit_transform(encoded_data[feature].astype(str))
                    self.label_encoders[feature] = le
                    logger.info(f"Encodeur créé pour {feature}: {len(le.classes_)} classes")
                else:
                    if feature in self.label_encoders:
                        le = self.label_encoders[feature]
                        # Gérer les nouvelles catégories
                        encoded_data[feature] = encoded_data[feature].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return encoded_data
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalisation des features numériques"""
        logger.info("Normalisation des features numériques")
        
        scaled_data = data.copy()
        
        # Features à normaliser (numériques + dérivées)
        numeric_features = [col for col in self.numerical_cols if col in scaled_data.columns]
        
        # Ajouter toutes les features dérivées numériques
        derived_numeric = [
            'living_lot_ratio', 'above_living_ratio', 'bed_bath_ratio', 'total_rooms',
            'house_age', 'years_since_renovation', 'quality_score', 'total_sqft', 'basement_ratio',
            'year', 'month', 'quarter', 'day_of_year', 'season', 'age_squared', 'effective_age',
            'log_sqft_living', 'log_sqft_lot', 'log_total_sqft', 'city_price_median',
            'city_popularity', 'sqft_living_condition', 'sqft_age_interaction', 'luxury_score',
            'price_per_sqft_potential', 'street_frequency', 'rooms_per_sqft', 'high_quality',
            'has_basement', 'is_renovated'
        ]
        numeric_features.extend([col for col in derived_numeric if col in scaled_data.columns])
        
        # Retirer les doublons
        numeric_features = list(set(numeric_features))
        
        if fit:
            scaled_data[numeric_features] = self.scaler.fit_transform(scaled_data[numeric_features])
            logger.info(f"Scaler ajusté pour {len(numeric_features)} features")
        else:
            scaled_data[numeric_features] = self.scaler.transform(scaled_data[numeric_features])
        
        return scaled_data
    
    def prepare_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Pipeline complet de préparation des features"""
        logger.info("Pipeline de préparation des features")
        
        processed_data = data.copy()
        processed_data = self.clean_data(processed_data)
        processed_data = self.feature_engineering(processed_data)
        processed_data = self.encode_categorical_features(processed_data, fit=fit)
        processed_data = self.scale_features(processed_data, fit=fit)
        
        if fit:
            self.is_fitted = True
        
        logger.info(f"Préparation des features terminée. Shape finale: {processed_data.shape}")
        return processed_data
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Division des données"""
        # Retirer les colonnes non-features
        X = data.drop(columns=[self.target_col, self.date_col], errors='ignore')
        y = data[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        logger.info(f"Division: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
        logger.info(f"Features utilisées: {X_train.shape[1]}")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessors(self, save_path: str = "models/preprocessors.joblib"):
        """Sauvegarde des préprocesseurs"""
        preprocessors = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'is_fitted': self.is_fitted,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'config': self.config
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessors, save_path)
        logger.info(f"Préprocesseurs sauvegardés: {save_path}")