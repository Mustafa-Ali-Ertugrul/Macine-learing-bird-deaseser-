#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin Daily PnL Analysis System - Enhanced Version

Bu sistem Bitcoin fiyat verilerini analiz eder, makine öğrenmesi modelleri ile
tahminler yapar ve kapsamlı risk analizi sağlar.

Author: AI Assistant
Version: 4.0
Last Updated: 2024
"""

# Standard library imports
import os
import json
import time
import datetime
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Union, Any
import logging

# Third-party imports
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pickle

# Machine learning imports
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats

# Configure warnings and logging
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Config:
    """Enhanced configuration settings with validation and security improvements"""
    
    # API settings
    api_url: str = "https://api.binance.com/api/v3/klines"
    symbol: str = "BTCUSDT"
    interval: str = "1d"
    limit: int = 500
    
    # File settings - using relative paths for security
    output_file: str = "btc_daily_pnl_improved.html"
    history_file: str = "btc_prediction_history.json"
    model_file: str = "btc_ensemble_model.pkl"
    scaler_file: str = "btc_scaler.pkl"
    
    # ML settings with validation
    lookback_days: int = 10
    feature_selection_k: int = 20
    train_test_split_ratio: float = 0.8
    cv_folds: int = 5
    
    # Network settings
    max_retries: int = 3
    backoff_factor: float = 0.3
    timeout: int = 30
    
    # Risk settings
    var_confidence: float = 0.05
    max_drawdown_threshold: float = -0.15
    
    # Prediction settings
    prediction_days: int = 7
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.limit <= 0 or self.limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if not 0 < self.train_test_split_ratio < 1:
            raise ValueError("Train test split ratio must be between 0 and 1")
        if self.cv_folds < 2:
            raise ValueError("CV folds must be at least 2")
        if self.lookback_days < 1:
            raise ValueError("Lookback days must be at least 1")
        
        # Ensure output directory exists
        output_dir = Path(self.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

class AdvancedBTCAnalyzer:
    """Enhanced Bitcoin analysis and prediction system with improved error handling"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the analyzer with configuration validation"""
        try:
            self.config = config or Config()
            self.logger = logging.getLogger(__name__)
            self.session = self._create_session()
            
            # Initialize ML components
            self.scaler: Optional[RobustScaler] = None
            self.feature_selector: Optional[SelectKBest] = None
            self.ensemble_models: Dict[str, Any] = {}
            self.feature_importance: Dict[str, np.ndarray] = {}
            self.prediction_history: List[Dict] = []
            
            # Load existing models if available
            self._safe_load_models()
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise
    
    def _safe_load_models(self) -> None:
        """Safely load existing models with error handling"""
        try:
            self.load_ensemble_model()
            self.load_prediction_history()
            self.logger.info("Models loaded successfully")
        except FileNotFoundError:
            self.logger.info("No existing models found, will train new ones")
        except Exception as e:
            self.logger.warning(f"Error loading models: {e}. Will train new ones.")
        
    def _create_session(self) -> requests.Session:
        """Create HTTP session with enhanced security and retry logic"""
        session = requests.Session()
        
        # Enhanced retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],  # Only allow GET requests
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Security headers
        session.headers.update({
            'User-Agent': 'BTC-Analysis-Tool/2.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from Binance API with enhanced security and validation"""
        self.logger.info("Fetching data from Binance API...")
        
        # Validate parameters
        if not self.config.symbol or not self.config.symbol.isalnum():
            raise ValueError("Invalid symbol format")
        
        # Build URL with parameter validation
        params = {
            'symbol': self.config.symbol,
            'interval': self.config.interval,
            'limit': min(self.config.limit, 1000)  # API limit
        }
        
        try:
            response = self.session.get(
                self.config.api_url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            # Validate response content type
            if 'application/json' not in response.headers.get('content-type', ''):
                raise ValueError("Invalid response content type")
            data = response.json()
            
            # Validate response data
            if not data or not isinstance(data, list):
                raise ValueError("Invalid API response format")
            
            if len(data) < 10:
                raise ValueError("Insufficient data received from API")
            
            # Create DataFrame with proper column mapping
            columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                      'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                      'taker_buy_quote', 'ignore']
            
            df = pd.DataFrame(data, columns=columns)
            
            # Convert data types with error handling
            try:
                df['date'] = pd.to_datetime(df['open_time'], unit='ms')
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            except Exception as e:
                raise ValueError(f"Data conversion error: {e}")
            
            # Validate converted data
            if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
                raise ValueError("Invalid price data detected")
            
            # Basic price validation
            if (df['high'] < df['low']).any() or (df['close'] <= 0).any():
                raise ValueError("Invalid price relationships detected")
            
            # Select and sort data
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date')
            df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Successfully fetched {len(df)} days of data")
            return df
            
        except requests.RequestException as e:
            self.logger.error(f"API request error: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Data validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in data fetching: {e}")
            raise
    
    def _add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators with optimized performance"""
        self.logger.info("Calculating advanced technical indicators...")
        
        if len(df) < 50:
            self.logger.warning("Insufficient data for reliable technical indicators")
        
        # Basic price changes
        df['pnl'] = df['close'].diff()
        df['pnl_pct'] = df['close'].pct_change()
        
        # Vectorized calculations for better performance
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        # Çoklu hareketli ortalamalar
        for window in [5, 7, 12, 14, 21, 26, 50, 100]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Hareketli ortalama çaprazlamaları
        df['sma_cross_7_21'] = np.where(df['sma_7'] > df['sma_21'], 1, 0)
        df['sma_cross_21_50'] = np.where(df['sma_21'] > df['sma_50'], 1, 0)
        df['ema_cross_12_26'] = np.where(df['ema_12'] > df['ema_26'], 1, 0)
        
        # Volatilite göstergeleri
        for window in [7, 14, 21, 30]:
            df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(window=50).mean()
        
        # Çoklu RSI periyotları
        for period in [14, 21, 30]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # MACD ailesi
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_slope'] = df['macd'].diff()
        
        # Bollinger Bands - çoklu periyot
        for bb_window, bb_std in [(20, 2), (20, 1.5), (10, 2)]:
            bb_middle = df['close'].rolling(window=bb_window).mean()
            bb_std_dev = df['close'].rolling(window=bb_window).std()
            df[f'bb_upper_{bb_window}_{bb_std}'] = bb_middle + (bb_std_dev * bb_std)
            df[f'bb_lower_{bb_window}_{bb_std}'] = bb_middle - (bb_std_dev * bb_std)
            df[f'bb_position_{bb_window}_{bb_std}'] = (df['close'] - df[f'bb_lower_{bb_window}_{bb_std}']) / (df[f'bb_upper_{bb_window}_{bb_std}'] - df[f'bb_lower_{bb_window}_{bb_std}'])
            df[f'bb_squeeze_{bb_window}'] = (df[f'bb_upper_{bb_window}_{bb_std}'] - df[f'bb_lower_{bb_window}_{bb_std}']) / bb_middle
        
        # Volume göstergeleri
        df['volume_sma_21'] = df['volume'].rolling(window=21).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_21']
        df['volume_rsi'] = self._calculate_rsi(df['volume'], 14)
        df['price_volume_trend'] = (df['close'].diff() / df['close'].shift(1)) * df['volume']
        
        # Momentum göstergeleri - çoklu periyot
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Yüksek-düşük oranları ve range göstergeleri
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']
        df['true_range'] = np.maximum(df['high'] - df['low'], 
                                     np.maximum(abs(df['high'] - df['close'].shift(1)),
                                               abs(df['low'] - df['close'].shift(1))))
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        
        # Stochastic Oscillator
        df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / 
                           (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
        
        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['cci_20'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
        
        # Lag features - geçmiş fiyat bilgileri
        for lag in [1, 2, 3, 5, 7, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['pnl_pct'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Regime detection - trend rejimi
        df['regime_sma'] = np.where(df['sma_21'] > df['sma_50'], 1, 0)
        df['regime_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, 0)
        df['regime_price'] = np.where(df['close'] > df['sma_21'], 1, 0)
        
        print("✅ Geliştirilmiş teknik göstergeler hesaplandı")
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _detect_advanced_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gelişmiş outlier detection"""
        print("Gelişmiş outlier detection uygulanıyor...")
        
        # IQR method
        Q1 = df['pnl'].quantile(0.25)
        Q3 = df['pnl'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df['is_outlier_iqr'] = (df['pnl'] < lower_bound) | (df['pnl'] > upper_bound)
        
        # Z-score method
        z_scores = np.abs(stats.zscore(df['pnl'].dropna()))
        df.loc[df['pnl'].notna(), 'is_outlier_zscore'] = z_scores > 3
        
        # Combined outlier flag
        df['is_outlier'] = df['is_outlier_iqr'] | df.get('is_outlier_zscore', False)
        
        outlier_count = df['is_outlier'].sum()
        if outlier_count > 0:
            print(f"⚠️ {outlier_count} outlier tespit edildi")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Gelişmiş özellik mühendisliği"""
        print("Gelişmiş özellik mühendisliği yapılıyor...")
        
        # Önce teknik göstergeleri hesapla (pnl sütunu burada oluşturulur)
        df = self._add_advanced_technical_indicators(df)
        
        # Outlier detection
        df = self._detect_advanced_outliers(df)
        
        # Feature columns - otomatik seçim
        feature_cols = [col for col in df.columns if col not in ['date', 'close', 'pnl', 'is_outlier', 'is_outlier_iqr', 'is_outlier_zscore']]
        
        # Numeric columns only
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"💡 Kullanılabilir özellik sayısı: {len(numeric_cols)}")
        
        # NaN değerleri temizle
        df_clean = df.dropna()
        
        if len(df_clean) < self.config.lookback_days + 20:
            raise ValueError("Yeterli temiz veri yok")
        
        X, y = [], []
        
        # Sliding window ile özellik oluştur
        for i in range(self.config.lookback_days, len(df_clean)):
            # Son N günün özelliklerini al
            features = []
            for j in range(self.config.lookback_days):
                try:
                    row_idx = i - self.config.lookback_days + j
                    row_features = df_clean[numeric_cols].iloc[row_idx].values
                    # NaN kontrolü
                    row_features = np.nan_to_num(row_features, nan=0.0, posinf=0.0, neginf=0.0)
                    features.extend(row_features)
                except Exception as e:
                    print(f"Özellik oluşturma hatası: {e}")
                    continue
            
            if len(features) > 0:
                X.append(features)
                y.append(df_clean['close'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Feature names oluştur
        feature_names = []
        for j in range(self.config.lookback_days):
            for col in numeric_cols:
                feature_names.append(f"{col}_lag_{j}")
        
        print(f"✅ {X.shape[0]} örnek, {X.shape[1]} özellik oluşturuldu")
        return X, y, feature_names
    
    def optimize_hyperparameters(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray) -> object:
        """Hyperparameter optimization"""
        print("Hyperparameter optimizasyonu yapılıyor...")
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        print(f"En iyi parametreler: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train ensemble models with enhanced performance and validation"""
        self.logger.info("Training ensemble models...")
        
        # Validate input data
        if len(X) < 50:
            raise ValueError("Insufficient training data")
        
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Training data contains NaN values")
        
        # Time series aware train-test split
        split_idx = int(len(X) * self.config.train_test_split_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Feature selection
        print("En iyi özellikler seçiliyor...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(self.config.feature_selection_k, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Scaling
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Model definitions with hyperparameter grids
        model_configs = {
            'ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'lasso': {
                'model': Lasso(),
                'params': {'alpha': [0.001, 0.01, 0.1, 1.0]}
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'min_samples_split': [5, 10]
                }
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20]
                }
            },
            'gradient_boost': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [6, 10],
                    'learning_rate': [0.01, 0.1]
                }
            }
        }
        
        results = {}
        trained_models = {}
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        for name, config in model_configs.items():
            print(f"  📊 {name} modeli eğitiliyor...")
            
            try:
                # Hyperparameter optimization
                best_model = self.optimize_hyperparameters(
                    config['model'], 
                    config['params'], 
                    X_train_scaled, 
                    y_train
                )
                
                # Cross validation
                cv_scores = cross_val_score(
                    best_model, X_train_scaled, y_train, 
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                
                # Test predictions
                y_pred = best_model.predict(X_test_scaled)
                
                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                cv_rmse = np.sqrt(-cv_scores.mean())
                cv_std = np.sqrt(-cv_scores).std()
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    feature_importance = np.abs(best_model.coef_)
                
                results[name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_rmse': cv_rmse,
                    'cv_std': cv_std,
                    'feature_importance': feature_importance,
                    'y_pred': y_pred,
                    'y_test': y_test
                }
                
                trained_models[name] = best_model
                
                print(f"    RMSE: {rmse:.2f}, R²: {r2:.3f}, CV-RMSE: {cv_rmse:.2f}±{cv_std:.2f}")
                
            except Exception as e:
                print(f"    ❌ {name} model eğitim hatası: {str(e)}")
                continue
        
        # En iyi modeli seç
        if results:
            best_model_name = min(results.keys(), key=lambda x: results[x]['cv_rmse'])
            print(f"🏆 En iyi model: {best_model_name}")
            
            # Ensemble prediction (top 3 model)
            top_models = sorted(results.items(), key=lambda x: x[1]['cv_rmse'])[:3]
            print(f"🎯 Top 3 model: {[model[0] for model in top_models]}")
            
            self.ensemble_models = {name: trained_models[name] for name, _ in top_models}
            
            # Modelleri kaydet
            self.save_ensemble_model(self.ensemble_models)
            
            # En iyi modelin tahminlerini al
            best_model = trained_models[best_model_name]
            best_predictions = best_model.predict(X_test_scaled)
            
            return {
                'results': results,
                'best_model': best_model_name,
                'X_test': X_test_scaled,
                'y_test': y_test,
                'trained_models': trained_models,
                'predictions': best_predictions.tolist(),
                'actual': y_test.tolist()
            }
        else:
            raise Exception("Hiçbir model eğitilemedi!")
    
    def create_ensemble_prediction(self, X: np.ndarray) -> float:
        """Ensemble prediction with weighted average"""
        if not self.ensemble_models:
            raise ValueError("Ensemble modeller mevcut değil")
        
        if not hasattr(self, 'feature_selector') or not hasattr(self, 'scaler'):
            raise ValueError("Feature selector ve scaler mevcut değil")
        
        # Apply feature selection and scaling
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        
        predictions = []
        weights = []
        
        for name, model in self.ensemble_models.items():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
            # Daha iyi modellere daha yüksek ağırlık (1/rmse ile ağırlık)
            weight = 1.0  # Basit eşit ağırlık, geliştirilebilir
            weights.append(weight)
        
        # Weighted average
        weighted_pred = np.average(predictions, weights=weights)
        return float(weighted_pred)
    
    def generate_future_predictions(self, df: pd.DataFrame, days: int = 7) -> List[Dict]:
        """Gelecek için çoklu gün tahmini"""
        print(f"Gelecek {days} gün için tahmin yapılıyor...")
        
        if not self.ensemble_models:
            raise ValueError("Model eğitilmemiş")
        
        future_predictions = []
        current_date = df['date'].iloc[-1]
        current_data = df.copy()
        
        for i in range(days):
            try:
                # Özellik oluştur
                X, y, _ = self.create_advanced_features(current_data)
                
                if len(X) == 0:
                    break
                
                # Tahmin yap
                prediction = self.create_ensemble_prediction(X[-1:, :])
                
                # Yeni tahmin tarihini hesapla
                future_date = current_date + pd.Timedelta(days=i+1)
                
                # Confidence interval (basit yaklaşım)
                historical_errors = np.std([abs(y[j] - self.create_ensemble_prediction(X[j:j+1, :])) for j in range(max(0, len(X)-30), len(X))])
                confidence_lower = prediction - 1.96 * historical_errors
                confidence_upper = prediction + 1.96 * historical_errors
                
                future_predictions.append({
                    'date': future_date,
                    'prediction': prediction,
                    'confidence_lower': confidence_lower,
                    'confidence_upper': confidence_upper,
                    'day': i + 1
                })
                
                # Yeni satır ekle (tahmin edilen değerle)
                new_row = current_data.iloc[-1].copy()
                new_row['date'] = future_date
                new_row['close'] = prediction
                new_row['open'] = prediction
                new_row['high'] = prediction * 1.02  # Basit yaklaşım
                new_row['low'] = prediction * 0.98   # Basit yaklaşım
                
                # DataFrame'e ekle
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                
                # Teknik göstergeleri yeniden hesapla
                current_data = self._add_advanced_technical_indicators(current_data)
                
            except Exception as e:
                print(f"Gün {i+1} tahmin hatası: {e}")
                break
        
        print(f"✅ {len(future_predictions)} günlük tahmin oluşturuldu")
        return future_predictions
    
    def calculate_advanced_risk_metrics(self, prices: pd.Series) -> Dict:
        """Gelişmiş risk metriklerini hesapla"""
        if len(prices) < 2:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'var_95': 0,
                'var_99': 0,
                'expected_shortfall_95': 0,
                'skewness': 0,
                'kurtosis': 0,
                'total_return': 0,
                'win_rate': 0,
                'best_day': 0,
                'worst_day': 0
            }
        
        # Fiyat verilerinden günlük getirileri hesapla
        returns = prices.pct_change().dropna()
        
        if len(returns) == 0:
            return self.calculate_advanced_risk_metrics(pd.Series([100, 101]))
        
        # Aşırı değerleri temizle (±20% günlük değişim - Bitcoin için daha gerçekçi)
        returns_clean = returns[(returns > -0.2) & (returns < 0.2)]
        
        if len(returns_clean) == 0:
            returns_clean = returns
        
        # Temel istatistikler
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        
        if std_return == 0 or np.isnan(std_return):
            std_return = 0.01  # %1 varsayılan volatilite
        
        # Sharpe Ratio (risk-free rate = 0 varsayımı)
        sharpe_ratio = (mean_return * np.sqrt(252)) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        
        # Sortino Ratio (sadece negatif getiriler)
        negative_returns = returns_clean[returns_clean < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else std_return
        if downside_std == 0 or np.isnan(downside_std):
            downside_std = std_return
        sortino_ratio = (mean_return * np.sqrt(252)) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Eğer max_drawdown çok büyükse (-%100'den fazla), sınırla
        if max_drawdown < -1.0:
            max_drawdown = -0.99  # Maksimum %99 drawdown
        
        # Calmar Ratio
        annual_return = mean_return * 252
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Value at Risk (multiple confidence levels)
        var_95 = returns_clean.quantile(0.05)
        var_99 = returns_clean.quantile(0.01)
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = returns_clean[returns_clean <= var_95]
        es_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns_clean) if len(returns_clean) > 2 else 0
        kurtosis = stats.kurtosis(returns_clean) if len(returns_clean) > 2 else 0
        
        # Volatility (annualized, percentage)
        volatility = std_return * np.sqrt(252) * 100  # Yüzde olarak
        
        # Volatilite çok yüksekse sınırla (Bitcoin için maksimum %200 yıllık volatilite)
        if volatility > 200:
            volatility = min(volatility, 200)
        
        return {
            'sharpe_ratio': round(sharpe_ratio, 3),
            'sortino_ratio': round(sortino_ratio, 3),
            'calmar_ratio': round(calmar_ratio, 3),
            'max_drawdown': round(max_drawdown * 100, 2),  # Yüzde olarak
            'volatility': round(volatility, 2),
            'var_95': round(var_95 * 100, 2),  # Yüzde olarak
            'var_99': round(var_99 * 100, 2),  # Yüzde olarak
            'expected_shortfall_95': round(es_95 * 100, 2),  # Yüzde olarak
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            'total_return': round(min((cumulative.iloc[-1] - 1) * 100, 1000), 2) if len(cumulative) > 0 else 0,  # Yüzde olarak, maksimum %1000
            'win_rate': round(min((returns_clean > 0).mean() * 100, 100), 2),  # Yüzde olarak, maksimum %100
            'best_day': round(returns_clean.max() * 100, 2),  # Yüzde olarak
            'worst_day': round(returns_clean.min() * 100, 2)  # Yüzde olarak
        }
    
    def save_ensemble_model(self, models_dict: Dict, model_name: str = None):
        """Ensemble modelleri kaydet"""
        model_path = model_name or self.config.model_file
        full_path = os.path.abspath(model_path)
        
        try:
            model_data = {
                'models': models_dict,
                'feature_selector': self.feature_selector,
                'scaler': self.scaler
            }
            with open(full_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Ensemble model kaydedildi: {full_path}")
        except Exception as e:
            print(f"❌ Model kaydetme hatası: {str(e)}")
    
    def load_ensemble_model(self, model_name: str = None):
        """Ensemble modelleri yükle"""
        model_path = model_name or self.config.model_file
        full_path = os.path.abspath(model_path)
        
        try:
            if os.path.exists(full_path):
                with open(full_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.ensemble_models = model_data.get('models', {})
                self.feature_selector = model_data.get('feature_selector')
                self.scaler = model_data.get('scaler')
                print(f"✅ Ensemble model yüklendi: {full_path}")
            else:
                print(f"ℹ️ Model dosyası bulunamadı: {full_path}")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)}")
    
    def load_prediction_history(self):
        """Tahmin geçmişini yükle"""
        try:
            if os.path.exists(self.config.history_file):
                with open(self.config.history_file, 'r') as f:
                    self.prediction_history = json.load(f)
        except Exception as e:
            print(f"Tahmin geçmişi yükleme hatası: {e}")
            self.prediction_history = []

    def create_comprehensive_charts(self, df: pd.DataFrame, training_results: Dict = None, future_predictions: List[Dict] = None) -> str:
        """Comprehensive charts oluştur"""
        try:
            # Analiz sonuçlarını hesapla
            current_price = df['close'].iloc[-1]
            risk_metrics = self.calculate_advanced_risk_metrics(df['close'])
            
            # Tahmin yap (eğer model varsa)
            prediction_text = ""
            if hasattr(self, 'ensemble_models') and self.ensemble_models:
                try:
                    X, _, _ = self.create_advanced_features(df)
                    if len(X) > 0:
                        latest_features = X[-1:]
                        predicted_price = self.create_ensemble_prediction(latest_features)
                        change = predicted_price - current_price
                        change_pct = (change / current_price) * 100
                        prediction_text = f"🎯 Tahmin: ${predicted_price:,.2f} ({change_pct:+.2f}%)"
                except:
                    prediction_text = "🎯 Tahmin: Hesaplanamadı"
            
            # Analiz sonuçları metni
            analysis_text = f"""
            📊 BITCOIN ANALİZ SONUÇLARI
            
            • 💰 Mevcut Fiyat: ${current_price:,.2f}
            • {prediction_text}
            • 📉 Maksimum Drawdown: {risk_metrics['max_drawdown']:.2f}%
            • 📊 Volatilite: {risk_metrics['volatility']:.2f}%
            • ⚡ Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}
            • 🎯 VaR (95%): {risk_metrics['var_95']:.2f}%
            • 📈 Toplam Getiri: {risk_metrics['total_return']:.2f}%
            • 🎲 Kazanma Oranı: {risk_metrics['win_rate']:.2f}%
            • 🟢 En İyi Gün: {risk_metrics['best_day']:.2f}%
            • 🔴 En Kötü Gün: {risk_metrics['worst_day']:.2f}%
            """
            
            # Subplot'ları oluştur
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Bitcoin Fiyat Grafiği', 'Teknik İndikatörler',
                    'Volatilite ve Volume', 'Risk Metrikleri',
                    'Günlük Getiriler', 'Tahmin vs Gerçek'
                ),
                specs=[
                    [{"secondary_y": True}, {"secondary_y": False}],
                    [{"secondary_y": True}, {"type": "bar"}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # 1. Bitcoin Fiyat Grafiği
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='BTC Fiyat'
                ),
                row=1, col=1
            )
            
            # Moving averages ekle
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'ema_12' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ema_12'],
                        mode='lines',
                        name='EMA 12',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            # 2. Teknik İndikatörler (RSI)
            if 'rsi' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['rsi'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=1, col=2
                )
                
                # RSI seviyelerini ekle
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
            
            # 3. Volume
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Volatilite (eğer varsa)
            if 'volatility' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['volatility'],
                        mode='lines',
                        name='Volatilite',
                        line=dict(color='red'),
                        yaxis='y2'
                    ),
                    row=2, col=1, secondary_y=True
                )
            
            # 4. Risk Metrikleri (Bar chart)
            risk_metrics = self.calculate_advanced_risk_metrics(df['close'])
            risk_names = ['Sharpe Ratio', 'Max Drawdown', 'Volatilite', 'Win Rate']
            risk_values = [
                risk_metrics.get('sharpe_ratio', 0),
                risk_metrics.get('max_drawdown', 0),
                risk_metrics.get('volatility', 0) / 10,  # Scale down for visibility
                risk_metrics.get('win_rate', 0)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=risk_names,
                    y=risk_values,
                    name='Risk Metrikleri',
                    marker_color=['green', 'red', 'orange', 'blue']
                ),
                row=2, col=2
            )
            
            # 5. Günlük Getiriler
            returns = df['close'].pct_change().dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns * 100,
                    name='Günlük Getiriler (%)',
                    nbinsx=50,
                    marker_color='lightgreen',
                    opacity=0.7
                ),
                row=3, col=1
            )
            
            # 6. Tahmin vs Gerçek (eğer training_results varsa)
            if training_results and 'predictions' in training_results:
                predictions = training_results['predictions']
                actual = training_results.get('actual', [])
                
                if len(predictions) == len(actual):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(actual))),
                            y=actual,
                            mode='lines',
                            name='Gerçek',
                            line=dict(color='blue')
                        ),
                        row=3, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(predictions))),
                            y=predictions,
                            mode='lines',
                            name='Tahmin',
                            line=dict(color='red', dash='dash')
                        ),
                        row=3, col=2
                    )
            
            # Layout ayarları
            fig.update_layout(
                title={
                    'text': 'Bitcoin Comprehensive Analysis Dashboard',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=1200,
                showlegend=True,
                template='plotly_white'
            )
            
            # Analiz sonuçlarını HTML olarak ekle
            analysis_html = f"""
            <div style="margin: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #f9f9f9;">
                <h2 style="color: #333; text-align: center; margin-bottom: 20px;">📊 Bitcoin Analiz Sonuçları</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="color: #2E86AB; margin-bottom: 10px;">💰 Fiyat Bilgileri</h3>
                        <p><strong>Mevcut BTC Fiyatı:</strong> ${current_price:,.2f}</p>
                        {f'<p><strong>Tahmin Edilen Fiyat:</strong> ${predicted_price:,.2f}</p>' if predicted_price else ''}
                        {f'<p><strong>Beklenen Değişim:</strong> ${predicted_price - current_price:,.2f} ({((predicted_price - current_price) / current_price * 100):+.2f}%)</p>' if predicted_price else ''}
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="color: #A23B72; margin-bottom: 10px;">📉 Risk Metrikleri</h3>
                        <p><strong>Maksimum Drawdown:</strong> {risk_metrics.get('max_drawdown', 0):.2f}%</p>
                        <p><strong>Volatilite:</strong> {risk_metrics.get('volatility', 0):.2f}%</p>
                        <p><strong>Sharpe Ratio:</strong> {risk_metrics.get('sharpe_ratio', 0):.3f}</p>
                        <p><strong>VaR (95%):</strong> {risk_metrics.get('var_95', 0):.2f}%</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="color: #F18F01; margin-bottom: 10px;">📈 Performans</h3>
                        <p><strong>Toplam Getiri:</strong> {risk_metrics.get('total_return', 0):.2f}%</p>
                        <p><strong>Kazanma Oranı:</strong> {risk_metrics.get('win_rate', 0):.2f}%</p>
                        <p><strong>En İyi Gün:</strong> {risk_metrics.get('best_day', 0):.2f}%</p>
                        <p><strong>En Kötü Gün:</strong> {risk_metrics.get('worst_day', 0):.2f}%</p>
                    </div>
                </div>
            </div>
            """
            
            # Y-axis etiketleri
            fig.update_yaxes(title_text="Fiyat ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=1, col=2)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="Değer", row=2, col=2)
            fig.update_yaxes(title_text="Frekans", row=3, col=1)
            fig.update_yaxes(title_text="Fiyat ($)", row=3, col=2)
            
            # X-axis etiketleri
            fig.update_xaxes(title_text="Tarih", row=3, col=1)
            fig.update_xaxes(title_text="Tarih", row=3, col=2)
            
            # HTML dosyasını analiz sonuçlarıyla birlikte kaydet
            html_content = fig.to_html(include_plotlyjs=True)
            
            # Analiz sonuçlarını HTML'in başına ekle
            html_with_analysis = html_content.replace(
                '<body>',
                f'<body>{analysis_html}'
            )
            
            output_file = "btc_daily_pnl_improved.html"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_with_analysis)
            
            print(f"📊 Comprehensive charts kaydedildi: {os.path.abspath(output_file)}")
            return output_file
            
        except Exception as e:
            print(f"❌ Chart oluşturma hatası: {str(e)}")
            return None

if __name__ == "__main__":
    # Configure logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('btc_analysis.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Bitcoin Daily PnL Analysis...")
    
    try:
        # Initialize analyzer with error handling
        analyzer = AdvancedBTCAnalyzer()
        logger.info("Analyzer initialized successfully")
        
        # Fetch and validate data
        logger.info("📊 Fetching data from Binance API...")
        df = analyzer.fetch_data()
        
        # Add technical indicators
        df = analyzer._add_advanced_technical_indicators(df)
        logger.info(f"✅ Successfully processed {len(df)} days of data")
        
        # Outlier detection and feature engineering
        logger.info("🔍 Performing outlier detection...")
        df = analyzer._detect_advanced_outliers(df)
        
        logger.info("⚙️ Creating advanced features...")
        X, y, feature_names = analyzer.create_advanced_features(df)
        logger.info(f"✅ Created {len(y)} samples with {len(feature_names)} features")
        
        # Feature sayısı kontrolü ve model uyumluluğu
        model_compatible = False
        if hasattr(analyzer, 'ensemble_models') and analyzer.ensemble_models:
            try:
                # Test için küçük bir örnek ile kontrol et
                test_features = X[-1:] if len(X) > 0 else None
                if test_features is not None:
                    _ = analyzer.create_ensemble_prediction(test_features)
                    model_compatible = True
                    print("🤖 Önceden eğitilmiş modeller uyumlu")
            except Exception as e:
                print(f"⚠️ Model uyumsuzluğu tespit edildi: {str(e)}")
                print("🔄 Yeni model eğitiliyor...")
                model_compatible = False
        
        if model_compatible and len(X) > 0:
            # Model uyumlu, tahmin yap
            latest_features = X[-1:]
            prediction = analyzer.create_ensemble_prediction(latest_features)
            
            current_price = df['close'].iloc[-1]
            predicted_price = prediction
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            
            print("\n" + "="*50)
            print("📈 BITCOIN ANALİZ SONUÇLARI")
            print("="*50)
            print(f"💰 Mevcut BTC Fiyatı: ${current_price:,.2f}")
            print(f"🎯 Tahmin Edilen Fiyat: ${predicted_price:,.2f}")
            print(f"📊 Beklenen Değişim: ${change:,.2f} ({change_pct:+.2f}%)")
            
            # Risk analizi
            print("\n🔍 RİSK ANALİZİ:")
            print("-" * 30)
            
            # Fiyat verilerini kullanarak risk metriklerini hesapla
            risk_metrics = analyzer.calculate_advanced_risk_metrics(df['close'])
            
            print(f"📉 Maksimum Drawdown: {risk_metrics['max_drawdown']:.2f}%")
            print(f"📊 Volatilite: {risk_metrics['volatility']:.2f}%")
            print(f"⚡ Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
            print(f"🎯 VaR (95%): {risk_metrics['var_95']:.2f}%")
            print(f"📈 Toplam Getiri: {risk_metrics['total_return']:.2f}%")
            print(f"🎲 Kazanma Oranı: {risk_metrics['win_rate']:.2f}%")
            print(f"🟢 En İyi Gün: {risk_metrics['best_day']:.2f}%")
            print(f"🔴 En Kötü Gün: {risk_metrics['worst_day']:.2f}%")
            
            # Grafikleri oluştur
            print("\n📊 Grafikler oluşturuluyor...")
            # Mevcut modelden training_results oluştur
            if len(X) > 10:
                latest_X = X[-20:] if len(X) >= 20 else X
                latest_y = y[-20:] if len(y) >= 20 else y
                predictions = []
                for i in range(len(latest_X)):
                    pred = analyzer.create_ensemble_prediction(latest_X[i:i+1])
                    predictions.append(pred)
                
                training_results = {
                    'predictions': predictions,
                    'actual': latest_y.tolist()
                }
                chart_file = analyzer.create_comprehensive_charts(df, training_results)
            else:
                chart_file = analyzer.create_comprehensive_charts(df)
            if chart_file:
                print(f"✅ Grafikler başarıyla oluşturuldu: {chart_file}")
             
        else:
            print("🔄 Model eğitiliyor...")
            
            if len(X) > 10:  # Minimum veri kontrolü
                # Model eğit
                training_results = analyzer.train_ensemble_models(X, y)
                
                # Modeli kaydet
                analyzer.save_ensemble_model(training_results['trained_models'])
                
                print("✅ Model eğitimi tamamlandı ve kaydedildi")
                
                # Yeni eğitilen model ile tahmin yap
                if len(X) > 0:
                    latest_features = X[-1:]
                    prediction = analyzer.create_ensemble_prediction(latest_features)
                    
                    current_price = df['close'].iloc[-1]
                    predicted_price = prediction
                    change = predicted_price - current_price
                    change_pct = (change / current_price) * 100
                    
                    print("\n" + "="*50)
                    print("📈 BITCOIN ANALİZ SONUÇLARI")
                    print("="*50)
                    print(f"💰 Mevcut BTC Fiyatı: ${current_price:,.2f}")
                    print(f"🎯 Tahmin Edilen Fiyat: ${predicted_price:,.2f}")
                    print(f"📊 Beklenen Değişim: ${change:,.2f} ({change_pct:+.2f}%)")
                    
                    # Risk analizi
                    print("\n🔍 RİSK ANALİZİ:")
                    print("-" * 30)
                    
                    risk_metrics = analyzer.calculate_advanced_risk_metrics(df['close'])
                    
                    print(f"📉 Maksimum Drawdown: {risk_metrics['max_drawdown']:.2f}%")
                    print(f"📊 Volatilite: {risk_metrics['volatility']:.2f}%")
                    print(f"⚡ Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
                    print(f"🎯 VaR (95%): {risk_metrics['var_95']:.2f}%")
                    print(f"📈 Toplam Getiri: {risk_metrics['total_return']:.2f}%")
                    print(f"🎲 Kazanma Oranı: {risk_metrics['win_rate']:.2f}%")
                    print(f"🟢 En İyi Gün: {risk_metrics['best_day']:.2f}%")
                    print(f"🔴 En Kötü Gün: {risk_metrics['worst_day']:.2f}%")
                    
                    # Grafikleri oluştur (training_results ile)
                    print("\n📊 Grafikler oluşturuluyor...")
                    chart_file = analyzer.create_comprehensive_charts(df, training_results)
                    if chart_file:
                        print(f"✅ Grafikler başarıyla oluşturuldu: {chart_file}")
                        # Otomatik olarak tarayıcıda aç
                        try:
                            import webbrowser
                            webbrowser.open(chart_file)
                            print("🌐 HTML raporu otomatik olarak açılıyor...")
                        except Exception as e:
                            print(f"⚠️ Tarayıcı açılamadı: {e}")
                            print(f"📁 Dosya konumu: {chart_file}")
            else:
                print("❌ Yetersiz veri - model eğitilemedi")
                
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()