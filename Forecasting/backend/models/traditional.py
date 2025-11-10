import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class TraditionalForecaster:
    """Traditional time series forecasting models"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
    
    def moving_average_forecast(self, data: pd.Series, window: int = 7, 
                                steps: int = 24) -> Tuple[np.ndarray, Dict]:
        """
        Simple Moving Average forecast
        
        Args:
            data: Historical price series
            window: Moving average window size
            steps: Number of steps to forecast
        
        Returns:
            Tuple of (predictions, metrics)
        """
        self.model_name = f'MA_{window}'
        
        # Calculate moving average
        ma = data.rolling(window=window).mean()
        
        # Use last MA value as forecast (naive approach)
        last_ma = ma.iloc[-1]
        predictions = np.full(steps, last_ma)
        
        # Calculate metrics on validation set
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]
        
        if len(test) > 0:
            ma_train = train.rolling(window=window).mean()
            test_pred = np.full(len(test), ma_train.iloc[-1])
            
            rmse = np.sqrt(mean_squared_error(test, test_pred))
            mae = mean_absolute_error(test, test_pred)
            mape = np.mean(np.abs((test - test_pred) / test)) * 100
        else:
            rmse, mae, mape = 0, 0, 0
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'model': self.model_name
        }
        
        return predictions, metrics
    
    def exponential_smoothing_forecast(self, data: pd.Series, alpha: float = 0.3,
                                      steps: int = 24) -> Tuple[np.ndarray, Dict]:
        """
        Exponential Smoothing forecast
        
        Args:
            data: Historical price series
            alpha: Smoothing parameter
            steps: Number of steps to forecast
        
        Returns:
            Tuple of (predictions, metrics)
        """
        self.model_name = 'EXP_SMOOTHING'
        
        # Calculate exponential smoothing
        smoothed = data.ewm(alpha=alpha, adjust=False).mean()
        last_value = smoothed.iloc[-1]
        
        # Forecast (flat line from last smoothed value)
        predictions = np.full(steps, last_value)
        
        # Calculate metrics
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]
        
        if len(test) > 0:
            smoothed_train = train.ewm(alpha=alpha, adjust=False).mean()
            test_pred = np.full(len(test), smoothed_train.iloc[-1])
            
            rmse = np.sqrt(mean_squared_error(test, test_pred))
            mae = mean_absolute_error(test, test_pred)
            mape = np.mean(np.abs((test - test_pred) / test)) * 100
        else:
            rmse, mae, mape = 0, 0, 0
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'model': self.model_name
        }
        
        return predictions, metrics
    
    def arima_forecast(self, data: pd.Series, order: Tuple[int, int, int] = (5, 1, 0),
                      steps: int = 24) -> Tuple[np.ndarray, Dict]:
        """
        ARIMA forecast
        
        Args:
            data: Historical price series
            order: ARIMA order (p, d, q)
            steps: Number of steps to forecast
        
        Returns:
            Tuple of (predictions, metrics)
        """
        self.model_name = f'ARIMA_{order}'
        
        try:
            # Split data
            train_size = int(len(data) * 0.8)
            train, test = data[:train_size], data[train_size:]
            
            # Fit ARIMA model
            model = ARIMA(train, order=order)
            fitted_model = model.fit()
            
            # Forecast
            predictions = fitted_model.forecast(steps=steps)
            
            # Calculate metrics on test set
            if len(test) > 0:
                test_pred = fitted_model.forecast(steps=len(test))
                
                rmse = np.sqrt(mean_squared_error(test, test_pred))
                mae = mean_absolute_error(test, test_pred)
                mape = np.mean(np.abs((test - test_pred) / test)) * 100
            else:
                rmse, mae, mape = 0, 0, 0
            
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'model': self.model_name,
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic)
            }
            
            self.model = fitted_model
            
            return predictions.values, metrics
        
        except Exception as e:
            print(f"ARIMA error: {e}")
            # Fallback to simple forecast
            predictions = np.full(steps, data.iloc[-1])
            metrics = {
                'rmse': 0,
                'mae': 0,
                'mape': 0,
                'model': self.model_name,
                'error': str(e)
            }
            return predictions, metrics
    
    def ensemble_forecast(self, data: pd.Series, steps: int = 24, 
                         symbol: str = None, db=None) -> Tuple[np.ndarray, Dict]:
        """
        Ensemble of multiple traditional models with adaptive weighting
        
        Args:
            data: Historical price series
            steps: Number of steps to forecast
            symbol: Stock/crypto symbol (for adaptive weights)
            db: Database instance (for adaptive weights)
        
        Returns:
            Tuple of (predictions, metrics)
        """
        self.model_name = 'ENSEMBLE_TRADITIONAL'
        
        # Get predictions from multiple models
        ma_pred, ma_metrics = self.moving_average_forecast(data, window=7, steps=steps)
        exp_pred, exp_metrics = self.exponential_smoothing_forecast(data, alpha=0.3, steps=steps)
        arima_pred, arima_metrics = self.arima_forecast(data, order=(5, 1, 0), steps=steps)
        
        # Try to use adaptive weights if available
        weights = None
        if symbol and db:
            try:
                from backend.adaptive_learning.ensemble_rebalancer import AdaptiveEnsemble
                ensemble_rebalancer = AdaptiveEnsemble(db)
                
                # Get current weights
                weights = ensemble_rebalancer.get_current_weights(symbol)
                
                # If no weights exist, initialize with equal weights and store
                if weights is None:
                    print(f"[ENSEMBLE] Initializing weights for {symbol}")
                    from datetime import datetime
                    weights = {
                        'ma': 1.0/3.0,
                        'arima': 1.0/3.0,
                        'exp_smoothing': 1.0/3.0
                    }
                    
                    # Store initial weights
                    weight_doc = {
                        'symbol': symbol,
                        'timestamp': datetime.utcnow(),
                        'weights': weights,
                        'recent_errors': {},
                        'lookback_days': 7,
                        'note': 'Initial equal weights'
                    }
                    ensemble_rebalancer.weights_collection.insert_one(weight_doc)
                    print(f"[ENSEMBLE] Initialized equal weights for {symbol}")
                
            except Exception as e:
                print(f"[ENSEMBLE] Could not load adaptive weights: {e}")
                weights = None
        
        # Apply weights if available
        if weights:
            # Use adaptive weights
            w_ma = weights.get('ma', 1.0/3.0)
            w_arima = weights.get('arima', 1.0/3.0)
            w_exp = weights.get('exp_smoothing', 1.0/3.0)
            
            # Normalize weights
            total = w_ma + w_arima + w_exp
            w_ma, w_arima, w_exp = w_ma/total, w_arima/total, w_exp/total
            
            predictions = (ma_pred * w_ma) + (arima_pred * w_arima) + (exp_pred * w_exp)
            print(f"[ENSEMBLE] Using adaptive weights: MA={w_ma:.2%}, ARIMA={w_arima:.2%}, EXP={w_exp:.2%}")
        else:
            # Use equal weights
            predictions = (ma_pred + exp_pred + arima_pred) / 3
        
        # Calculate ensemble metrics
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]
        
        if len(test) > 0:
            # Get test predictions from each model
            ma_test, _ = self.moving_average_forecast(train, window=7, steps=len(test))
            exp_test, _ = self.exponential_smoothing_forecast(train, alpha=0.3, steps=len(test))
            arima_test, _ = self.arima_forecast(train, order=(5, 1, 0), steps=len(test))
            
            if weights:
                ensemble_test = (ma_test * w_ma) + (arima_test * w_arima) + (exp_test * w_exp)
            else:
                ensemble_test = (ma_test + exp_test + arima_test) / 3
            
            rmse = np.sqrt(mean_squared_error(test, ensemble_test))
            mae = mean_absolute_error(test, ensemble_test)
            mape = np.mean(np.abs((test - ensemble_test) / test)) * 100
        else:
            rmse, mae, mape = 0, 0, 0
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'model': self.model_name,
            'component_models': ['MA_7', 'EXP_SMOOTHING', 'ARIMA_(5, 1, 0)'],
            'weights_used': weights if weights else 'equal'
        }
        
        return predictions, metrics
