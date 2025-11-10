import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from backend.models.traditional import TraditionalForecaster
from backend.models.neural import NeuralForecaster

@pytest.fixture
def sample_data():
    """Generate sample time series data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    return pd.Series(prices, index=dates)

def test_moving_average_forecast(sample_data):
    """Test moving average forecasting"""
    forecaster = TraditionalForecaster()
    predictions, metrics = forecaster.moving_average_forecast(sample_data, window=7, steps=10)
    
    assert len(predictions) == 10
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics
    assert metrics['model'] == 'MA_7'

def test_arima_forecast(sample_data):
    """Test ARIMA forecasting"""
    forecaster = TraditionalForecaster()
    predictions, metrics = forecaster.arima_forecast(sample_data, order=(5, 1, 0), steps=10)
    
    assert len(predictions) == 10
    assert 'rmse' in metrics
    assert 'model' in metrics

def test_ensemble_forecast(sample_data):
    """Test ensemble forecasting"""
    forecaster = TraditionalForecaster()
    predictions, metrics = forecaster.ensemble_forecast(sample_data, steps=10)
    
    assert len(predictions) == 10
    assert metrics['model'] == 'ENSEMBLE_TRADITIONAL'
    assert 'component_models' in metrics

def test_lstm_forecast(sample_data):
    """Test LSTM forecasting"""
    forecaster = NeuralForecaster()
    predictions, metrics = forecaster.lstm_forecast(sample_data, steps=10, epochs=5)
    
    assert len(predictions) == 10
    assert metrics['model'] == 'LSTM'
    assert 'rmse' in metrics

def test_gru_forecast(sample_data):
    """Test GRU forecasting"""
    forecaster = NeuralForecaster()
    predictions, metrics = forecaster.gru_forecast(sample_data, steps=10, epochs=5)
    
    assert len(predictions) == 10
    assert metrics['model'] == 'GRU'
