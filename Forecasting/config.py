"""
Configuration settings for the Stock Forecasting Application
"""
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Database settings
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = 'stock_forecasting'
    
    # Data fetching settings
    DEFAULT_PERIOD = '1y'
    DEFAULT_INTERVAL = '1d'
    MAX_HISTORICAL_RECORDS = 1000
    
    # Model settings
    LSTM_EPOCHS = 30
    GRU_EPOCHS = 30
    LSTM_HIDDEN_SIZE = 64
    GRU_HIDDEN_SIZE = 64
    SEQUENCE_LENGTH = 60
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Forecast settings
    FORECAST_HORIZONS = {
        '1h': 1,
        '3h': 3,
        '24h': 24,
        '72h': 72,
        '1d': 1,
        '3d': 3,
        '7d': 7
    }
    
    # Popular symbols
    POPULAR_SYMBOLS = {
        'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'],
        'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD'],
        'forex': ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'AUDUSD=X']
    }
    
    # Cache settings
    CACHE_PREDICTIONS = True
    CACHE_DURATION = timedelta(hours=1)
    
    # API rate limiting (requests per minute)
    RATE_LIMIT = 60

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Override with production values
    SECRET_KEY = os.getenv('SECRET_KEY')
    MONGODB_URI = os.getenv('MONGODB_URI')

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_NAME = 'stock_forecasting_test'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
