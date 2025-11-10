import pytest
import sys
sys.path.append('..')

from backend.data_fetcher import DataFetcher

def test_fetch_data():
    """Test data fetching from yfinance"""
    fetcher = DataFetcher()
    df = fetcher.fetch_data('AAPL', period='5d', interval='1d')
    
    assert df is not None
    assert len(df) > 0
    assert 'close' in df.columns
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns

def test_get_latest_price():
    """Test getting latest price"""
    fetcher = DataFetcher()
    price = fetcher.get_latest_price('AAPL')
    
    assert price is not None
    assert price > 0

def test_get_symbol_info():
    """Test getting symbol information"""
    fetcher = DataFetcher()
    info = fetcher.get_symbol_info('AAPL')
    
    assert 'symbol' in info
    assert info['symbol'] == 'AAPL'
