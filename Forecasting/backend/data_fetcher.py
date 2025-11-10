import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List

class DataFetcher:
    """Fetch and process financial data using yfinance"""
    
    def __init__(self):
        self.supported_intervals = ['1h', '1d', '1wk']
    
    def fetch_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a given symbol with multiple fallback mechanisms
        
        Args:
            symbol: Stock/Crypto symbol (e.g., 'AAPL', 'BTC-USD')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        # Try different periods in order of preference
        periods_to_try = [period, '6mo', '3mo', '1mo', '5d']
        
        for attempt_period in periods_to_try:
            try:
                print(f"Attempting to fetch {symbol} with period={attempt_period}...")
                
                # Method 1: Try with download (more reliable)
                df = yf.download(symbol, period=attempt_period, interval=interval, 
                               progress=False, show_errors=False)
                
                if not df.empty:
                    print(f"✓ Successfully fetched {len(df)} records for {symbol}")
                    
                    # Reset index to make Date a column
                    df.reset_index(inplace=True)
                    
                    # Rename columns to lowercase
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Ensure we have the required columns
                    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if not all(col in df.columns for col in required_cols):
                        print(f"Missing required columns, trying next period...")
                        continue
                    
                    return df
                else:
                    print(f"Empty data for period={attempt_period}, trying next...")
                    
            except Exception as e:
                print(f"Error with period={attempt_period}: {str(e)[:100]}")
                continue
        
        # Final fallback: try Ticker.history with original period
        try:
            print(f"Final attempt: using Ticker.history for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                df.reset_index(inplace=True)
                df.columns = [col.lower() for col in df.columns]
                print(f"✓ Ticker.history succeeded with {len(df)} records")
                return df
        except Exception as e:
            print(f"Ticker.history also failed: {str(e)[:100]}")
        
        print(f"✗ All attempts failed for {symbol}")
        return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except Exception as e:
            print(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def prepare_for_model(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        Prepare data for ML models
        
        Args:
            df: Raw OHLCV dataframe
            target_col: Column to predict
        
        Returns:
            Processed dataframe
        """
        # Create a copy
        data = df.copy()
        
        # Add technical indicators
        data['returns'] = data[target_col].pct_change()
        data['log_returns'] = np.log(data[target_col] / data[target_col].shift(1))
        
        # Moving averages
        data['ma_7'] = data[target_col].rolling(window=7).mean()
        data['ma_21'] = data[target_col].rolling(window=21).mean()
        data['ma_50'] = data[target_col].rolling(window=50).mean()
        
        # Volatility
        data['volatility'] = data['returns'].rolling(window=21).std()
        
        # RSI
        delta = data[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get information about a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown')
            }
        except Exception as e:
            print(f"Error getting info for {symbol}: {e}")
            return {'symbol': symbol, 'name': symbol, 'currency': 'USD', 'exchange': 'Unknown'}

import numpy as np
