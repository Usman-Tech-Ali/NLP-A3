from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import List, Dict, Optional
import os

class Database:
    """MongoDB database handler for storing historical data and predictions"""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize database connection
        
        Args:
            connection_string: MongoDB connection string (defaults to localhost)
        """
        if connection_string is None:
            connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        
        self.client = MongoClient(connection_string)
        self.db = self.client['stock_forecasting']
        
        # Collections
        self.historical_data = self.db['historical_data']
        self.predictions = self.db['predictions']
        self.metadata = self.db['metadata']
        self.models = self.db['models']
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for efficient querying"""
        self.historical_data.create_index([('symbol', ASCENDING), ('date', DESCENDING)])
        self.predictions.create_index([('symbol', ASCENDING), ('created_at', DESCENDING)])
        self.metadata.create_index([('symbol', ASCENDING)])
        self.models.create_index([('symbol', ASCENDING), ('model_name', ASCENDING), ('created_at', DESCENDING)])
    
    def store_historical_data(self, symbol: str, data: List[Dict]):
        """
        Store historical OHLCV data
        
        Args:
            symbol: Stock/Crypto symbol
            data: List of OHLCV records
        """
        for record in data:
            record['symbol'] = symbol
            self.historical_data.update_one(
                {'symbol': symbol, 'date': record['date']},
                {'$set': record},
                upsert=True
            )
    
    def get_historical_data(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """
        Retrieve historical data for a symbol
        
        Args:
            symbol: Stock/Crypto symbol
            limit: Maximum number of records to return
        
        Returns:
            List of historical records
        """
        cursor = self.historical_data.find(
            {'symbol': symbol}
        ).sort('date', DESCENDING).limit(limit)
        
        return list(cursor)
    
    def store_prediction(self, symbol: str, model_name: str, 
                        forecast_horizon: str, predictions: List[Dict],
                        metrics: Dict):
        """
        Store model predictions
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the forecasting model
            forecast_horizon: Prediction horizon (e.g., '1h', '24h')
            predictions: List of predicted values with timestamps
            metrics: Model performance metrics
        """
        prediction_doc = {
            'symbol': symbol,
            'model_name': model_name,
            'forecast_horizon': forecast_horizon,
            'predictions': predictions,
            'metrics': metrics,
            'created_at': datetime.utcnow()
        }
        
        self.predictions.insert_one(prediction_doc)
    
    def get_latest_prediction(self, symbol: str, model_name: str, 
                             forecast_horizon: str) -> Optional[Dict]:
        """
        Get the most recent prediction for a symbol
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the forecasting model
            forecast_horizon: Prediction horizon
        
        Returns:
            Latest prediction document or None
        """
        return self.predictions.find_one(
            {
                'symbol': symbol,
                'model_name': model_name,
                'forecast_horizon': forecast_horizon
            },
            sort=[('created_at', DESCENDING)]
        )
    
    def store_metadata(self, symbol: str, info: Dict):
        """
        Store symbol metadata
        
        Args:
            symbol: Stock/Crypto symbol
            info: Symbol information
        """
        info['symbol'] = symbol
        info['updated_at'] = datetime.utcnow()
        
        self.metadata.update_one(
            {'symbol': symbol},
            {'$set': info},
            upsert=True
        )
    
    def get_metadata(self, symbol: str) -> Optional[Dict]:
        """Get metadata for a symbol"""
        return self.metadata.find_one({'symbol': symbol})
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all tracked symbols"""
        return self.metadata.distinct('symbol')
    
    def store_model(self, symbol: str, model_name: str, model_state: bytes, 
                   scaler_state: bytes, config: Dict):
        """
        Store trained model state
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model (lstm, gru, etc.)
            model_state: Serialized model state dict
            scaler_state: Serialized scaler state
            config: Model configuration (hidden_size, num_layers, etc.)
        """
        import bson
        
        model_doc = {
            'symbol': symbol,
            'model_name': model_name,
            'model_state': bson.Binary(model_state),
            'scaler_state': bson.Binary(scaler_state),
            'config': config,
            'created_at': datetime.utcnow()
        }
        
        # Remove old models for this symbol/model_name combination
        self.models.delete_many({
            'symbol': symbol,
            'model_name': model_name
        })
        
        self.models.insert_one(model_doc)
    
    def get_model(self, symbol: str, model_name: str) -> Optional[Dict]:
        """
        Retrieve stored model
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
        
        Returns:
            Model document with state dicts or None
        """
        return self.models.find_one(
            {
                'symbol': symbol,
                'model_name': model_name
            },
            sort=[('created_at', DESCENDING)]
        )
    
    def delete_model(self, symbol: str, model_name: str):
        """Delete stored model"""
        self.models.delete_many({
            'symbol': symbol,
            'model_name': model_name
        })
    
    def close(self):
        """Close database connection"""
        self.client.close()
