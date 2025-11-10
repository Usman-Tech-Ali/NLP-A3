"""
Adaptive Ensemble Rebalancer

Dynamically adjusts model weights based on recent performance.
Implements performance-based model selection and weighting.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pymongo import DESCENDING


class AdaptiveEnsemble:
    """Dynamic ensemble with performance-based weighting"""
    
    def __init__(self, db):
        """
        Initialize adaptive ensemble
        
        Args:
            db: Database instance
        """
        self.db = db
        self.weights_collection = db.db['ensemble_weights']
        self.performance_collection = db.db['performance_history']
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create database indexes"""
        self.weights_collection.create_index([
            ('symbol', 1),
            ('timestamp', -1)
        ])
    
    def calculate_inverse_error_weights(self, errors: Dict[str, float],
                                       min_weight: float = 0.05) -> Dict[str, float]:
        """
        Calculate weights based on inverse errors
        
        Args:
            errors: Dictionary of model_name -> error (MAPE)
            min_weight: Minimum weight threshold
        
        Returns:
            Dictionary of normalized weights
        """
        # Calculate inverse weights
        weights = {}
        for model_name, error in errors.items():
            # Use inverse of error (add small epsilon to avoid division by zero)
            weights[model_name] = 1.0 / (error + 1e-6)
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Apply minimum threshold
        weights = {k: max(v, min_weight) for k, v in weights.items()}
        
        # Re-normalize after applying minimum
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def get_recent_errors(self, symbol: str, lookback_days: int = 7) -> Dict[str, float]:
        """
        Get recent MAPE for each model
        
        Args:
            symbol: Stock/Crypto symbol
            lookback_days: Number of days to look back
        
        Returns:
            Dictionary of model_name -> MAPE
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        # Get all predictions in the lookback window
        records = list(self.performance_collection.find({
            'symbol': symbol,
            'timestamp': {'$gte': cutoff_date}
        }))
        
        # Group by model
        model_errors = {}
        for record in records:
            model_name = record['model_name']
            if model_name not in model_errors:
                model_errors[model_name] = []
            model_errors[model_name].append(record['percentage_error'])
        
        # Calculate average MAPE for each model
        model_mape = {}
        for model_name, errors in model_errors.items():
            model_mape[model_name] = np.mean(errors)
        
        return model_mape
    
    def rebalance_weights(self, symbol: str, lookback_days: int = 7,
                         min_weight: float = 0.05) -> Dict[str, float]:
        """
        Rebalance ensemble weights based on recent performance
        
        Args:
            symbol: Stock/Crypto symbol
            lookback_days: Number of days to consider
            min_weight: Minimum weight for any model
        
        Returns:
            Dictionary of new weights
        """
        print(f"⚖️  Rebalancing ensemble weights for {symbol}")
        
        # Get recent errors
        recent_errors = self.get_recent_errors(symbol, lookback_days)
        
        if not recent_errors:
            print(f"  ⚠️  No recent performance data, using equal weights")
            # Default equal weights
            models = ['lstm', 'gru', 'arima', 'ma', 'ensemble']
            weights = {m: 1.0/len(models) for m in models}
        else:
            # Calculate inverse-error weights
            weights = self.calculate_inverse_error_weights(recent_errors, min_weight)
        
        # Store new weights
        weight_doc = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'weights': weights,
            'recent_errors': recent_errors,
            'lookback_days': lookback_days
        }
        
        self.weights_collection.insert_one(weight_doc)
        
        # Print weight distribution
        print(f"  New weights:")
        for model_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            error = recent_errors.get(model_name, 0)
            print(f"    {model_name:12s}: {weight:5.1%} (MAPE: {error:.2f}%)")
        
        return weights
    
    def get_current_weights(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current ensemble weights
        
        Args:
            symbol: Stock/Crypto symbol
        
        Returns:
            Dictionary of weights or None
        """
        doc = self.weights_collection.find_one(
            {'symbol': symbol},
            sort=[('timestamp', DESCENDING)]
        )
        
        return doc['weights'] if doc else None
    
    def predict_with_ensemble(self, symbol: str, predictions: Dict[str, float]) -> float:
        """
        Make weighted ensemble prediction
        
        Args:
            symbol: Stock/Crypto symbol
            predictions: Dictionary of model_name -> prediction
        
        Returns:
            Weighted ensemble prediction
        """
        # Get current weights
        weights = self.get_current_weights(symbol)
        
        if weights is None:
            # Use equal weights if no weights stored
            weights = {k: 1.0/len(predictions) for k in predictions.keys()}
        
        # Calculate weighted prediction
        ensemble_pred = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            ensemble_pred += weight * prediction
            total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def remove_poor_models(self, symbol: str, performance_threshold: float = 10.0) -> List[str]:
        """
        Identify models that should be removed from ensemble
        
        Args:
            symbol: Stock/Crypto symbol
            performance_threshold: MAPE threshold for removal
        
        Returns:
            List of model names to remove
        """
        recent_errors = self.get_recent_errors(symbol, lookback_days=7)
        
        poor_models = []
        for model_name, mape in recent_errors.items():
            if mape > performance_threshold:
                poor_models.append(model_name)
                print(f"  ⚠️  {model_name} performing poorly (MAPE: {mape:.2f}%)")
        
        return poor_models
    
    def get_weight_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """
        Get weight history over time
        
        Args:
            symbol: Stock/Crypto symbol
            days: Number of days to look back
        
        Returns:
            List of weight documents
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        history = list(self.weights_collection.find(
            {
                'symbol': symbol,
                'timestamp': {'$gte': cutoff_date}
            },
            sort=[('timestamp', 1)]
        ))
        
        return history
    
    def analyze_weight_stability(self, symbol: str, days: int = 30) -> Dict:
        """
        Analyze how stable ensemble weights are over time
        
        Args:
            symbol: Stock/Crypto symbol
            days: Number of days to analyze
        
        Returns:
            Stability analysis dictionary
        """
        history = self.get_weight_history(symbol, days)
        
        if len(history) < 2:
            return {'status': 'insufficient_data'}
        
        # Extract weights for each model over time
        model_weights = {}
        for doc in history:
            for model_name, weight in doc['weights'].items():
                if model_name not in model_weights:
                    model_weights[model_name] = []
                model_weights[model_name].append(weight)
        
        # Calculate statistics
        stability = {}
        for model_name, weights in model_weights.items():
            weights_array = np.array(weights)
            stability[model_name] = {
                'mean': float(np.mean(weights_array)),
                'std': float(np.std(weights_array)),
                'min': float(np.min(weights_array)),
                'max': float(np.max(weights_array)),
                'coefficient_of_variation': float(np.std(weights_array) / (np.mean(weights_array) + 1e-8))
            }
        
        return {
            'symbol': symbol,
            'period_days': days,
            'rebalance_count': len(history),
            'model_stability': stability
        }
    
    def auto_rebalance_if_needed(self, symbol: str, 
                                 rebalance_interval_hours: int = 24) -> bool:
        """
        Automatically rebalance if enough time has passed
        
        Args:
            symbol: Stock/Crypto symbol
            rebalance_interval_hours: Hours between rebalances
        
        Returns:
            True if rebalanced, False otherwise
        """
        # Get last rebalance time
        last_rebalance = self.weights_collection.find_one(
            {'symbol': symbol},
            sort=[('timestamp', DESCENDING)]
        )
        
        if last_rebalance is None:
            # Never rebalanced, do it now
            self.rebalance_weights(symbol)
            return True
        
        # Check if enough time has passed
        hours_since_rebalance = (datetime.utcnow() - last_rebalance['timestamp']).total_seconds() / 3600
        
        if hours_since_rebalance >= rebalance_interval_hours:
            self.rebalance_weights(symbol)
            return True
        
        return False
