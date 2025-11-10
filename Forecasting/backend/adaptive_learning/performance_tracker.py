"""
Performance Tracker

Tracks model performance over time, detects degradation, and triggers retraining.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pymongo import DESCENDING


class PerformanceTracker:
    """Track and analyze model performance over time"""
    
    def __init__(self, db):
        """
        Initialize performance tracker
        
        Args:
            db: Database instance
        """
        self.db = db
        self.performance_collection = db.db['performance_history']
        self.training_logs = db.db['training_logs']
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create database indexes"""
        self.performance_collection.create_index([
            ('symbol', 1),
            ('model_name', 1),
            ('timestamp', -1)
        ])
        self.training_logs.create_index([
            ('symbol', 1),
            ('model_name', 1),
            ('training_started', -1)
        ])
    
    def log_prediction(self, symbol: str, model_name: str, version: str,
                      actual_price: float, predicted_price: float,
                      metrics: Optional[Dict] = None):
        """
        Log a single prediction for performance tracking
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            version: Model version
            actual_price: Actual price
            predicted_price: Predicted price
            metrics: Optional additional metrics
        """
        error = abs(actual_price - predicted_price)
        percentage_error = (error / actual_price) * 100 if actual_price != 0 else 0
        
        doc = {
            'symbol': symbol,
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.utcnow(),
            'actual_price': float(actual_price),
            'predicted_price': float(predicted_price),
            'error': float(error),
            'percentage_error': float(percentage_error),
            'metrics': metrics or {}
        }
        
        self.performance_collection.insert_one(doc)
    
    def log_training_event(self, symbol: str, model_name: str, version: str,
                          trigger: str, data_points: int, epochs: int,
                          final_loss: float, metrics: Dict, 
                          status: str = 'success',
                          training_started: Optional[datetime] = None):
        """
        Log a model training event
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            version: Model version
            trigger: What triggered training (scheduled, performance_drop, manual)
            data_points: Number of training data points
            epochs: Number of training epochs
            final_loss: Final training loss
            metrics: Performance metrics
            status: Training status (success, failed, in_progress)
            training_started: When training started (None for now)
        """
        if training_started is None:
            training_started = datetime.utcnow()
        
        doc = {
            'symbol': symbol,
            'model_name': model_name,
            'version': version,
            'training_started': training_started,
            'training_completed': datetime.utcnow(),
            'trigger': trigger,
            'data_points': data_points,
            'epochs': epochs,
            'final_loss': float(final_loss),
            'metrics': metrics,
            'status': status
        }
        
        self.training_logs.insert_one(doc)
        print(f"📝 Logged training event: {model_name} {version} ({trigger})")
    
    def get_recent_performance(self, symbol: str, model_name: str,
                              days: int = 7) -> Dict:
        """
        Get recent performance metrics
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            days: Number of days to look back
        
        Returns:
            Dictionary with aggregated metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        records = list(self.performance_collection.find({
            'symbol': symbol,
            'model_name': model_name,
            'timestamp': {'$gte': cutoff_date}
        }))
        
        if not records:
            return {
                'count': 0,
                'rmse': None,
                'mae': None,
                'mape': None
            }
        
        errors = [r['error'] for r in records]
        percentage_errors = [r['percentage_error'] for r in records]
        
        return {
            'count': len(records),
            'rmse': float(np.sqrt(np.mean(np.array(errors) ** 2))),
            'mae': float(np.mean(errors)),
            'mape': float(np.mean(percentage_errors))
        }
    
    def get_baseline_performance(self, symbol: str, model_name: str) -> Dict:
        """
        Get baseline performance (first 30 days after initial training)
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
        
        Returns:
            Baseline metrics dictionary
        """
        # Get first training date
        first_training = self.training_logs.find_one(
            {'symbol': symbol, 'model_name': model_name},
            sort=[('training_started', 1)]
        )
        
        if not first_training:
            return {'rmse': None, 'mae': None, 'mape': None}
        
        # Get performance in first 30 days
        start_date = first_training['training_started']
        end_date = start_date + timedelta(days=30)
        
        records = list(self.performance_collection.find({
            'symbol': symbol,
            'model_name': model_name,
            'timestamp': {'$gte': start_date, '$lte': end_date}
        }))
        
        if not records:
            # Use training metrics as baseline
            return first_training.get('metrics', {})
        
        errors = [r['error'] for r in records]
        percentage_errors = [r['percentage_error'] for r in records]
        
        return {
            'rmse': float(np.sqrt(np.mean(np.array(errors) ** 2))),
            'mae': float(np.mean(errors)),
            'mape': float(np.mean(percentage_errors))
        }
    
    def detect_performance_degradation(self, symbol: str, model_name: str,
                                      threshold: float = 1.2) -> Tuple[bool, Dict]:
        """
        Detect if model performance has degraded
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            threshold: Degradation threshold (1.2 = 20% worse)
        
        Returns:
            (is_degraded, details) tuple
        """
        baseline = self.get_baseline_performance(symbol, model_name)
        recent = self.get_recent_performance(symbol, model_name, days=7)
        
        if baseline['mape'] is None or recent['mape'] is None:
            return False, {'reason': 'insufficient_data'}
        
        # Check if recent performance is significantly worse
        degradation_ratio = recent['mape'] / baseline['mape']
        
        is_degraded = degradation_ratio > threshold
        
        details = {
            'baseline_mape': baseline['mape'],
            'recent_mape': recent['mape'],
            'degradation_ratio': degradation_ratio,
            'threshold': threshold,
            'is_degraded': is_degraded
        }
        
        if is_degraded:
            print(f"⚠️  Performance degradation detected for {model_name} on {symbol}")
            print(f"   Baseline MAPE: {baseline['mape']:.2f}%")
            print(f"   Recent MAPE: {recent['mape']:.2f}%")
            print(f"   Degradation: {(degradation_ratio - 1) * 100:.1f}%")
        
        return is_degraded, details
    
    def count_consecutive_failures(self, symbol: str, model_name: str,
                                  failure_threshold: float = 5.0) -> int:
        """
        Count consecutive predictions with high error
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            failure_threshold: MAPE threshold for failure (default 5%)
        
        Returns:
            Number of consecutive failures
        """
        # Get last 10 predictions
        records = list(self.performance_collection.find(
            {'symbol': symbol, 'model_name': model_name},
            sort=[('timestamp', DESCENDING)],
            limit=10
        ))
        
        consecutive_failures = 0
        for record in records:
            if record['percentage_error'] > failure_threshold:
                consecutive_failures += 1
            else:
                break
        
        return consecutive_failures
    
    def get_performance_trend(self, symbol: str, model_name: str,
                             days: int = 30) -> List[Dict]:
        """
        Get performance trend over time
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            days: Number of days to analyze
        
        Returns:
            List of daily aggregated metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        records = list(self.performance_collection.find({
            'symbol': symbol,
            'model_name': model_name,
            'timestamp': {'$gte': cutoff_date}
        }, sort=[('timestamp', 1)]))
        
        # Group by day
        daily_metrics = {}
        for record in records:
            date_key = record['timestamp'].date()
            
            if date_key not in daily_metrics:
                daily_metrics[date_key] = {
                    'errors': [],
                    'percentage_errors': []
                }
            
            daily_metrics[date_key]['errors'].append(record['error'])
            daily_metrics[date_key]['percentage_errors'].append(record['percentage_error'])
        
        # Calculate daily aggregates
        trend = []
        for date_key in sorted(daily_metrics.keys()):
            metrics = daily_metrics[date_key]
            errors = np.array(metrics['errors'])
            percentage_errors = np.array(metrics['percentage_errors'])
            
            trend.append({
                'date': date_key.isoformat(),
                'count': len(errors),
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'mae': float(np.mean(errors)),
                'mape': float(np.mean(percentage_errors))
            })
        
        return trend
    
    def get_training_history(self, symbol: str, model_name: str,
                            limit: int = 20) -> List[Dict]:
        """
        Get training history
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
            limit: Maximum number of records
        
        Returns:
            List of training events
        """
        logs = list(self.training_logs.find(
            {'symbol': symbol, 'model_name': model_name},
            sort=[('training_started', DESCENDING)],
            limit=limit
        ))
        
        return logs
    
    def should_retrain(self, symbol: str, model_name: str) -> Tuple[bool, str]:
        """
        Determine if model should be retrained
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
        
        Returns:
            (should_retrain, reason) tuple
        """
        # Check 1: Performance degradation
        is_degraded, details = self.detect_performance_degradation(symbol, model_name)
        if is_degraded:
            return True, 'performance_degradation'
        
        # Check 2: Consecutive failures
        failures = self.count_consecutive_failures(symbol, model_name)
        if failures >= 3:
            return True, f'consecutive_failures_{failures}'
        
        # Check 3: Time since last training
        last_training = self.training_logs.find_one(
            {'symbol': symbol, 'model_name': model_name},
            sort=[('training_started', DESCENDING)]
        )
        
        if last_training:
            days_since_training = (datetime.utcnow() - last_training['training_started']).days
            
            if days_since_training > 30:
                return True, 'scheduled_monthly'
            elif days_since_training > 7:
                # Check if weekly retrain is beneficial
                recent_perf = self.get_recent_performance(symbol, model_name, days=7)
                if recent_perf['mape'] and recent_perf['mape'] > 2.5:
                    return True, 'scheduled_weekly'
        
        return False, 'no_retrain_needed'
    
    def get_model_statistics(self, symbol: str, model_name: str) -> Dict:
        """
        Get comprehensive model statistics
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
        
        Returns:
            Statistics dictionary
        """
        # Get all-time performance
        all_records = list(self.performance_collection.find({
            'symbol': symbol,
            'model_name': model_name
        }))
        
        if not all_records:
            return {'status': 'no_data'}
        
        errors = [r['error'] for r in all_records]
        percentage_errors = [r['percentage_error'] for r in all_records]
        
        # Get training count
        training_count = self.training_logs.count_documents({
            'symbol': symbol,
            'model_name': model_name
        })
        
        # Get last training
        last_training = self.training_logs.find_one(
            {'symbol': symbol, 'model_name': model_name},
            sort=[('training_started', DESCENDING)]
        )
        
        return {
            'total_predictions': len(all_records),
            'all_time_performance': {
                'rmse': float(np.sqrt(np.mean(np.array(errors) ** 2))),
                'mae': float(np.mean(errors)),
                'mape': float(np.mean(percentage_errors))
            },
            'recent_performance': self.get_recent_performance(symbol, model_name, days=7),
            'baseline_performance': self.get_baseline_performance(symbol, model_name),
            'training_count': training_count,
            'last_training': last_training['training_started'] if last_training else None,
            'days_since_training': (datetime.utcnow() - last_training['training_started']).days if last_training else None
        }
