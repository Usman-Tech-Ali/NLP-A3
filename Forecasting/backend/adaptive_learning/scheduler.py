"""
Retraining Scheduler

Manages automated model retraining based on schedules and triggers.
Coordinates adaptive learning across all models.
"""

import schedule
import time
import threading
from datetime import datetime
from typing import Dict, List, Callable, Optional
from .performance_tracker import PerformanceTracker
from .rolling_window_trainer import RollingWindowTrainer
from .ensemble_rebalancer import AdaptiveEnsemble
from .model_versioning import ModelVersionManager


class RetrainingScheduler:
    """Automated retraining scheduler"""
    
    def __init__(self, db, data_fetcher):
        """
        Initialize scheduler
        
        Args:
            db: Database instance
            data_fetcher: DataFetcher instance
        """
        self.db = db
        self.data_fetcher = data_fetcher
        
        # Initialize components
        self.performance_tracker = PerformanceTracker(db)
        self.rolling_trainer = RollingWindowTrainer(db, data_fetcher)
        self.ensemble_rebalancer = AdaptiveEnsemble(db)
        self.version_manager = ModelVersionManager(db)
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread = None
        self.monitored_symbols = []
        
        # Callbacks
        self.on_retrain_start = None
        self.on_retrain_complete = None
        self.on_rebalance = None
    
    def add_symbol(self, symbol: str):
        """Add symbol to monitoring list"""
        if symbol not in self.monitored_symbols:
            self.monitored_symbols.append(symbol)
            print(f"[MONITOR] Added {symbol} to monitoring")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from monitoring list"""
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
            print(f"🗑️  Removed {symbol} from monitoring")
    
    def check_and_retrain(self, symbol: str, model_name: str):
        """
        Check if retraining is needed and execute if necessary
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
        """
        # Check if retraining needed
        should_retrain, reason = self.performance_tracker.should_retrain(symbol, model_name)
        
        if not should_retrain:
            return
        
        print(f"\n🔄 Retraining triggered for {symbol}/{model_name}")
        print(f"   Reason: {reason}")
        
        # Callback
        if self.on_retrain_start:
            self.on_retrain_start(symbol, model_name, reason)
        
        try:
            # Retrain based on model type
            if model_name == 'lstm':
                model, scaler, metrics = self.rolling_trainer.retrain_lstm(
                    symbol, epochs=10, use_transfer_learning=True
                )
                
                # Save new version
                config = {
                    'input_size': 1,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'seq_length': 60
                }
                
                version = self.version_manager.save_version(
                    symbol, model_name, model, scaler, config, metrics,
                    update_type='patch'
                )
                
                # Log training event
                self.performance_tracker.log_training_event(
                    symbol, model_name, version, reason,
                    data_points=len(self.rolling_trainer.fetch_recent_data(symbol, 365)),
                    epochs=10,
                    final_loss=metrics.get('final_loss', 0.0),
                    metrics=metrics,
                    status='success'
                )
                
            elif model_name == 'gru':
                model, scaler, metrics = self.rolling_trainer.retrain_gru(
                    symbol, epochs=10, use_transfer_learning=True
                )
                
                # Save new version
                config = {
                    'input_size': 1,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'seq_length': 60
                }
                
                version = self.version_manager.save_version(
                    symbol, model_name, model, scaler, config, metrics,
                    update_type='patch'
                )
                
                # Log training event
                self.performance_tracker.log_training_event(
                    symbol, model_name, version, reason,
                    data_points=len(self.rolling_trainer.fetch_recent_data(symbol, 365)),
                    epochs=10,
                    final_loss=metrics.get('final_loss', 0.0),
                    metrics=metrics,
                    status='success'
                )
            
            print(f"[OK] Retraining complete for {symbol}/{model_name}")
            
            # Callback
            if self.on_retrain_complete:
                self.on_retrain_complete(symbol, model_name, metrics)
                
        except Exception as e:
            print(f"❌ Retraining failed for {symbol}/{model_name}: {e}")
            
            # Log failure
            self.performance_tracker.log_training_event(
                symbol, model_name, 'unknown', reason,
                data_points=0, epochs=0, final_loss=0.0,
                metrics={}, status='failed'
            )
    
    def check_all_models(self, symbol: str):
        """Check and retrain all models for a symbol"""
        models = ['lstm', 'gru']  # Neural models that support adaptive learning
        
        for model_name in models:
            try:
                self.check_and_retrain(symbol, model_name)
            except Exception as e:
                print(f"Error checking {model_name}: {e}")
    
    def rebalance_ensemble(self, symbol: str):
        """Rebalance ensemble weights for a symbol"""
        try:
            print(f"\n⚖️  Rebalancing ensemble for {symbol}")
            weights = self.ensemble_rebalancer.rebalance_weights(symbol)
            
            # Callback
            if self.on_rebalance:
                self.on_rebalance(symbol, weights)
                
        except Exception as e:
            print(f"Error rebalancing ensemble: {e}")
    
    def daily_check(self):
        """Daily check for all monitored symbols"""
        print(f"\n[DAILY] Daily check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for symbol in self.monitored_symbols:
            print(f"\n--- Checking {symbol} ---")
            
            # Check models
            self.check_all_models(symbol)
            
            # Rebalance ensemble
            self.rebalance_ensemble(symbol)
    
    def hourly_check(self):
        """Hourly light check"""
        print(f"\n⏰ Hourly check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for symbol in self.monitored_symbols:
            # Only rebalance ensemble (lightweight operation)
            if self.ensemble_rebalancer.auto_rebalance_if_needed(symbol, rebalance_interval_hours=24):
                print(f"  Rebalanced {symbol}")
    
    def setup_schedules(self):
        """Setup scheduled tasks"""
        # Daily full check at 2 AM
        schedule.every().day.at("02:00").do(self.daily_check)
        
        # Hourly light check
        schedule.every().hour.do(self.hourly_check)
        
        print("[SCHEDULE] Schedules configured:")
        print("   - Daily full check: 02:00")
        print("   - Hourly ensemble rebalance")
    
    def run_scheduler(self):
        """Run scheduler in background thread"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self, symbols: Optional[List[str]] = None):
        """
        Start the scheduler
        
        Args:
            symbols: List of symbols to monitor (optional)
        """
        if self.is_running:
            print("[WARNING] Scheduler already running")
            return
        
        # Add symbols
        if symbols:
            for symbol in symbols:
                self.add_symbol(symbol)
        
        # Setup schedules
        self.setup_schedules()
        
        # Start scheduler thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("[OK] Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            print("[WARNING] Scheduler not running")
            return
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        print("🛑 Scheduler stopped")
    
    def trigger_manual_retrain(self, symbol: str, model_name: str):
        """
        Manually trigger retraining
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model
        """
        print(f"\n🔧 Manual retrain triggered for {symbol}/{model_name}")
        self.check_and_retrain(symbol, model_name)
    
    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            'is_running': self.is_running,
            'monitored_symbols': self.monitored_symbols,
            'scheduled_jobs': len(schedule.jobs),
            'next_run': str(schedule.next_run()) if schedule.jobs else None
        }
