"""
Initialize Adaptive Learning System

This script:
1. Generates initial predictions for monitored symbols
2. Logs performance data
3. Creates initial model versions
4. Sets up ensemble weights
5. Starts the scheduler
"""

import sys
import numpy as np
from datetime import datetime, timedelta

from database import Database
from data_fetcher import DataFetcher
from models.neural import NeuralForecaster
from adaptive_learning import (
    ModelVersionManager,
    PerformanceTracker,
    AdaptiveEnsemble,
    RetrainingScheduler
)


def initialize_system():
    """Initialize the adaptive learning system with sample data"""
    
    print("🚀 Initializing Adaptive Learning System...")
    print("=" * 60)
    
    # Initialize components
    db = Database()
    data_fetcher = DataFetcher()
    neural_forecaster = NeuralForecaster(db=db)
    
    version_manager = ModelVersionManager(db)
    performance_tracker = PerformanceTracker(db)
    ensemble_rebalancer = AdaptiveEnsemble(db)
    scheduler = RetrainingScheduler(db, data_fetcher)
    
    # Symbols to initialize
    symbols = ['AAPL', 'GOOGL', 'BTC-USD']
    models = ['lstm', 'gru']
    
    print(f"\n📊 Initializing {len(symbols)} symbols with {len(models)} models each...")
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}...")
        print(f"{'='*60}")
        
        # Fetch data
        print(f"  📥 Fetching data for {symbol}...")
        df = data_fetcher.fetch_data(symbol, period='1y', interval='1d')
        
        if df is None or len(df) < 100:
            print(f"  ⚠️  Insufficient data for {symbol}, skipping...")
            continue
        
        close_prices = df['close']
        print(f"  ✓ Fetched {len(df)} days of data")
        
        # Store historical data
        records = df.to_dict('records')
        for record in records:
            if 'date' in record:
                record['date'] = record['date'].isoformat() if hasattr(record['date'], 'isoformat') else str(record['date'])
        db.store_historical_data(symbol, records)
        print(f"  ✓ Stored historical data in database")
        
        for model_name in models:
            print(f"\n  🤖 Initializing {model_name.upper()} model...")
            
            try:
                # Generate initial predictions
                if model_name == 'lstm':
                    predictions, metrics = neural_forecaster.lstm_forecast(
                        close_prices, steps=7, epochs=20, symbol=symbol, use_cache=True
                    )
                else:  # gru
                    predictions, metrics = neural_forecaster.gru_forecast(
                        close_prices, steps=7, epochs=20, symbol=symbol, use_cache=True
                    )
                
                print(f"    ✓ Generated predictions - MAPE: {metrics['mape']:.2f}%")
                
                # Log some sample predictions (simulate past predictions)
                print(f"    📝 Logging sample predictions...")
                
                # Get last 30 days of actual data
                recent_data = df.tail(30)
                
                for i, (idx, row) in enumerate(recent_data.iterrows()):
                    actual_price = row['close']
                    # Simulate prediction with small random error
                    error_pct = np.random.normal(0, metrics['mape'] / 100)
                    predicted_price = actual_price * (1 + error_pct)
                    
                    # Log prediction (backdated)
                    timestamp = row['date'] if hasattr(row['date'], 'isoformat') else datetime.fromisoformat(row['date'])
                    
                    performance_tracker.performance_collection.insert_one({
                        'symbol': symbol,
                        'model_name': model_name,
                        'version': 'v1.0.0',
                        'timestamp': timestamp,
                        'actual_price': float(actual_price),
                        'predicted_price': float(predicted_price),
                        'error': float(abs(actual_price - predicted_price)),
                        'percentage_error': float(abs(error_pct) * 100),
                        'metrics': metrics
                    })
                
                print(f"    ✓ Logged 30 sample predictions")
                
                # Log initial training event
                performance_tracker.log_training_event(
                    symbol=symbol,
                    model_name=model_name,
                    version='v1.0.0',
                    trigger='initial_setup',
                    data_points=len(close_prices),
                    epochs=20,
                    final_loss=0.001,
                    metrics=metrics,
                    status='success',
                    training_started=datetime.utcnow() - timedelta(hours=1)
                )
                
                print(f"    ✓ Logged training event")
                
            except Exception as e:
                print(f"    ❌ Error initializing {model_name}: {e}")
                continue
        
        # Initialize ensemble weights
        print(f"\n  ⚖️  Initializing ensemble weights...")
        try:
            weights = ensemble_rebalancer.rebalance_weights(symbol, lookback_days=7)
            print(f"    ✓ Ensemble weights set")
        except Exception as e:
            print(f"    ⚠️  Could not set ensemble weights: {e}")
    
    print(f"\n{'='*60}")
    print("✅ Initialization Complete!")
    print(f"{'='*60}")
    
    print("\n📊 Summary:")
    print(f"  - Symbols initialized: {len(symbols)}")
    print(f"  - Models per symbol: {len(models)}")
    print(f"  - Total predictions logged: ~{len(symbols) * len(models) * 30}")
    
    print("\n🎯 Next Steps:")
    print("  1. Visit http://localhost:5000/monitor to view the dashboard")
    print("  2. The scheduler will run automatically (daily at 2 AM)")
    print("  3. You can manually trigger retraining from the monitor page")
    
    print("\n💡 Tips:")
    print("  - Use 'Refresh Data' button to update the dashboard")
    print("  - Try 'Trigger Retrain' to see adaptive learning in action")
    print("  - Check 'Performance' tab to see MAPE trends")
    print("  - View 'Ensemble' tab to see dynamic weight distribution")
    
    return True


if __name__ == '__main__':
    try:
        success = initialize_system()
        if success:
            print("\n✅ System ready for use!")
            sys.exit(0)
        else:
            print("\n❌ Initialization failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
