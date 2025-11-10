"""
Populate Ensemble Weights

This script manually triggers ensemble weight calculation for all symbols
that have prediction history. Run this to populate the ensemble_weights collection.
"""

from backend.database import Database
from backend.adaptive_learning.ensemble_rebalancer import AdaptiveEnsemble

def populate_ensemble_weights():
    """Populate ensemble weights for all symbols with predictions"""
    
    print("=" * 60)
    print("POPULATING ENSEMBLE WEIGHTS")
    print("=" * 60)
    
    # Initialize
    db = Database()
    ensemble = AdaptiveEnsemble(db)
    
    # Get all symbols that have predictions
    symbols = db.db['predictions'].distinct('symbol')
    
    if not symbols:
        print("\n⚠️  No predictions found in database")
        print("   Generate some forecasts first, then run this script")
        return
    
    print(f"\nFound {len(symbols)} symbol(s) with predictions")
    
    # Rebalance for each symbol
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Check if there's performance history
            perf_count = db.db['performance_history'].count_documents({'symbol': symbol})
            
            if perf_count == 0:
                print(f"  ⚠️  No performance history for {symbol}")
                print(f"     Predictions need to be evaluated against actual prices")
                print(f"     This happens automatically over time")
                
                # Create default equal weights
                print(f"  Creating default equal weights...")
                models = ['lstm', 'gru', 'arima', 'ma', 'ensemble']
                weights = {m: 1.0/len(models) for m in models}
                
                from datetime import datetime
                weight_doc = {
                    'symbol': symbol,
                    'timestamp': datetime.utcnow(),
                    'weights': weights,
                    'recent_errors': {},
                    'lookback_days': 7,
                    'note': 'Default equal weights (no performance history yet)'
                }
                
                ensemble.weights_collection.insert_one(weight_doc)
                print(f"  ✅ Created default weights")
                
            else:
                print(f"  Found {perf_count} performance records")
                
                # Rebalance based on actual performance
                weights = ensemble.rebalance_weights(symbol, lookback_days=7)
                print(f"  ✅ Rebalanced based on performance")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Show summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_weights = ensemble.weights_collection.count_documents({})
    print(f"Total ensemble weight records: {total_weights}")
    
    # Show latest weights for each symbol
    for symbol in symbols:
        latest = ensemble.get_current_weights(symbol)
        if latest:
            print(f"\n{symbol}:")
            for model, weight in sorted(latest.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model:12s}: {weight:5.1%}")
    
    print(f"\n{'='*60}")
    print("✅ ENSEMBLE WEIGHTS POPULATED")
    print(f"{'='*60}")
    print("\nYou can now:")
    print("1. View weights in MongoDB: db.ensemble_weights.find()")
    print("2. Check the Adaptive Monitor page")
    print("3. Weights will auto-update hourly via scheduler")

if __name__ == '__main__':
    try:
        populate_ensemble_weights()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
