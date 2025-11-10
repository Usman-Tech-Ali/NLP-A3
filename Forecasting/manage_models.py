"""
Utility script to manage trained models in MongoDB
"""
import sys
sys.path.append('backend')

from database import Database
from datetime import datetime

def list_models():
    """List all stored models"""
    db = Database()
    models = list(db.models.find())
    
    if not models:
        print("No models found in database.")
        return
    
    print("\n" + "="*80)
    print("STORED MODELS")
    print("="*80)
    print(f"{'Symbol':<15} {'Model':<15} {'Created':<25} {'Config'}")
    print("-"*80)
    
    for model in models:
        symbol = model.get('symbol', 'N/A')
        model_name = model.get('model_name', 'N/A')
        created = model.get('created_at', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        config = model.get('config', {})
        config_str = f"hidden={config.get('hidden_size', 'N/A')}, layers={config.get('num_layers', 'N/A')}"
        
        print(f"{symbol:<15} {model_name:<15} {created:<25} {config_str}")
    
    print("="*80)
    print(f"Total models: {len(models)}\n")
    db.close()

def delete_model(symbol: str, model_name: str):
    """Delete a specific model"""
    db = Database()
    
    # Check if model exists
    model = db.get_model(symbol, model_name)
    if model is None:
        print(f"Model not found: {model_name} for {symbol}")
        db.close()
        return
    
    # Delete
    db.delete_model(symbol, model_name)
    print(f"✓ Deleted model: {model_name} for {symbol}")
    db.close()

def delete_all_models():
    """Delete all stored models"""
    db = Database()
    
    count = db.models.count_documents({})
    if count == 0:
        print("No models to delete.")
        db.close()
        return
    
    confirm = input(f"Are you sure you want to delete all {count} models? (yes/no): ")
    if confirm.lower() == 'yes':
        db.models.delete_many({})
        print(f"✓ Deleted all {count} models")
    else:
        print("Cancelled")
    
    db.close()

def model_info(symbol: str, model_name: str):
    """Show detailed information about a model"""
    db = Database()
    
    model = db.get_model(symbol, model_name)
    if model is None:
        print(f"Model not found: {model_name} for {symbol}")
        db.close()
        return
    
    print("\n" + "="*80)
    print(f"MODEL INFORMATION: {model_name.upper()} for {symbol}")
    print("="*80)
    print(f"Created: {model.get('created_at', 'N/A')}")
    print(f"\nConfiguration:")
    config = model.get('config', {})
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nModel state size: {len(model.get('model_state', b''))} bytes")
    print(f"Scaler state size: {len(model.get('scaler_state', b''))} bytes")
    print("="*80 + "\n")
    
    db.close()

def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_models.py list                    - List all models")
        print("  python manage_models.py info <symbol> <model>   - Show model details")
        print("  python manage_models.py delete <symbol> <model> - Delete a model")
        print("  python manage_models.py delete-all              - Delete all models")
        print("\nExamples:")
        print("  python manage_models.py list")
        print("  python manage_models.py info AAPL lstm")
        print("  python manage_models.py delete AAPL lstm")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_models()
    
    elif command == 'info':
        if len(sys.argv) < 4:
            print("Error: Please provide symbol and model name")
            print("Usage: python manage_models.py info <symbol> <model>")
            return
        symbol = sys.argv[2]
        model_name = sys.argv[3]
        model_info(symbol, model_name)
    
    elif command == 'delete':
        if len(sys.argv) < 4:
            print("Error: Please provide symbol and model name")
            print("Usage: python manage_models.py delete <symbol> <model>")
            return
        symbol = sys.argv[2]
        model_name = sys.argv[3]
        delete_model(symbol, model_name)
    
    elif command == 'delete-all':
        delete_all_models()
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'python manage_models.py' to see available commands")

if __name__ == '__main__':
    main()
