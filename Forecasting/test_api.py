"""
Simple script to test the API endpoints
Run this after starting the Flask server
"""
import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_home():
    """Test home page"""
    print("Testing home page...")
    response = requests.get(f'{BASE_URL}/')
    print(f"Status: {response.status_code}")
    print(f"✓ Home page accessible\n")

def test_get_symbols():
    """Test getting available symbols"""
    print("Testing /api/symbols...")
    response = requests.get(f'{BASE_URL}/api/symbols')
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Symbols: {json.dumps(data, indent=2)}")
    print(f"✓ Symbols retrieved\n")

def test_fetch_data(symbol='AAPL'):
    """Test fetching historical data"""
    print(f"Testing /api/fetch_data for {symbol}...")
    payload = {
        'symbol': symbol,
        'period': '1mo',
        'interval': '1d'
    }
    response = requests.post(
        f'{BASE_URL}/api/fetch_data',
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    data = response.json()
    print(f"Status: {response.status_code}")
    if data.get('success'):
        print(f"Records fetched: {data.get('records')}")
        print(f"✓ Data fetched successfully\n")
    else:
        print(f"✗ Error: {data.get('error')}\n")
    return data.get('success', False)

def test_forecast(symbol='AAPL', model='ensemble', horizon='7d'):
    """Test generating forecast"""
    print(f"Testing /api/forecast for {symbol} using {model}...")
    payload = {
        'symbol': symbol,
        'model': model,
        'horizon': horizon
    }
    
    start_time = time.time()
    response = requests.post(
        f'{BASE_URL}/api/forecast',
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    elapsed_time = time.time() - start_time
    
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    if data.get('success'):
        metrics = data.get('metrics', {})
        print(f"Model: {metrics.get('model')}")
        print(f"RMSE: {metrics.get('rmse', 0):.2f}")
        print(f"MAE: {metrics.get('mae', 0):.2f}")
        print(f"MAPE: {metrics.get('mape', 0):.2f}%")
        print(f"Predictions: {len(data.get('predictions', []))} points")
        print(f"✓ Forecast generated successfully\n")
    else:
        print(f"✗ Error: {data.get('error')}\n")
    
    return data.get('success', False)

def test_compare_models(symbol='AAPL', horizon='7d'):
    """Test comparing multiple models"""
    print(f"Testing /api/compare_models for {symbol}...")
    payload = {
        'symbol': symbol,
        'horizon': horizon
    }
    
    start_time = time.time()
    response = requests.post(
        f'{BASE_URL}/api/compare_models',
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    elapsed_time = time.time() - start_time
    
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    if data.get('success'):
        results = data.get('results', {})
        print(f"\nModel Comparison Results:")
        print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
        print("-" * 50)
        
        for model_name, model_data in results.items():
            metrics = model_data.get('metrics', {})
            print(f"{model_name.upper():<20} "
                  f"{metrics.get('rmse', 0):<10.2f} "
                  f"{metrics.get('mae', 0):<10.2f} "
                  f"{metrics.get('mape', 0):<10.2f}")
        
        print(f"\n✓ Model comparison completed\n")
    else:
        print(f"✗ Error: {data.get('error')}\n")
    
    return data.get('success', False)

def run_all_tests():
    """Run all API tests"""
    print("=" * 60)
    print("Stock Forecasting API Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Home page
        test_home()
        
        # Test 2: Get symbols
        test_get_symbols()
        
        # Test 3: Fetch data
        test_fetch_data('AAPL')
        
        # Test 4: Generate forecast with traditional model (fast)
        test_forecast('AAPL', 'ensemble', '7d')
        
        # Test 5: Generate forecast with neural model (slower)
        print("Note: Neural models take longer (30-60 seconds)...")
        # Uncomment to test neural models:
        # test_forecast('AAPL', 'lstm', '7d')
        
        # Test 6: Compare models
        print("Note: Model comparison takes longer as it runs all models...")
        # Uncomment to test model comparison:
        # test_compare_models('AAPL', '7d')
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to the server.")
        print("Make sure the Flask server is running on http://localhost:5000")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == '__main__':
    run_all_tests()
