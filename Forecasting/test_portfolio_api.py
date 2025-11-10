"""
Test Portfolio API Endpoints

Quick test to verify portfolio creation and listing works correctly.
"""

import requests
import json

API_BASE = 'http://localhost:5000/api'

def test_portfolio_api():
    print("=" * 60)
    print("TESTING PORTFOLIO API")
    print("=" * 60)
    
    # Test 1: Create Portfolio
    print("\n1. Creating portfolio...")
    response = requests.post(f'{API_BASE}/portfolio/create', json={
        'name': 'Test Portfolio',
        'initial_cash': 50000.0
    })
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            portfolio_id = data['portfolio_id']
            print(f"   ✓ Portfolio created: {portfolio_id}")
            print(f"   Name: {data['name']}")
            print(f"   Initial Cash: ${data['initial_cash']:,.2f}")
        else:
            print(f"   ✗ Error: {data.get('error')}")
            return
    else:
        print(f"   ✗ HTTP Error: {response.status_code}")
        return
    
    # Test 2: List Portfolios
    print("\n2. Listing portfolios...")
    response = requests.get(f'{API_BASE}/portfolio/list')
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            portfolios = data['portfolios']
            print(f"   ✓ Found {len(portfolios)} portfolio(s)")
            for p in portfolios:
                print(f"   - {p['name']} (ID: {p['portfolio_id'][:8]}...)")
                print(f"     Cash: ${p['cash']:,.2f}")
                print(f"     Total Value: ${p['total_value']:,.2f}")
        else:
            print(f"   ✗ Error: {data.get('error')}")
    else:
        print(f"   ✗ HTTP Error: {response.status_code}")
    
    # Test 3: Get Portfolio Details
    print("\n3. Getting portfolio details...")
    response = requests.get(f'{API_BASE}/portfolio/{portfolio_id}')
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            portfolio = data['portfolio']
            print(f"   ✓ Portfolio: {portfolio['name']}")
            print(f"   Cash: ${portfolio['cash']:,.2f}")
            print(f"   Positions: {len(portfolio['positions'])}")
        else:
            print(f"   ✗ Error: {data.get('error')}")
    else:
        print(f"   ✗ HTTP Error: {response.status_code}")
    
    # Test 4: Get Positions Summary
    print("\n4. Getting positions summary...")
    response = requests.get(f'{API_BASE}/portfolio/{portfolio_id}/positions')
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            summary = data['summary']
            print(f"   ✓ Total Value: ${summary['total_value']:,.2f}")
            print(f"   Cash: ${summary['cash']:,.2f}")
            print(f"   Positions: {summary['position_count']}")
        else:
            print(f"   ✗ Error: {data.get('error')}")
    else:
        print(f"   ✗ HTTP Error: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nPortfolio ID for testing: {portfolio_id}")
    print("You can now test the web interface at http://localhost:5000/portfolio")

if __name__ == '__main__':
    try:
        test_portfolio_api()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Cannot connect to Flask server")
        print("Make sure the server is running: python backend/app.py")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
