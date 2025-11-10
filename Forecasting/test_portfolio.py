"""
Test Portfolio Management System

This script tests the portfolio management functionality including:
- Portfolio creation
- Trade execution
- Signal generation
- Performance tracking
- Risk management
"""

import sys
import time
from backend.database import Database
from backend.data_fetcher import DataFetcher
from backend.portfolio import (
    PortfolioManager,
    TradingStrategy,
    RiskManager,
    PerformanceMetrics
)


def test_portfolio_system():
    """Test the complete portfolio management system"""
    
    print("=" * 60)
    print("PORTFOLIO MANAGEMENT SYSTEM TEST")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    db = Database()
    data_fetcher = DataFetcher()
    portfolio_manager = PortfolioManager(db, data_fetcher, initial_cash=100000.0)
    trading_strategy = TradingStrategy(db, portfolio_manager)
    risk_manager = RiskManager(db, portfolio_manager)
    performance_metrics = PerformanceMetrics(db, portfolio_manager)
    print("✅ Components initialized")
    
    # Create portfolio
    print("\n2. Creating portfolio...")
    portfolio_id = portfolio_manager.create_portfolio()
    print(f"✅ Portfolio created: {portfolio_id}")
    
    # Get portfolio state
    print("\n3. Getting portfolio state...")
    portfolio = portfolio_manager.get_portfolio(portfolio_id)
    print(f"   Cash: ${portfolio['cash']:,.2f}")
    print(f"   Total Value: ${portfolio['total_value']:,.2f}")
    print(f"   Positions: {len(portfolio['positions'])}")
    
    # Test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Execute test trades
    print("\n4. Testing trade execution...")
    for symbol in test_symbols:
        print(f"\n   Testing {symbol}...")
        
        # Get current price
        current_price = portfolio_manager.get_current_price(symbol)
        print(f"   Current price: ${current_price:.2f}")
        
        # Simulate prediction (5% upside)
        predicted_price = current_price * 1.05
        confidence = 0.75
        
        # Generate signal
        signal = trading_strategy.generate_signal(
            symbol, current_price, predicted_price,
            confidence, 'test_model', portfolio_id
        )
        
        print(f"   Signal: {signal['action'].upper()}")
        print(f"   Reason: {signal['reason']}")
        
        if signal['action'] == 'buy':
            shares = signal.get('shares', 0)
            print(f"   Shares to buy: {shares}")
            
            # Validate trade
            validation = risk_manager.validate_trade(
                portfolio_id, symbol, 'buy', shares, current_price
            )
            
            print(f"   Risk validation: {'✅ Approved' if validation['approved'] else '❌ Rejected'}")
            print(f"   Risk score: {validation['risk_score']:.2f}")
            
            if validation['approved']:
                # Execute trade
                result = portfolio_manager.execute_trade(
                    portfolio_id, symbol, 'buy', shares,
                    'test', 'test_model', predicted_price, confidence
                )
                
                if result['success']:
                    print(f"   ✅ Trade executed successfully")
                    print(f"   Total cost: ${result['total_cost']:,.2f}")
                    print(f"   Remaining cash: ${result['remaining_cash']:,.2f}")
                else:
                    print(f"   ❌ Trade failed: {result.get('error')}")
        
        time.sleep(1)  # Rate limiting
    
    # Get updated portfolio
    print("\n5. Updated portfolio state...")
    positions_summary = portfolio_manager.get_positions_summary(portfolio_id)
    print(f"   Total Value: ${positions_summary['total_value']:,.2f}")
    print(f"   Cash: ${positions_summary['cash']:,.2f}")
    print(f"   Positions Value: ${positions_summary['positions_value']:,.2f}")
    print(f"   Number of Positions: {positions_summary['position_count']}")
    print(f"   Cash Percentage: {positions_summary['cash_percentage']:.1f}%")
    
    # Show positions
    print("\n6. Current positions:")
    for symbol, position in positions_summary['positions'].items():
        print(f"   {symbol}:")
        print(f"      Shares: {position['shares']}")
        print(f"      Avg Price: ${position['avg_price']:.2f}")
        print(f"      Current Value: ${position['current_value']:,.2f}")
        print(f"      Unrealized P&L: ${position['unrealized_pnl']:,.2f}")
    
    # Calculate performance
    print("\n7. Performance metrics...")
    performance = performance_metrics.calculate_daily_performance(portfolio_id)
    print(f"   Total Value: ${performance['total_value']:,.2f}")
    print(f"   Daily Return: {performance['daily_return']:.2%}")
    print(f"   Cumulative Return: {performance['cumulative_return']:.2%}")
    print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   Volatility: {performance['volatility']:.2%}")
    
    # Risk dashboard
    print("\n8. Risk dashboard...")
    risk_dashboard = risk_manager.get_risk_dashboard(portfolio_id)
    print(f"   Risk Score: {risk_dashboard['risk_score']:.0f}/100")
    print(f"   Risk Level: {risk_dashboard['risk_level']}")
    print(f"   Cash Percentage: {risk_dashboard['cash_percentage']:.1f}%")
    print(f"   Max Position: {risk_dashboard['max_position_percentage']:.1f}%")
    print(f"   Stop Loss Alerts: {risk_dashboard['stop_loss_alerts']}")
    
    # Trade history
    print("\n9. Trade history...")
    trades = portfolio_manager.get_trade_history(portfolio_id, limit=10)
    print(f"   Total trades: {len(trades)}")
    for i, trade in enumerate(trades[:5], 1):
        print(f"   {i}. {trade['action'].upper()} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")
    
    # Test sell trade
    print("\n10. Testing sell trade...")
    if positions_summary['positions']:
        # Sell first position
        symbol = list(positions_summary['positions'].keys())[0]
        position = positions_summary['positions'][symbol]
        shares_to_sell = position['shares'] // 2  # Sell half
        
        if shares_to_sell > 0:
            print(f"   Selling {shares_to_sell} shares of {symbol}...")
            
            current_price = portfolio_manager.get_current_price(symbol)
            
            # Validate
            validation = risk_manager.validate_trade(
                portfolio_id, symbol, 'sell', shares_to_sell, current_price
            )
            
            if validation['approved']:
                result = portfolio_manager.execute_trade(
                    portfolio_id, symbol, 'sell', shares_to_sell,
                    'test', 'test_model', None, None
                )
                
                if result['success']:
                    print(f"   ✅ Sell executed")
                    print(f"   Proceeds: ${result['total_proceeds']:,.2f}")
                    print(f"   Realized P&L: ${result['realized_pnl']:,.2f}")
    
    # Final summary
    print("\n11. Final portfolio summary...")
    final_summary = performance_metrics.get_performance_summary(portfolio_id, days=1)
    print(f"   Total Value: ${final_summary['total_value']:,.2f}")
    print(f"   Cumulative Return: {final_summary['cumulative_return']:.2%}")
    print(f"   Number of Positions: {final_summary['number_of_positions']}")
    print(f"   Total Trades: {final_summary['total_trades']}")
    print(f"   Win Rate: {final_summary['win_rate']:.1%}")
    
    print("\n" + "=" * 60)
    print("✅ PORTFOLIO SYSTEM TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return portfolio_id


if __name__ == '__main__':
    try:
        portfolio_id = test_portfolio_system()
        print(f"\nPortfolio ID: {portfolio_id}")
        print("You can use this ID to access the portfolio in the web interface.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
