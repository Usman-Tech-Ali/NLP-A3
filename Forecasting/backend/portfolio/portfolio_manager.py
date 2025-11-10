"""
Portfolio Manager

Core portfolio management functionality including position tracking,
trade execution, and portfolio state management.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from pymongo import DESCENDING


class PortfolioManager:
    """Manages portfolio state, positions, and trades"""
    
    def __init__(self, db, data_fetcher, initial_cash: float = 100000.0):
        """
        Initialize portfolio manager
        
        Args:
            db: Database instance
            data_fetcher: DataFetcher instance
            initial_cash: Starting cash amount
        """
        self.db = db
        self.data_fetcher = data_fetcher
        self.initial_cash = initial_cash
        
        # Collections
        self.portfolio_collection = db.db['portfolio_state']
        self.trades_collection = db.db['trades']
        self.performance_collection = db.db['portfolio_performance']
        
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create database indexes"""
        self.portfolio_collection.create_index('portfolio_id')
        self.trades_collection.create_index([('portfolio_id', 1), ('timestamp', -1)])
        self.performance_collection.create_index([('portfolio_id', 1), ('date', -1)])
    
    def create_portfolio(self, portfolio_id: str = None, name: str = None) -> str:
        """
        Create a new portfolio
        
        Args:
            portfolio_id: Optional portfolio ID (generates UUID if None)
            name: Optional portfolio name
        
        Returns:
            Portfolio ID
        """
        if portfolio_id is None:
            portfolio_id = str(uuid.uuid4())
        
        # Check if portfolio already exists
        existing = self.portfolio_collection.find_one({'portfolio_id': portfolio_id})
        if existing:
            return portfolio_id
        
        # Create new portfolio
        portfolio = {
            'portfolio_id': portfolio_id,
            'name': name or 'Portfolio',
            'cash': float(self.initial_cash),
            'positions': {},
            'total_value': float(self.initial_cash),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        self.portfolio_collection.insert_one(portfolio)
        
        # Log initial performance record
        self.performance_collection.insert_one({
            'portfolio_id': portfolio_id,
            'date': datetime.utcnow(),
            'total_value': float(self.initial_cash),
            'cash': float(self.initial_cash),
            'positions_value': 0.0,
            'daily_return': 0.0,
            'cumulative_return': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0
        })
        
        print(f"[OK] Created portfolio '{name or 'Portfolio'}' ({portfolio_id}) with ${self.initial_cash:,.2f}")
        return portfolio_id
    
    def get_portfolio(self, portfolio_id: str) -> Optional[Dict]:
        """
        Get portfolio state
        
        Args:
            portfolio_id: Portfolio ID
        
        Returns:
            Portfolio dictionary or None
        """
        return self.portfolio_collection.find_one({'portfolio_id': portfolio_id})
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol
        
        Args:
            symbol: Stock/crypto symbol
        
        Returns:
            Current price
        """
        try:
            df = self.data_fetcher.fetch_data(symbol, period='1d', interval='1m')
            if df is not None and len(df) > 0:
                return float(df['close'].iloc[-1])
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
        
        # Fallback: try to get from recent predictions
        try:
            recent_pred = self.db.db['performance_history'].find_one(
                {'symbol': symbol},
                sort=[('timestamp', DESCENDING)]
            )
            if recent_pred:
                return recent_pred['actual_price']
        except:
            pass
        
        return 100.0  # Default fallback price
    
    def execute_trade(self, portfolio_id: str, symbol: str, action: str, 
                     shares: int, trigger: str = 'manual', 
                     model_used: str = None, predicted_price: float = None,
                     confidence: float = None) -> Dict:
        """
        Execute a trade (buy, sell, hold)
        
        Args:
            portfolio_id: Portfolio ID
            symbol: Stock/crypto symbol
            action: 'buy', 'sell', or 'hold'
            shares: Number of shares
            trigger: What triggered the trade
            model_used: Model that generated the signal
            predicted_price: Predicted price
            confidence: Model confidence
        
        Returns:
            Trade result dictionary
        """
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {'success': False, 'error': 'Portfolio not found'}
        
        if action.lower() == 'hold':
            return {'success': True, 'action': 'hold', 'message': 'No trade executed'}
        
        # Get current price
        current_price = self.get_current_price(symbol)
        
        if action.lower() == 'buy':
            return self._execute_buy(portfolio, symbol, shares, current_price, 
                                   trigger, model_used, predicted_price, confidence)
        elif action.lower() == 'sell':
            return self._execute_sell(portfolio, symbol, shares, current_price,
                                    trigger, model_used, predicted_price, confidence)
        else:
            return {'success': False, 'error': 'Invalid action'}
    
    def _execute_buy(self, portfolio: Dict, symbol: str, shares: int, 
                    price: float, trigger: str, model_used: str,
                    predicted_price: float, confidence: float) -> Dict:
        """Execute buy order"""
        total_cost = shares * price
        
        # Check if enough cash
        if portfolio['cash'] < total_cost:
            return {
                'success': False, 
                'error': f'Insufficient cash. Need ${total_cost:,.2f}, have ${portfolio["cash"]:,.2f}'
            }
        
        # Update portfolio
        portfolio['cash'] -= total_cost
        
        if symbol in portfolio['positions']:
            # Add to existing position
            old_shares = portfolio['positions'][symbol]['shares']
            old_avg_price = portfolio['positions'][symbol]['avg_price']
            
            new_shares = old_shares + shares
            new_avg_price = ((old_shares * old_avg_price) + (shares * price)) / new_shares
            
            portfolio['positions'][symbol] = {
                'shares': new_shares,
                'avg_price': new_avg_price,
                'current_value': new_shares * price,
                'unrealized_pnl': new_shares * (price - new_avg_price)
            }
        else:
            # New position
            portfolio['positions'][symbol] = {
                'shares': shares,
                'avg_price': price,
                'current_value': shares * price,
                'unrealized_pnl': 0.0
            }
        
        # Update total value
        portfolio['total_value'] = self._calculate_total_value(portfolio)
        portfolio['updated_at'] = datetime.utcnow()
        
        # Save portfolio
        self.portfolio_collection.replace_one(
            {'portfolio_id': portfolio['portfolio_id']},
            portfolio
        )
        
        # Log trade
        trade = {
            'portfolio_id': portfolio['portfolio_id'],
            'symbol': symbol,
            'action': 'buy',
            'shares': shares,
            'price': price,
            'total_cost': total_cost,
            'timestamp': datetime.utcnow(),
            'trigger': trigger,
            'model_used': model_used,
            'predicted_price': predicted_price,
            'confidence': confidence
        }
        
        self.trades_collection.insert_one(trade)
        
        print(f"[BUY] {shares} shares of {symbol} at ${price:.2f} (Total: ${total_cost:,.2f})")
        
        return {
            'success': True,
            'action': 'buy',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'total_cost': total_cost,
            'remaining_cash': portfolio['cash'],
            'total_value': portfolio['total_value']
        }
    
    def _execute_sell(self, portfolio: Dict, symbol: str, shares: int,
                     price: float, trigger: str, model_used: str,
                     predicted_price: float, confidence: float) -> Dict:
        """Execute sell order"""
        if symbol not in portfolio['positions']:
            return {'success': False, 'error': f'No position in {symbol}'}
        
        position = portfolio['positions'][symbol]
        
        if position['shares'] < shares:
            return {
                'success': False,
                'error': f'Insufficient shares. Have {position["shares"]}, trying to sell {shares}'
            }
        
        total_proceeds = shares * price
        
        # Calculate realized P&L
        realized_pnl = shares * (price - position['avg_price'])
        
        # Update portfolio
        portfolio['cash'] += total_proceeds
        
        if position['shares'] == shares:
            # Selling entire position
            del portfolio['positions'][symbol]
        else:
            # Partial sell
            remaining_shares = position['shares'] - shares
            portfolio['positions'][symbol] = {
                'shares': remaining_shares,
                'avg_price': position['avg_price'],
                'current_value': remaining_shares * price,
                'unrealized_pnl': remaining_shares * (price - position['avg_price'])
            }
        
        # Update total value
        portfolio['total_value'] = self._calculate_total_value(portfolio)
        portfolio['updated_at'] = datetime.utcnow()
        
        # Save portfolio
        self.portfolio_collection.replace_one(
            {'portfolio_id': portfolio['portfolio_id']},
            portfolio
        )
        
        # Log trade
        trade = {
            'portfolio_id': portfolio['portfolio_id'],
            'symbol': symbol,
            'action': 'sell',
            'shares': shares,
            'price': price,
            'total_proceeds': total_proceeds,
            'realized_pnl': realized_pnl,
            'timestamp': datetime.utcnow(),
            'trigger': trigger,
            'model_used': model_used,
            'predicted_price': predicted_price,
            'confidence': confidence
        }
        
        self.trades_collection.insert_one(trade)
        
        print(f"[SELL] {shares} shares of {symbol} at ${price:.2f} (Total: ${total_proceeds:,.2f}, P&L: ${realized_pnl:,.2f})")
        
        return {
            'success': True,
            'action': 'sell',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'total_proceeds': total_proceeds,
            'realized_pnl': realized_pnl,
            'remaining_cash': portfolio['cash'],
            'total_value': portfolio['total_value']
        }
    
    def _calculate_total_value(self, portfolio: Dict) -> float:
        """Calculate total portfolio value"""
        total = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            current_price = self.get_current_price(symbol)
            position_value = position['shares'] * current_price
            total += position_value
            
            # Update position current value
            position['current_value'] = position_value
            position['unrealized_pnl'] = position['shares'] * (current_price - position['avg_price'])
        
        return total
    
    def update_portfolio_values(self, portfolio_id: str) -> Dict:
        """Update portfolio with current market prices"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return None
        
        # Update all position values
        portfolio['total_value'] = self._calculate_total_value(portfolio)
        portfolio['updated_at'] = datetime.utcnow()
        
        # Save updated portfolio
        self.portfolio_collection.replace_one(
            {'portfolio_id': portfolio_id},
            portfolio
        )
        
        return portfolio
    
    def get_trade_history(self, portfolio_id: str, limit: int = 50) -> List[Dict]:
        """Get trade history for portfolio"""
        trades = list(self.trades_collection.find(
            {'portfolio_id': portfolio_id},
            sort=[('timestamp', DESCENDING)],
            limit=limit
        ))
        
        # Convert ObjectId to string
        for trade in trades:
            trade['_id'] = str(trade['_id'])
            if 'timestamp' in trade:
                trade['timestamp'] = trade['timestamp'].isoformat()
        
        return trades
    
    def get_positions_summary(self, portfolio_id: str) -> Dict:
        """Get summary of current positions"""
        portfolio = self.update_portfolio_values(portfolio_id)
        if not portfolio:
            return {}
        
        positions_value = sum(pos['current_value'] for pos in portfolio['positions'].values())
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in portfolio['positions'].values())
        
        return {
            'total_value': portfolio['total_value'],
            'cash': portfolio['cash'],
            'positions_value': positions_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions': portfolio['positions'],
            'position_count': len(portfolio['positions']),
            'cash_percentage': (portfolio['cash'] / portfolio['total_value']) * 100 if portfolio['total_value'] > 0 else 0
        }
