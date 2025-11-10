"""Risk Manager - Implements risk management rules and position limits"""

from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np


class RiskManager:
    """Manages portfolio risk and validates trades"""
    
    def __init__(self, db, portfolio_manager):
        self.db = db
        self.portfolio_manager = portfolio_manager
        
        # Risk limits
        self.max_position_pct = 0.10  # Max 10% per position
        self.max_positions = 5
        self.min_cash_reserve = 0.20  # Keep 20% in cash
        self.max_daily_loss_pct = 0.05  # Max 5% daily loss
        self.stop_loss_pct = 0.05  # 5% stop loss
    
    def validate_trade(self, portfolio_id: str, symbol: str, action: str,
                      shares: int, current_price: float) -> Dict:
        """Validate if a trade meets risk management criteria"""
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        if not portfolio:
            return {
                'approved': False,
                'reason': 'Portfolio not found',
                'risk_score': 1.0
            }
        
        if action.lower() == 'buy':
            return self._validate_buy_trade(portfolio, symbol, shares, current_price)
        elif action.lower() == 'sell':
            return self._validate_sell_trade(portfolio, symbol, shares, current_price)
        else:
            return {
                'approved': True,
                'reason': 'Hold action - no validation needed',
                'risk_score': 0.0
            }
    
    def _validate_buy_trade(self, portfolio: Dict, symbol: str,
                           shares: int, current_price: float) -> Dict:
        """Validate buy trade"""
        total_cost = shares * current_price
        total_value = portfolio['total_value']
        
        # Check cash availability
        required_cash_reserve = total_value * self.min_cash_reserve
        available_cash = portfolio['cash'] - required_cash_reserve
        
        if total_cost > available_cash:
            return {
                'approved': False,
                'reason': f'Insufficient cash. Need ${total_cost:,.2f}, available ${available_cash:,.2f}',
                'risk_score': 1.0
            }
        
        # Check position size limit
        if symbol in portfolio['positions']:
            current_value = portfolio['positions'][symbol]['current_value']
            new_position_value = current_value + total_cost
        else:
            new_position_value = total_cost
        
        position_pct = new_position_value / total_value
        if position_pct > self.max_position_pct:
            return {
                'approved': False,
                'reason': f'Position size {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}',
                'risk_score': 0.8
            }
        
        # Check maximum positions
        if symbol not in portfolio['positions'] and len(portfolio['positions']) >= self.max_positions:
            return {
                'approved': False,
                'reason': f'Maximum positions ({self.max_positions}) reached',
                'risk_score': 0.6
            }
        
        # Check daily loss limit
        daily_loss_check = self._check_daily_loss_limit(portfolio['portfolio_id'])
        if not daily_loss_check['approved']:
            return daily_loss_check
        
        # Calculate risk score
        risk_score = self._calculate_buy_risk_score(portfolio, symbol, shares, current_price)
        
        return {
            'approved': True,
            'reason': 'Trade approved',
            'risk_score': risk_score,
            'position_percentage': position_pct,
            'cash_after_trade': portfolio['cash'] - total_cost
        }
    
    def _validate_sell_trade(self, portfolio: Dict, symbol: str,
                            shares: int, current_price: float) -> Dict:
        """Validate sell trade"""
        if symbol not in portfolio['positions']:
            return {
                'approved': False,
                'reason': f'No position in {symbol}',
                'risk_score': 0.0
            }
        
        position = portfolio['positions'][symbol]
        
        if shares > position['shares']:
            return {
                'approved': False,
                'reason': f'Insufficient shares. Have {position["shares"]}, trying to sell {shares}',
                'risk_score': 0.0
            }
        
        risk_score = max(0, 0.5 - (shares / position['shares']) * 0.3)
        
        return {
            'approved': True,
            'reason': 'Sell trade approved',
            'risk_score': risk_score,
            'proceeds': shares * current_price
        }
    
    def _calculate_buy_risk_score(self, portfolio: Dict, symbol: str,
                                 shares: int, current_price: float) -> float:
        """Calculate risk score for buy trade (0 = low risk, 1 = high risk)"""
        total_value = portfolio['total_value']
        position_value = shares * current_price
        position_pct = position_value / total_value
        
        size_risk = position_pct / self.max_position_pct
        
        cash_after = portfolio['cash'] - position_value
        cash_pct_after = cash_after / total_value
        cash_risk = max(0, (self.min_cash_reserve - cash_pct_after) / self.min_cash_reserve)
        
        num_positions = len(portfolio['positions'])
        concentration_risk = max(0, (num_positions - 3) / self.max_positions)
        
        total_risk = (size_risk * 0.4 + cash_risk * 0.4 + concentration_risk * 0.2)
        
        return min(1.0, total_risk)
    
    def _check_daily_loss_limit(self, portfolio_id: str) -> Dict:
        """Check if portfolio has exceeded daily loss limit"""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        today_record = self.db.db['portfolio_performance'].find_one({
            'portfolio_id': portfolio_id,
            'date': today
        })
        
        if not today_record:
            return {'approved': True, 'reason': 'No daily performance record'}
        
        daily_return = today_record.get('daily_return', 0)
        
        if daily_return < -self.max_daily_loss_pct:
            return {
                'approved': False,
                'reason': f'Daily loss limit exceeded: {daily_return:.2%}',
                'risk_score': 1.0
            }
        
        return {'approved': True, 'reason': 'Daily loss within limits'}
    
    def check_stop_losses(self, portfolio_id: str) -> List[Dict]:
        """Check all positions for stop loss triggers"""
        portfolio = self.portfolio_manager.update_portfolio_values(portfolio_id)
        if not portfolio or not portfolio['positions']:
            return []
        
        stop_loss_trades = []
        
        for symbol, position in portfolio['positions'].items():
            current_price = self.portfolio_manager.get_current_price(symbol)
            avg_price = position['avg_price']
            
            loss_pct = (avg_price - current_price) / avg_price
            
            if loss_pct > self.stop_loss_pct:
                stop_loss_trades.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': position['shares'],
                    'current_price': current_price,
                    'avg_price': avg_price,
                    'loss_percentage': loss_pct,
                    'reason': f'Stop loss triggered: {loss_pct:.2%} loss',
                    'urgency': 'high'
                })
        
        return stop_loss_trades
    
    def get_risk_dashboard(self, portfolio_id: str) -> Dict:
        """Get comprehensive risk dashboard"""
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        if not portfolio:
            return {}
        
        total_value = portfolio['total_value']
        cash_pct = portfolio['cash'] / total_value if total_value > 0 else 1
        num_positions = len(portfolio['positions'])
        
        if portfolio['positions']:
            position_values = [pos['current_value'] for pos in portfolio['positions'].values()]
            max_position_value = max(position_values)
            max_position_pct = max_position_value / total_value
        else:
            max_position_pct = 0
        
        stop_losses = self.check_stop_losses(portfolio_id)
        
        risk_factors = {
            'cash_risk': max(0, (self.min_cash_reserve - cash_pct) / self.min_cash_reserve) * 20,
            'concentration_risk': (max_position_pct / self.max_position_pct) * 30,
            'diversification_risk': max(0, (3 - num_positions) / 3) * 25,
            'stop_loss_risk': len(stop_losses) * 25
        }
        
        overall_risk_score = min(100, sum(risk_factors.values()))
        
        if overall_risk_score < 30:
            risk_level = 'Low'
            risk_color = 'green'
        elif overall_risk_score < 60:
            risk_level = 'Medium'
            risk_color = 'yellow'
        else:
            risk_level = 'High'
            risk_color = 'red'
        
        return {
            'portfolio_id': portfolio_id,
            'risk_score': float(overall_risk_score),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'cash_percentage': float(cash_pct * 100),
            'max_position_percentage': float(max_position_pct * 100),
            'number_of_positions': num_positions,
            'limits': {
                'max_position_pct': self.max_position_pct * 100,
                'min_cash_reserve': self.min_cash_reserve * 100,
                'max_positions': self.max_positions,
                'stop_loss_pct': self.stop_loss_pct * 100
            },
            'stop_loss_alerts': len(stop_losses),
            'stop_loss_details': stop_losses[:3]
        }
