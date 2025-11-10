"""Trading Strategy - Generates trading signals based on model predictions"""

import numpy as np
from typing import Dict
from datetime import datetime, timedelta


class TradingStrategy:
    """Generates trading signals based on model predictions"""
    
    def __init__(self, db, portfolio_manager):
        self.db = db
        self.portfolio_manager = portfolio_manager
        
        # Strategy parameters
        self.buy_threshold = 0.02  # 2% upside needed for buy
        self.sell_threshold = -0.02  # 2% downside triggers sell
        self.min_confidence = 0.6  # Minimum model confidence
        self.max_position_pct = 0.10  # Max 10% of portfolio per position
        self.max_positions = 5
        self.min_cash_reserve = 0.20  # Keep 20% in cash
    
    def generate_signal(self, symbol: str, current_price: float, 
                       predicted_price: float, confidence: float,
                       model_name: str, portfolio_id: str) -> Dict:
        """Generate trading signal based on prediction"""
        expected_return = (predicted_price - current_price) / current_price
        
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        if not portfolio:
            return {'action': 'hold', 'reason': 'Portfolio not found'}
        
        if confidence < self.min_confidence:
            return {
                'action': 'hold',
                'reason': f'Low confidence: {confidence:.2f}',
                'expected_return': expected_return,
                'confidence': confidence
            }
        
        if expected_return > self.buy_threshold:
            return self._generate_buy_signal(symbol, current_price, expected_return, 
                                           confidence, portfolio)
        elif expected_return < self.sell_threshold:
            return self._generate_sell_signal(symbol, current_price, expected_return,
                                            confidence, portfolio)
        else:
            return {
                'action': 'hold',
                'reason': f'Expected return {expected_return:.2%} within hold range',
                'expected_return': expected_return,
                'confidence': confidence
            }
    
    def _generate_buy_signal(self, symbol: str, current_price: float,
                           expected_return: float, confidence: float,
                           portfolio: Dict) -> Dict:
        """Generate buy signal with position sizing"""
        if len(portfolio['positions']) >= self.max_positions:
            if symbol not in portfolio['positions']:
                return {
                    'action': 'hold',
                    'reason': f'Max positions reached ({self.max_positions})',
                    'expected_return': expected_return,
                    'confidence': confidence
                }
        
        max_investment = portfolio['total_value'] * self.max_position_pct
        available_cash = portfolio['cash'] * (1 - self.min_cash_reserve)
        investment_amount = min(max_investment, available_cash)
        
        if investment_amount < current_price:
            return {
                'action': 'hold',
                'reason': 'Insufficient cash for minimum position',
                'expected_return': expected_return,
                'confidence': confidence
            }
        
        shares = int(investment_amount / current_price)
        confidence_multiplier = min(confidence / self.min_confidence, 2.0)
        shares = int(shares * confidence_multiplier)
        
        if shares == 0:
            return {
                'action': 'hold',
                'reason': 'Position size too small',
                'expected_return': expected_return,
                'confidence': confidence
            }
        
        return {
            'action': 'buy',
            'shares': shares,
            'investment_amount': shares * current_price,
            'expected_return': expected_return,
            'confidence': confidence,
            'reason': f'Strong buy signal: {expected_return:.2%} expected return'
        }
    
    def _generate_sell_signal(self, symbol: str, current_price: float,
                            expected_return: float, confidence: float,
                            portfolio: Dict) -> Dict:
        """Generate sell signal"""
        if symbol not in portfolio['positions']:
            return {
                'action': 'hold',
                'reason': 'No position to sell',
                'expected_return': expected_return,
                'confidence': confidence
            }
        
        position = portfolio['positions'][symbol]
        shares_to_sell = position['shares']
        
        if confidence < 0.8:
            shares_to_sell = int(shares_to_sell * 0.5)
        
        if shares_to_sell == 0:
            return {
                'action': 'hold',
                'reason': 'Position too small to sell',
                'expected_return': expected_return,
                'confidence': confidence
            }
        
        return {
            'action': 'sell',
            'shares': shares_to_sell,
            'expected_return': expected_return,
            'confidence': confidence,
            'reason': f'Sell signal: {expected_return:.2%} expected decline'
        }
