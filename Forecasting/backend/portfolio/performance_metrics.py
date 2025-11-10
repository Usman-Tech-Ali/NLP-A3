"""Performance Metrics - Calculates portfolio performance and risk metrics"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from pymongo import ASCENDING


class PerformanceMetrics:
    """Calculates portfolio performance and risk metrics"""
    
    def __init__(self, db, portfolio_manager):
        self.db = db
        self.portfolio_manager = portfolio_manager
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_daily_performance(self, portfolio_id: str) -> Dict:
        """Calculate and store daily performance metrics"""
        portfolio = self.portfolio_manager.update_portfolio_values(portfolio_id)
        if not portfolio:
            return {}
        
        current_value = portfolio['total_value']
        current_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get yesterday's value
        yesterday = current_date - timedelta(days=1)
        yesterday_record = self.db.db['portfolio_performance'].find_one({
            'portfolio_id': portfolio_id,
            'date': {'$gte': yesterday, '$lt': current_date}
        })
        
        if yesterday_record:
            yesterday_value = yesterday_record['total_value']
            daily_return = (current_value - yesterday_value) / yesterday_value
        else:
            daily_return = 0.0
        
        # Calculate cumulative return
        initial_value = self.portfolio_manager.initial_cash
        cumulative_return = (current_value - initial_value) / initial_value
        
        # Get historical returns for volatility and Sharpe calculation
        returns_data = self._get_historical_returns(portfolio_id, days=252)
        
        if len(returns_data) > 1:
            volatility = np.std(returns_data) * np.sqrt(252)  # Annualized
            avg_return = np.mean(returns_data) * 252  # Annualized
            sharpe_ratio = (avg_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
        
        positions_value = sum(pos['current_value'] for pos in portfolio['positions'].values())
        
        performance = {
            'portfolio_id': portfolio_id,
            'date': current_date,
            'total_value': float(current_value),
            'cash': float(portfolio['cash']),
            'positions_value': float(positions_value),
            'daily_return': float(daily_return),
            'cumulative_return': float(cumulative_return),
            'sharpe_ratio': float(sharpe_ratio),
            'volatility': float(volatility)
        }
        
        self.db.db['portfolio_performance'].replace_one(
            {'portfolio_id': portfolio_id, 'date': current_date},
            performance,
            upsert=True
        )
        
        return performance
    
    def _get_historical_returns(self, portfolio_id: str, days: int = 252) -> List[float]:
        """Get historical daily returns"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        records = list(self.db.db['portfolio_performance'].find(
            {
                'portfolio_id': portfolio_id,
                'date': {'$gte': cutoff_date}
            },
            sort=[('date', ASCENDING)]
        ))
        
        if len(records) < 2:
            return []
        
        returns = []
        for i in range(1, len(records)):
            prev_value = records[i-1]['total_value']
            curr_value = records[i]['total_value']
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        return returns
    
    def calculate_max_drawdown(self, portfolio_id: str, days: int = 252) -> Dict:
        """Calculate maximum drawdown over specified period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        records = list(self.db.db['portfolio_performance'].find(
            {
                'portfolio_id': portfolio_id,
                'date': {'$gte': cutoff_date}
            },
            sort=[('date', ASCENDING)]
        ))
        
        if len(records) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'current_drawdown': 0.0
            }
        
        values = [record['total_value'] for record in records]
        dates = [record['date'] for record in records]
        
        peaks = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - peaks) / peaks
        
        max_drawdown = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        
        max_dd_start = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdowns[i] == 0:
                max_dd_start = i
                break
        
        max_dd_duration = max_dd_idx - max_dd_start
        current_drawdown = drawdowns[-1]
        
        return {
            'max_drawdown': float(max_drawdown),
            'max_drawdown_duration': int(max_dd_duration),
            'current_drawdown': float(current_drawdown),
            'max_drawdown_date': dates[max_dd_idx].isoformat() if max_dd_idx < len(dates) else None
        }
    
    def calculate_win_rate(self, portfolio_id: str, days: int = 90) -> Dict:
        """Calculate trading win rate and profit metrics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        trades = list(self.db.db['trades'].find({
            'portfolio_id': portfolio_id,
            'action': 'sell',
            'timestamp': {'$gte': cutoff_date},
            'realized_pnl': {'$exists': True}
        }))
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0
            }
        
        pnls = [trade['realized_pnl'] for trade in trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'total_pnl': float(sum(pnls))
        }
    
    def get_performance_summary(self, portfolio_id: str, days: int = 30) -> Dict:
        """Get comprehensive performance summary"""
        current_performance = self.calculate_daily_performance(portfolio_id)
        trading_metrics = self.calculate_win_rate(portfolio_id, days)
        drawdown_metrics = self.calculate_max_drawdown(portfolio_id)
        positions_summary = self.portfolio_manager.get_positions_summary(portfolio_id)
        
        period_start = datetime.utcnow() - timedelta(days=days)
        start_record = self.db.db['portfolio_performance'].find_one({
            'portfolio_id': portfolio_id,
            'date': {'$gte': period_start}
        }, sort=[('date', ASCENDING)])
        
        if start_record:
            period_return = (current_performance['total_value'] - start_record['total_value']) / start_record['total_value']
        else:
            period_return = current_performance.get('cumulative_return', 0)
        
        return {
            'portfolio_id': portfolio_id,
            'as_of_date': datetime.utcnow().isoformat(),
            'period_days': days,
            'total_value': current_performance.get('total_value', 0),
            'cash': positions_summary.get('cash', 0),
            'positions_value': positions_summary.get('positions_value', 0),
            'unrealized_pnl': positions_summary.get('total_unrealized_pnl', 0),
            'period_return': float(period_return),
            'cumulative_return': current_performance.get('cumulative_return', 0),
            'daily_return': current_performance.get('daily_return', 0),
            'volatility': current_performance.get('volatility', 0),
            'sharpe_ratio': current_performance.get('sharpe_ratio', 0),
            'max_drawdown': drawdown_metrics.get('max_drawdown', 0),
            'current_drawdown': drawdown_metrics.get('current_drawdown', 0),
            'total_trades': trading_metrics.get('total_trades', 0),
            'win_rate': trading_metrics.get('win_rate', 0),
            'profit_factor': trading_metrics.get('profit_factor', 0),
            'realized_pnl': trading_metrics.get('total_pnl', 0),
            'number_of_positions': len(positions_summary.get('positions', {})),
            'cash_percentage': positions_summary.get('cash_percentage', 0)
        }
    
    def get_performance_history(self, portfolio_id: str, days: int = 90) -> List[Dict]:
        """Get historical performance data for charting"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        records = list(self.db.db['portfolio_performance'].find(
            {
                'portfolio_id': portfolio_id,
                'date': {'$gte': cutoff_date}
            },
            sort=[('date', ASCENDING)]
        ))
        
        for record in records:
            record['_id'] = str(record['_id'])
            record['date'] = record['date'].isoformat()
        
        return records
