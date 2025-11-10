"""
Online Learner

Implements incremental learning for neural network models.
Updates model parameters with each new data point.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


class OnlineLearner:
    """Incremental learning for neural models"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 device: str = None):
        """
        Initialize online learner
        
        Args:
            model: PyTorch model
            learning_rate: Learning rate for updates
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = MinMaxScaler()
        
        # Statistics
        self.update_count = 0
        self.running_loss = 0.0
        self.loss_history = []
        
    def update(self, sequence: np.ndarray, target: float) -> float:
        """
        Update model with single new observation
        
        Args:
            sequence: Input sequence (e.g., last 60 prices)
            target: Target value (next price)
        
        Returns:
            Loss value
        """
        self.model.train()
        
        # Prepare input
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        y = torch.FloatTensor([[target]]).to(self.device)
        
        # Forward pass
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update statistics
        loss_value = loss.item()
        self.update_count += 1
        self.running_loss += loss_value
        self.loss_history.append(loss_value)
        
        # Keep only last 1000 losses
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)
        
        return loss_value
    
    def batch_update(self, sequences: np.ndarray, targets: np.ndarray,
                    batch_size: int = 32) -> float:
        """
        Update model with batch of new observations
        
        Args:
            sequences: Input sequences
            targets: Target values
            batch_size: Batch size for updates
        
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(sequences), batch_size):
            batch_x = sequences[i:i+batch_size]
            batch_y = targets[i:i+batch_size]
            
            x = torch.FloatTensor(batch_x).to(self.device)
            y = torch.FloatTensor(batch_y).unsqueeze(-1).to(self.device)
            
            # Forward pass
            predictions = self.model(x)
            loss = self.criterion(predictions, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        self.update_count += len(sequences)
        
        return avg_loss
    
    def should_full_retrain(self, window_size: int = 100,
                           threshold: float = 1.5) -> bool:
        """
        Determine if full retraining is needed based on loss trend
        
        Args:
            window_size: Number of recent losses to consider
            threshold: Loss increase threshold
        
        Returns:
            True if full retrain recommended
        """
        if len(self.loss_history) < window_size * 2:
            return False
        
        # Compare recent losses to older losses
        recent_losses = self.loss_history[-window_size:]
        older_losses = self.loss_history[-window_size*2:-window_size]
        
        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)
        
        # If recent loss is significantly higher, recommend retrain
        if recent_avg > older_avg * threshold:
            print(f"⚠️  Loss increased: {older_avg:.4f} → {recent_avg:.4f}")
            return True
        
        return False
    
    def get_statistics(self) -> dict:
        """Get learning statistics"""
        if len(self.loss_history) == 0:
            return {
                'update_count': self.update_count,
                'avg_loss': 0.0,
                'recent_loss': 0.0
            }
        
        return {
            'update_count': self.update_count,
            'avg_loss': self.running_loss / self.update_count if self.update_count > 0 else 0.0,
            'recent_loss': np.mean(self.loss_history[-100:]) if len(self.loss_history) >= 100 else np.mean(self.loss_history),
            'loss_trend': 'increasing' if self.should_full_retrain() else 'stable'
        }
    
    def reset_statistics(self):
        """Reset learning statistics"""
        self.update_count = 0
        self.running_loss = 0.0
        self.loss_history = []
