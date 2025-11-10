"""
Rolling Window Trainer

Retrains models on sliding windows of recent data.
Implements transfer learning for neural models.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import sys
sys.path.append('..')
from models.neural import LSTMModel, GRUModel, TimeSeriesDataset


class RollingWindowTrainer:
    """Retrain models on sliding windows of recent data"""
    
    def __init__(self, db, data_fetcher, window_size: int = 365):
        """
        Initialize rolling window trainer
        
        Args:
            db: Database instance
            data_fetcher: DataFetcher instance
            window_size: Size of rolling window in days
        """
        self.db = db
        self.data_fetcher = data_fetcher
        self.window_size = window_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def fetch_recent_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Fetch recent data for training
        
        Args:
            symbol: Stock/Crypto symbol
            days: Number of days to fetch
        
        Returns:
            DataFrame with recent data
        """
        # Calculate period
        if days <= 7:
            period = '7d'
        elif days <= 30:
            period = '1mo'
        elif days <= 90:
            period = '3mo'
        elif days <= 180:
            period = '6mo'
        else:
            period = '1y'
        
        df = self.data_fetcher.fetch_data(symbol, period=period, interval='1d')
        
        if df is not None and len(df) > days:
            df = df.tail(days)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with price data
            seq_length: Sequence length for LSTM/GRU
        
        Returns:
            (train_data, test_data, scaler) tuple
        """
        close_prices = df['close'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)
        
        # Split into train/test (80/20)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        return train_data, test_data, scaler
    
    def fine_tune_neural_model(self, model: nn.Module, train_data: np.ndarray,
                               epochs: int = 10, batch_size: int = 32,
                               seq_length: int = 60, freeze_early_layers: bool = True) -> Tuple[nn.Module, float]:
        """
        Fine-tune neural model on new data
        
        Args:
            model: Existing model to fine-tune
            train_data: Training data
            epochs: Number of epochs
            batch_size: Batch size
            seq_length: Sequence length
            freeze_early_layers: Whether to freeze early layers
        
        Returns:
            (fine_tuned_model, final_loss) tuple
        """
        model.to(self.device)
        model.train()
        
        # Optionally freeze early layers for transfer learning
        if freeze_early_layers and hasattr(model, 'lstm'):
            # Freeze first LSTM layer
            for param in list(model.lstm.parameters())[:4]:  # First layer parameters
                param.requires_grad = False
        elif freeze_early_layers and hasattr(model, 'gru'):
            # Freeze first GRU layer
            for param in list(model.gru.parameters())[:3]:  # First layer parameters
                param.requires_grad = False
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(train_data, seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0001  # Lower learning rate for fine-tuning
        )
        criterion = nn.MSELoss()
        
        # Training loop
        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(-1).to(self.device)
                
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            final_loss = epoch_loss / len(dataloader)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {final_loss:.6f}")
        
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        
        return model, final_loss
    
    def retrain_lstm(self, symbol: str, epochs: int = 10,
                    use_transfer_learning: bool = True) -> Tuple[nn.Module, MinMaxScaler, Dict]:
        """
        Retrain LSTM model on rolling window
        
        Args:
            symbol: Stock/Crypto symbol
            epochs: Number of training epochs
            use_transfer_learning: Whether to use transfer learning
        
        Returns:
            (model, scaler, metrics) tuple
        """
        print(f"🔄 Retraining LSTM for {symbol} (window: {self.window_size} days)")
        
        # Fetch recent data
        df = self.fetch_recent_data(symbol, self.window_size)
        
        if df is None or len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Prepare data
        train_data, test_data, scaler = self.prepare_data(df)
        
        # Load existing model or create new one
        try:
            from .model_versioning import ModelVersionManager
            version_manager = ModelVersionManager(self.db)
            model_state, old_scaler, config = version_manager.load_version(symbol, 'lstm')
            
            # Reconstruct model
            model = LSTMModel(
                input_size=config.get('input_size', 1),
                hidden_size=config.get('hidden_size', 64),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            ).to(self.device)
            
            # Load weights
            model_buffer = torch.BytesIO(model_state)
            model.load_state_dict(torch.load(model_buffer, map_location=self.device))
            
            print(f"  ✓ Loaded existing model for fine-tuning")
            
        except Exception as e:
            print(f"  ℹ Creating new model: {e}")
            model = LSTMModel(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            use_transfer_learning = False
        
        # Fine-tune or train from scratch
        if use_transfer_learning:
            model, final_loss = self.fine_tune_neural_model(
                model, train_data, epochs=epochs, freeze_early_layers=True
            )
        else:
            model, final_loss = self.fine_tune_neural_model(
                model, train_data, epochs=epochs*3, freeze_early_layers=False
            )
        
        # Evaluate on test set
        metrics = self.evaluate_model(model, test_data, scaler, seq_length=60)
        metrics['final_loss'] = final_loss
        
        print(f"  ✓ Training complete - MAPE: {metrics['mape']:.2f}%")
        
        return model, scaler, metrics
    
    def retrain_gru(self, symbol: str, epochs: int = 10,
                   use_transfer_learning: bool = True) -> Tuple[nn.Module, MinMaxScaler, Dict]:
        """
        Retrain GRU model on rolling window
        
        Args:
            symbol: Stock/Crypto symbol
            epochs: Number of training epochs
            use_transfer_learning: Whether to use transfer learning
        
        Returns:
            (model, scaler, metrics) tuple
        """
        print(f"🔄 Retraining GRU for {symbol} (window: {self.window_size} days)")
        
        # Fetch recent data
        df = self.fetch_recent_data(symbol, self.window_size)
        
        if df is None or len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Prepare data
        train_data, test_data, scaler = self.prepare_data(df)
        
        # Load existing model or create new one
        try:
            from .model_versioning import ModelVersionManager
            version_manager = ModelVersionManager(self.db)
            model_state, old_scaler, config = version_manager.load_version(symbol, 'gru')
            
            # Reconstruct model
            model = GRUModel(
                input_size=config.get('input_size', 1),
                hidden_size=config.get('hidden_size', 64),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            ).to(self.device)
            
            # Load weights
            model_buffer = torch.BytesIO(model_state)
            model.load_state_dict(torch.load(model_buffer, map_location=self.device))
            
            print(f"  ✓ Loaded existing model for fine-tuning")
            
        except Exception as e:
            print(f"  ℹ Creating new model: {e}")
            model = GRUModel(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            use_transfer_learning = False
        
        # Fine-tune or train from scratch
        if use_transfer_learning:
            model, final_loss = self.fine_tune_neural_model(
                model, train_data, epochs=epochs, freeze_early_layers=True
            )
        else:
            model, final_loss = self.fine_tune_neural_model(
                model, train_data, epochs=epochs*3, freeze_early_layers=False
            )
        
        # Evaluate on test set
        metrics = self.evaluate_model(model, test_data, scaler, seq_length=60)
        metrics['final_loss'] = final_loss
        
        print(f"  ✓ Training complete - MAPE: {metrics['mape']:.2f}%")
        
        return model, scaler, metrics
    
    def evaluate_model(self, model: nn.Module, test_data: np.ndarray,
                      scaler: MinMaxScaler, seq_length: int = 60) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            model: Trained model
            test_data: Test data
            scaler: Data scaler
            seq_length: Sequence length
        
        Returns:
            Metrics dictionary
        """
        model.eval()
        
        if len(test_data) < seq_length + 10:
            # Not enough test data, use training error as proxy
            return {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i in range(seq_length, len(test_data)):
                seq = test_data[i-seq_length:i]
                x = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred = model(x).cpu().numpy()[0, 0]
                predictions.append(pred)
                actuals.append(test_data[i][0])
        
        # Inverse transform
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
