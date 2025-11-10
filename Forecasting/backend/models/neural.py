import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, data: np.ndarray, seq_length: int = 60):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Return shape: (seq_length, 1) for x and (1,) for y
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """GRU model for time series forecasting"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class NeuralForecaster:
    """Neural network-based forecasting models"""
    
    def __init__(self, device: str = None, db=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = MinMaxScaler()
        self.seq_length = 60
        self.db = db
    
    def prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale data for neural networks"""
        # Scale data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Split into train and test
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        return train_data, test_data
    
    def train_model(self, model: nn.Module, train_data: np.ndarray,
                   epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        """Train a neural network model"""
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(train_data, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                # batch_x shape: (batch_size, seq_length, 1)
                # batch_y shape: (batch_size, 1)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(-1).to(self.device)
                
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def save_model(self, symbol: str, model_name: str, model: nn.Module, config: Dict):
        """
        Save trained model to MongoDB
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model (lstm, gru)
            model: Trained PyTorch model
            config: Model configuration
        """
        if self.db is None:
            print("Warning: No database connection, model not saved")
            return
        
        import io
        import pickle
        
        # Serialize model state dict
        model_buffer = io.BytesIO()
        torch.save(model.state_dict(), model_buffer)
        model_state = model_buffer.getvalue()
        
        # Serialize scaler
        scaler_state = pickle.dumps(self.scaler)
        
        # Store in MongoDB
        self.db.store_model(symbol, model_name, model_state, scaler_state, config)
        print(f"Model saved: {model_name} for {symbol}")
    
    def load_model(self, symbol: str, model_name: str) -> Optional[nn.Module]:
        """
        Load trained model from MongoDB
        
        Args:
            symbol: Stock/Crypto symbol
            model_name: Name of the model (lstm, gru)
        
        Returns:
            Loaded model or None if not found
        """
        if self.db is None:
            return None
        
        import io
        import pickle
        
        # Retrieve from MongoDB
        model_doc = self.db.get_model(symbol, model_name)
        
        if model_doc is None:
            return None
        
        try:
            # Load scaler
            self.scaler = pickle.loads(model_doc['scaler_state'])
            
            # Recreate model with saved config
            config = model_doc['config']
            if model_name.lower() == 'lstm':
                model = LSTMModel(
                    input_size=config.get('input_size', 1),
                    hidden_size=config.get('hidden_size', 64),
                    num_layers=config.get('num_layers', 2),
                    dropout=config.get('dropout', 0.2)
                ).to(self.device)
            elif model_name.lower() == 'gru':
                model = GRUModel(
                    input_size=config.get('input_size', 1),
                    hidden_size=config.get('hidden_size', 64),
                    num_layers=config.get('num_layers', 2),
                    dropout=config.get('dropout', 0.2)
                ).to(self.device)
            else:
                return None
            
            # Load model state
            model_buffer = io.BytesIO(model_doc['model_state'])
            model.load_state_dict(torch.load(model_buffer, map_location=self.device))
            model.eval()
            
            print(f"Model loaded: {model_name} for {symbol}")
            return model
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def lstm_forecast(self, data: pd.Series, steps: int = 24,
                     epochs: int = 50, symbol: str = None, use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        LSTM forecast
        
        Args:
            data: Historical price series
            steps: Number of steps to forecast
            epochs: Training epochs
            symbol: Stock symbol (for caching)
            use_cache: Whether to use cached model
        
        Returns:
            Tuple of (predictions, metrics)
        """
        try:
            # Prepare data (this fits the scaler)
            train_data, test_data = self.prepare_data(data)
            
            # Try to load cached model
            model = None
            if use_cache and symbol and self.db:
                try:
                    model = self.load_model(symbol, 'lstm')
                    print(f"Using cached LSTM model for {symbol}")
                except Exception as e:
                    print(f"Could not load cached model: {e}")
                    model = None
            
            # Create and train model if not loaded from cache
            should_save_version = False
            if model is None:
                print(f"Training new LSTM model for {symbol}")
                model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(self.device)
                self.train_model(model, train_data, epochs=epochs)
                should_save_version = True
            else:
                # Fine-tune existing model with new data
                print(f"Fine-tuning LSTM model for {symbol} ({epochs} epochs)")
                self.train_model(model, train_data, epochs=epochs)
                should_save_version = True
            
            # Save model version after training
            if should_save_version and symbol and self.db:
                try:
                    from adaptive_learning import ModelVersionManager
                    version_manager = ModelVersionManager(self.db)
                    
                    config = {
                        'input_size': 1,
                        'hidden_size': 64,
                        'num_layers': 2,
                        'dropout': 0.2,
                        'seq_length': self.seq_length
                    }
                    
                    # Calculate metrics first
                    metrics_for_save = self._calculate_metrics(model, train_data, test_data)
                    
                    # Save new version
                    version = version_manager.save_version(
                        symbol=symbol,
                        model_name='lstm',
                        model=model,
                        scaler=self.scaler,
                        config=config,
                        metrics=metrics_for_save,
                        update_type='patch'
                    )
                    print(f"✓ Saved model version: {version}")
                except Exception as e:
                    print(f"Could not save model version: {e}")
            
            # Make predictions
            model.eval()
            predictions = []
            
            # Use last seq_length points as input
            current_seq = train_data[-self.seq_length:].copy()
            
            with torch.no_grad():
                for _ in range(steps):
                    # Shape: (1, seq_length, 1)
                    x = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
                    pred = model(x).cpu().numpy()[0, 0]
                    predictions.append(pred)
                    
                    # Update sequence - append as (1,) array
                    current_seq = np.append(current_seq[1:], [[pred]], axis=0)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            
            # Calculate metrics on test set using one-step-ahead predictions
            if len(test_data) >= self.seq_length + 10:
                test_predictions = []
                
                # Use actual historical data for each prediction (not recursive)
                with torch.no_grad():
                    for i in range(len(test_data) - self.seq_length):
                        # Use actual data up to this point
                        test_seq = np.concatenate([train_data, test_data[:i+self.seq_length]])[-self.seq_length:]
                        x = torch.FloatTensor(test_seq).unsqueeze(0).to(self.device)
                        pred = model(x).cpu().numpy()[0, 0]
                        test_predictions.append(pred)
                
                # Get actual values for comparison
                test_actual_scaled = test_data[self.seq_length:self.seq_length+len(test_predictions)]
                test_predictions_array = np.array(test_predictions).reshape(-1, 1)
                
                test_predictions_inv = self.scaler.inverse_transform(test_predictions_array).flatten()
                test_actual_inv = self.scaler.inverse_transform(test_actual_scaled).flatten()
                
                rmse = np.sqrt(mean_squared_error(test_actual_inv, test_predictions_inv))
                mae = mean_absolute_error(test_actual_inv, test_predictions_inv)
                mape = np.mean(np.abs((test_actual_inv - test_predictions_inv) / (test_actual_inv + 1e-8))) * 100
            else:
                # Not enough test data, use training error as proxy
                train_predictions = []
                with torch.no_grad():
                    for i in range(self.seq_length, len(train_data)):
                        train_seq = train_data[i-self.seq_length:i]
                        x = torch.FloatTensor(train_seq).unsqueeze(0).to(self.device)
                        pred = model(x).cpu().numpy()[0, 0]
                        train_predictions.append(pred)
                
                if len(train_predictions) > 0:
                    train_actual = train_data[self.seq_length:self.seq_length+len(train_predictions)]
                    train_pred_inv = self.scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten()
                    train_actual_inv = self.scaler.inverse_transform(train_actual).flatten()
                    
                    rmse = np.sqrt(mean_squared_error(train_actual_inv, train_pred_inv))
                    mae = mean_absolute_error(train_actual_inv, train_pred_inv)
                    mape = np.mean(np.abs((train_actual_inv - train_pred_inv) / (train_actual_inv + 1e-8))) * 100
                else:
                    rmse, mae, mape = 0, 0, 0
        
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'model': 'LSTM'
            }
            
            self.model = model
            return predictions, metrics
        
        except Exception as e:
            print(f"LSTM forecast error: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy predictions on error
            return np.zeros(steps), {
                'rmse': 0,
                'mae': 0,
                'mape': 0,
                'model': 'LSTM',
                'error': str(e)
            }
    
    def _calculate_metrics(self, model: nn.Module, train_data: np.ndarray, test_data: np.ndarray) -> Dict:
        """Helper method to calculate metrics for model saving"""
        model.eval()
        
        # If not enough test data, use training data for metrics
        if len(test_data) < self.seq_length + 10:
            # Use last 50 points of training data
            if len(train_data) < self.seq_length + 50:
                return {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
            
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for i in range(len(train_data) - 50, len(train_data)):
                    if i < self.seq_length:
                        continue
                    seq = train_data[i-self.seq_length:i]
                    x = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    pred = model(x).cpu().numpy()[0, 0]
                    predictions.append(pred)
                    actuals.append(train_data[i][0])
        else:
            # Use test data
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for i in range(self.seq_length, min(len(test_data), self.seq_length + 50)):
                    test_seq = np.concatenate([train_data, test_data[:i]])[-self.seq_length:]
                    x = torch.FloatTensor(test_seq).unsqueeze(0).to(self.device)
                    pred = model(x).cpu().numpy()[0, 0]
                    predictions.append(pred)
                    actuals.append(test_data[i][0])
        
        if len(predictions) > 0:
            predictions_inv = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            actuals_inv = self.scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(actuals_inv, predictions_inv))
            mae = mean_absolute_error(actuals_inv, predictions_inv)
            mape = np.mean(np.abs((actuals_inv - predictions_inv) / (actuals_inv + 1e-8))) * 100
            
            return {'rmse': float(rmse), 'mae': float(mae), 'mape': float(mape)}
        
        return {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
    
    def gru_forecast(self, data: pd.Series, steps: int = 24,
                    epochs: int = 50, symbol: str = None, use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        GRU forecast
        
        Args:
            data: Historical price series
            steps: Number of steps to forecast
            epochs: Training epochs
            symbol: Stock symbol (for caching)
            use_cache: Whether to use cached model
        
        Returns:
            Tuple of (predictions, metrics)
        """
        try:
            train_data, test_data = self.prepare_data(data)
            
            # Try to load cached model
            model = None
            if use_cache and symbol and self.db:
                try:
                    model = self.load_model(symbol, 'gru')
                    print(f"Using cached GRU model for {symbol}")
                except Exception as e:
                    print(f"Could not load cached model: {e}")
                    model = None
            
            # Create and train model if not loaded from cache
            should_save_version = False
            if model is None:
                print(f"Training new GRU model for {symbol}")
                model = GRUModel(input_size=1, hidden_size=64, num_layers=2).to(self.device)
                self.train_model(model, train_data, epochs=epochs)
                should_save_version = True
            else:
                # Fine-tune existing model with new data
                print(f"Fine-tuning GRU model for {symbol} ({epochs} epochs)")
                self.train_model(model, train_data, epochs=epochs)
                should_save_version = True
            
            # Save model version after training
            if should_save_version and symbol and self.db:
                try:
                    from adaptive_learning import ModelVersionManager
                    version_manager = ModelVersionManager(self.db)
                    
                    config = {
                        'input_size': 1,
                        'hidden_size': 64,
                        'num_layers': 2,
                        'dropout': 0.2,
                        'seq_length': self.seq_length
                    }
                    
                    # Calculate metrics first
                    metrics_for_save = self._calculate_metrics(model, train_data, test_data)
                    
                    # Save new version
                    version = version_manager.save_version(
                        symbol=symbol,
                        model_name='gru',
                        model=model,
                        scaler=self.scaler,
                        config=config,
                        metrics=metrics_for_save,
                        update_type='patch'
                    )
                    print(f"✓ Saved model version: {version}")
                except Exception as e:
                    print(f"Could not save model version: {e}")
            
            model.eval()
            predictions = []
            current_seq = train_data[-self.seq_length:].copy()
            
            with torch.no_grad():
                for _ in range(steps):
                    x = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
                    pred = model(x).cpu().numpy()[0, 0]
                    predictions.append(pred)
                    current_seq = np.append(current_seq[1:], [[pred]], axis=0)
            
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            
            # Calculate metrics on test set using one-step-ahead predictions
            if len(test_data) >= self.seq_length + 10:
                test_predictions = []
                
                # Use actual historical data for each prediction (not recursive)
                with torch.no_grad():
                    for i in range(len(test_data) - self.seq_length):
                        # Use actual data up to this point
                        test_seq = np.concatenate([train_data, test_data[:i+self.seq_length]])[-self.seq_length:]
                        x = torch.FloatTensor(test_seq).unsqueeze(0).to(self.device)
                        pred = model(x).cpu().numpy()[0, 0]
                        test_predictions.append(pred)
                
                # Get actual values for comparison
                test_actual_scaled = test_data[self.seq_length:self.seq_length+len(test_predictions)]
                test_predictions_array = np.array(test_predictions).reshape(-1, 1)
                
                test_predictions_inv = self.scaler.inverse_transform(test_predictions_array).flatten()
                test_actual_inv = self.scaler.inverse_transform(test_actual_scaled).flatten()
                
                rmse = np.sqrt(mean_squared_error(test_actual_inv, test_predictions_inv))
                mae = mean_absolute_error(test_actual_inv, test_predictions_inv)
                mape = np.mean(np.abs((test_actual_inv - test_predictions_inv) / (test_actual_inv + 1e-8))) * 100
            else:
                # Not enough test data, use training error as proxy
                train_predictions = []
                with torch.no_grad():
                    for i in range(self.seq_length, len(train_data)):
                        train_seq = train_data[i-self.seq_length:i]
                        x = torch.FloatTensor(train_seq).unsqueeze(0).to(self.device)
                        pred = model(x).cpu().numpy()[0, 0]
                        train_predictions.append(pred)
                
                if len(train_predictions) > 0:
                    train_actual = train_data[self.seq_length:self.seq_length+len(train_predictions)]
                    train_pred_inv = self.scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten()
                    train_actual_inv = self.scaler.inverse_transform(train_actual).flatten()
                    
                    rmse = np.sqrt(mean_squared_error(train_actual_inv, train_pred_inv))
                    mae = mean_absolute_error(train_actual_inv, train_pred_inv)
                    mape = np.mean(np.abs((train_actual_inv - train_pred_inv) / (train_actual_inv + 1e-8))) * 100
                else:
                    rmse, mae, mape = 0, 0, 0
        
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'model': 'GRU'
            }
            
            self.model = model
            return predictions, metrics
        
        except Exception as e:
            print(f"GRU forecast error: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy predictions on error
            return np.zeros(steps), {
                'rmse': 0,
                'mae': 0,
                'mape': 0,
                'model': 'GRU',
                'error': str(e)
            }
