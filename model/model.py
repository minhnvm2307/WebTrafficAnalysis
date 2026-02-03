"""
Model implementations: Baseline LSTM and Enhanced Seq2Seq.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, encoder_outputs):
        # encoder_outputs: [batch, seq_len, hidden]
        # Tính năng lượng (scores)
        attn_logits = self.attention(encoder_outputs) # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # Nhân trọng số: [batch, seq_len, 1] * [batch, seq_len, hidden]
        context = torch.sum(attn_weights * encoder_outputs, dim=1) # [batch, hidden]
        return context, attn_weights

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_size)
        
        # Decoder 
        # Quan trọng: Kích thước phải bằng (số feature của y) + (hidden_size của context)
        # Nếu y_features có cùng số lượng cột với x_features, dùng input_size
        self.decoder_input_size = input_size + hidden_size 
        
        self.decoder = nn.LSTM(
            self.decoder_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x_features, y_features):
        # Encode: [batch, lookback, input_size]
        encoder_outputs, (hidden, cell) = self.encoder(x_features)
        
        # Attention: context [batch, hidden]
        context, attn_weights = self.attention(encoder_outputs)
        
        # Expand context: [batch, forecast_len, hidden]
        batch_size, forecast_len, _ = y_features.shape
        context_expanded = context.unsqueeze(1).repeat(1, forecast_len, 1)
        
        # Concatenate: [batch, forecast_len, input_size + hidden]
        decoder_input = torch.cat([y_features, context_expanded], dim=2)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Predict: [batch, forecast_len]
        predictions = self.fc_out(decoder_output).squeeze(-1)
        
        return predictions, attn_weights
    
    def predict(self, x_features, y_features):
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(x_features, y_features)
        return predictions

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)

        last_out = out[:, -1, :] 
        prediction = self.fc(last_out)
        return prediction
    
    def predict(self, series: np.ndarray):
        self.eval()
        input_tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            output = self.forward(input_tensor)
        pred = output.squeeze().cpu().numpy()
        return pred

class SMAPELoss(nn.Module):
    """
    Smooth SMAPE Loss for training.
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Args:
            epsilon: Smoothing parameter to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth SMAPE loss.
        
        Args:
            pred: Predictions
            target: Ground truth
            
        Returns:
            Loss value
        """
        pred = pred.view_as(target)
        
        diff = torch.abs(pred - target)
        denominator = torch.abs(pred) + torch.abs(target) + self.epsilon
        smape = 2.0 * diff / denominator
        
        return torch.mean(smape)


def get_loss_function(loss_type: str = 'l1', epsilon: float = 0.1):
    """
    Get loss function based on type.
    
    Args:
        loss_type: 'l1', 'mse', or 'smape'
        epsilon: SMAPE epsilon parameter
        
    Returns:
        Loss function
    """
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'smape':
        return SMAPELoss(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
