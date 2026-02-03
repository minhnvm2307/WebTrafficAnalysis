"""
Configuration parameters for baseline LSTM and enhanced Seq2Seq models.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaselineParams:
    """Parameters for the baseline LSTM model from notebook."""
    train_window: int = 64   
    predict_window: int = 1
    batch_size: int = 128     
    hidden_size: int = 128
    num_layers: int = 2      
    dropout: float = 0.2
    
    # Paths
    data_dir: str = './dataset'
    save_dir: str = './model/checkpoints/lstm_15p.pth'
    scaler_path: str = './model/checkpoints/scaler_lstm_15p.pkl'


@dataclass
class EnhancedParams:
    """Parameters for the enhanced Seq2Seq model from architecture document."""
    data_dir: str = './dataset'
    save_dir: str = './checkpoints/seq2seq'
    scaler_path: str = './checkpoints/scaler_seq2seq.pkl'
    train_window: int = 120 
    # LAG_FEATURES = [11, 16, 24, 30]
    
    # Model architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


def get_params(model_type: str = "baseline"):
    """
    Get parameter configuration for specified model type.
    
    Args:
        model_type: Either 'baseline' or 'enhanced'
        
    Returns:
        Parameter configuration object
    """
    if model_type == "baseline":
        return BaselineParams()
    elif model_type == "enhanced":
        return EnhancedParams()
    else:
        raise ValueError(f"Unknown model type: {model_type}")