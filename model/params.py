from dataclasses import dataclass


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
    scaler_path: str = './model/checkpoints/scaler.pkl'


@dataclass
class EnhancedParams:
    """Parameters for the enhanced Seq2Seq model from architecture document."""
    data_dir: str = './dataset'
    save_dir: str = './model/checkpoints/seq2seq_15p.pth'
    scaler_path: str = './model/checkpoints/scaler.pkl'
    train_window: int = 15
    LAG_FEATURES = [31, 32, 34, 35]
    
    # Model architecture (matching trained model)
    hidden_size: int = 267
    num_layers: int = 1
    dropout: float = 0.2


def get_params(model_type: str = "baseline"):
    if model_type == "baseline":
        return BaselineParams()
    elif model_type == "enhanced":
        return EnhancedParams()
    else:
        raise ValueError(f"Unknown model type: {model_type}")