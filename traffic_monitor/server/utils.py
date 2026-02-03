"""
ML Server Utilities - Model Loading and Inference
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from model.model import Seq2SeqModel, LSTMModel
from model.params import get_params
from model.input_pipe import ScalerWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor:
    def __init__(self, model_type: str = "baseline"):
        self.model_type = model_type
        self.config = get_params(model_type)
        self.scaler = ScalerWrapper(self.config.scaler_path, model_type)
        self.model = self._load_model()
        self._input_size = None  # For enhanced model

    def _load_model(self):
        if self.model_type == "baseline":
            model = LSTMModel(
                input_size=1,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=1,
                dropout=self.config.dropout
            )
        elif self.model_type == "enhanced":
            model = Seq2SeqModel(
                input_size= self._input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.load_state_dict(torch.load(self.config.save_dir, map_location=device))
        model.to(device)
        model.eval()
        return model
    
    def predict(self, series: np.ndarray, predict_horizon: int) -> np.ndarray:
        if len(series) < self.config.train_window:
            raise ValueError(f"Input series length {len(series)} is less than required window size {self.config.train_window}")
        else:
            series = series[-self.config.train_window:]
        scaled_series = self.scaler.transform(series.reshape(-1, 1)).flatten()
        
        predictions = []
        for i in range(predict_horizon):
            if self.model_type == "baseline":
                pred = self.model.predict(scaled_series)
                pred = np.atleast_1d(pred)
            elif self.model_type == "enhanced":
                pred = self.model.predict(scaled_series)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Update series with new prediction
            scaled_series = np.append(scaled_series, pred)[1:]
            predictions.append(pred)

        # Stack predictions into array and inverse transform
        predictions_array = np.concatenate(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions_array)

        return predictions.flatten()

def detect_anomaly(df, threshold_sigma=3.0):
    """Detect anomalies using Z-score method
    
    Args:
        df: DataFrame with request_count column
        threshold_sigma: Z-score threshold for anomaly detection
    
    Returns:
        dict: Anomaly detection results
    """
    if len(df) < 2:
        return {
            'is_anomaly': False,
            'z_score': 0.0,
            'reason': 'Insufficient data',
            'current_load': 0,
            'mean_load': 0.0,
            'std_load': 0.0
        }
    
    recent_load = df['request_count'].iloc[-1]
    mean_load = df['request_count'].mean()
    std_load = df['request_count'].std()
    
    if std_load == 0:
        return {
            'is_anomaly': False,
            'z_score': 0.0,
            'reason': 'Zero variance in data',
            'current_load': int(recent_load),
            'mean_load': float(mean_load),
            'std_load': 0.0
        }
    
    z_score = (recent_load - mean_load) / std_load
    is_anomaly = abs(z_score) > threshold_sigma
    
    reason = 'Normal traffic'
    if is_anomaly:
        if z_score > 0:
            reason = f'⚠️ Traffic SPIKE detected! Z-score: {z_score:.2f} (>{threshold_sigma}σ)'
        else:
            reason = f'Traffic DROP detected! Z-score: {z_score:.2f} (<-{threshold_sigma}σ)'
    
    return {
        'is_anomaly': bool(is_anomaly),
        'z_score': float(z_score),
        'reason': reason,
        'current_load': int(recent_load),
        'mean_load': float(mean_load),
        'std_load': float(std_load)
    }

