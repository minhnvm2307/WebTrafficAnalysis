"""
Data pipeline for loading and preprocessing time series data.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path
from typing import Tuple, Optional, List

class LSTMDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        return x, y

class Seq2SeqDataset(Dataset):
    def __init__(self, df, lookback=168):
        self.lookback = lookback
        
        self.target_values = df['target'].values.astype(np.float32)
        self.all_features = df.drop(columns=['timestamp']).values.astype(np.float32)

    def __len__(self):
        return len(self.target_values) - self.lookback

    def __getitem__(self, idx):
        # Encoder input: Gồm target quá khứ + features quá khứ
        x_feat = self.all_features[idx : idx + self.lookback]
        y_feat = self.all_features[idx + self.lookback : idx + self.lookback + 1, 1:]
        
        y_target = self.target_values[idx + self.lookback : idx + self.lookback + 1]
        
        return {
            'x_features': torch.tensor(x_feat),
            'y_features': torch.tensor(y_feat),
            'y_target': torch.tensor(y_target)
        }

class ScalerWrapper:
    def __init__(self, scaler_path: str, model_type: str = "baseline"):
        self.model_type = model_type
        self.scaler = joblib.load(scaler_path)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.model_type == "baseline":
            return self.scaler.transform(data)
        elif self.model_type == "enhanced":
            data_log = np.log1p(data)
            return self.scaler.transform(data_log)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.model_type == "baseline":
            return self.scaler.inverse_transform(data)
        elif self.model_type == "enhanced":
            data_inv = self.scaler.inverse_transform(data)
            return np.expm1(data_inv)  # Inverse of log1p
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def load_data(filepath, chunk_time='15min') -> pd.DataFrame:
    """Load and prepare hourly time series"""
    df = pd.read_csv(filepath)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d/%b/%Y:%H:%M:%S %z"
    )
    df_series = df.set_index('timestamp').resample(chunk_time).size().reset_index(name='count')
    
    return df_series

def make_features(data: pd.DataFrame, lags=[1, 3, 6, 12, 24]):
    df = data.copy()
    
    df['count_log'] = np.log1p(df['count'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    hour = df['timestamp'].dt.hour
    dayofweek = df['timestamp'].dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
    
    df['is_weekend'] = (dayofweek >= 5).astype(float)
    
    for lag in lags:
        df[f'lag_{lag}'] = df['count_log'].shift(lag)
    
    df = df.dropna().reset_index(drop=True)
    
    return df.drop(columns=['count'])