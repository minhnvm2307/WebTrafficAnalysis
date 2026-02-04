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


def load_data(filepath, chunk_time='15min') -> pd.DataFrame:
    """Load and prepare hourly time series"""
    df = pd.read_csv(filepath)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d/%b/%Y:%H:%M:%S %z"
    )
    df_series = df.set_index('timestamp').resample(chunk_time).size().reset_index(name='count')
    
    return df_series

def make_features(data: pd.DataFrame, mode='train', scaler=None):
    df = data.copy()
    
    df['target_scaled'] = scaler.transform(df[['target']])
    
    # 3. Thông tin thời gian CƠ BẢN
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    for lag in [31, 32, 34, 35]:
        df[f'lag_{lag}'] = df['target_scaled'].shift(lag)
    
    # 6. Binary features cho giờ cao điểm
    df['is_peak_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # Fill NaN values
    df = df.bfill().fillna(0)
    
    return df, scaler