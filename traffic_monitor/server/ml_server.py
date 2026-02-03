from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from traffic_monitor.server.utils import Predictor, detect_anomaly
from model.params import get_params

app = FastAPI(title="Traffic Analysis ML Server", version="1.0.0")

# --- DATA MODELS ---
class MetricPoint(BaseModel):
    timestamp: str
    request_count: int
    bytes: int

class MetricBatch(BaseModel):
    history: List[MetricPoint]

class ForecastResponse(BaseModel):
    forecast: List[float]
    confidence_interval: Optional[dict] = None
    timestamp_range: Optional[List[str]] = None

class ScalingRecommendation(BaseModel):
    action: str  # SCALE_OUT, SCALE_IN, MAINTAIN, WARNING, COOLDOWN
    reason: str
    suggested_instances: int
    predicted_load_avg: float
    current_load: int
    is_anomaly: bool
    anomaly_details: dict
    estimated_hourly_cost: float
    cooldown_remaining: float
    confidence: float

# --- CONFIG ---
UNIT_COST_PER_INSTANCE = 0.05  # $ per hour per instance
UNIT_COST_PER_REQUEST = 0.0001  # $ per request
UNIT_COST_PER_GB = 0.10  # $ per GB transferred
MAX_CAPACITY_PER_INSTANCE = 1000  # requests per minute
SCALE_COOLDOWN = 5  # minutes (Hysteresis)
HISTORY_WINDOW = 60  # minutes
FORECAST_HORIZON = 10  # minutes ahead to forecast
ANOMALY_THRESHOLD_SIGMA = 3.0  # Z-score threshold
SCALE_UP_THRESHOLD = 0.75  # Scale up at 75% capacity
SCALE_DOWN_THRESHOLD = 0.30  # Scale down below 30% capacity


# Global state
last_scale_action_time = datetime.now() - timedelta(minutes=10)
current_instances = 1  # Start with 1 instance
predictor = Predictor(model_type="baseline")
config = get_params("baseline")

# Load padding data from cached CSV
cache_path = Path('cache/padding_train_15min.csv')
if cache_path.exists():
    padding_df_resampled = pd.read_csv(cache_path)
    padding_df_resampled['timestamp'] = pd.to_datetime(padding_df_resampled['timestamp'])
    print(f"✅ Loaded {len(padding_df_resampled)} 15-minute intervals from cache")
else:
    print(f"⚠️ Cache file not found: {cache_path}. Run cache generation script.")
    padding_df_resampled = pd.DataFrame({'timestamp': [], 'request_count': [], 'bytes': []})




# --- ENDPOINTS ---

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Traffic Analysis ML Server",
        "version": "1.0.0",
        "endpoints": ["/forecast", "/recommend-scaling", "/anomaly-detect", "/cost-estimate"]
    }

@app.post("/forecast", response_model=ForecastResponse)
def get_forecast(data: MetricBatch):
    """
    Traffic forecast using trained Seq2Seq model
    
    Returns predictions for the next FORECAST_HORIZON minutes based on
    the historical data provided in the request.
    """
    if len(data.history) < 900:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient history. Need at least 900 data points"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame([p.model_dump() for p in data.history])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    time_span_minutes = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 60
    
    if time_span_minutes < 15:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient time span. Need at least 15 minutes of data"
        )
    
    # Resample to 15-minute intervals (model expects 15-min aggregated data)
    df_resampled = df.set_index('timestamp').resample('15min').agg({
        'request_count': 'sum', 
        'bytes': 'sum'
    }).reset_index()
    
    if len(df_resampled) < config.train_window:
        # Take last N intervals from padding data to fill the gap
        needed = config.train_window - len(df_resampled)
        padding_slice = padding_df_resampled.tail(needed).copy()
        
        # Shift padding timestamps to align before current data
        time_shift = df_resampled['timestamp'].iloc[0] - padding_slice['timestamp'].iloc[-1] - timedelta(minutes=15)
        padding_slice['timestamp'] = padding_slice['timestamp'] + time_shift
        
        # Combine padding + current data
        df_resampled = pd.concat([padding_slice, df_resampled]).reset_index(drop=True)
    
    try:
        # Use trained model for inference (expects 15-min aggregated data)
        forecast_values = predictor.predict(df_resampled['request_count'].values, predict_horizon=FORECAST_HORIZON)
        
        # Generate future timestamps (15-min intervals)
        last_timestamp = df_resampled['timestamp'].iloc[-1]
        future_timestamps = [
            (last_timestamp + timedelta(minutes=15*(i+1))).isoformat()
            for i in range(FORECAST_HORIZON)
        ]
        
        # Calculate confidence interval (simple approach using recent std)
        recent_std = df_resampled['request_count'].tail(30).std()
        confidence_interval = {
            'lower': (forecast_values - 1.96 * recent_std).clip(min=0).tolist(),
            'upper': (forecast_values + 1.96 * recent_std).tolist()
        }
        
        return ForecastResponse(
            forecast=forecast_values.tolist(),
            confidence_interval=confidence_interval,
            timestamp_range=future_timestamps
        )
        
    except Exception as e:
        # Fallback to moving average
        print(f"⚠️ Model inference failed: {e}.")
        return None


# Run server: uvicorn traffic_monitor.server.ml_server:app --reload --port 8000
