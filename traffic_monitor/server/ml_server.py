from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = FastAPI()

# --- DATA MODELS ---
class MetricPoint(BaseModel):
    timestamp: str
    request_count: int
    bytes: int

class MetricBatch(BaseModel):
    history: List[MetricPoint]

# --- CONFIG ---
UNIT_COST_PER_INSTANCE = 0.05  # $ per hour
MAX_CAPACITY_PER_INSTANCE = 1000 # requests per minute
SCALE_COOLDOWN = 5 # minutes (Hysteresis)
last_scale_action_time = pd.Timestamp.now() - pd.Timedelta(minutes=10)

# --- LOGIC ---

@app.post("/forecast")
def get_forecast(data: MetricBatch):
    """
    Dự báo traffic trong 5 phút tới sử dụng Exponential Smoothing (phù hợp time-series ngắn hạn)
    """
    if len(data.history) < 10:
        return {"forecast": []}
    
    df = pd.DataFrame([vars(p) for p in data.history])
    df['request_count'] = pd.to_numeric(df['request_count'])
    
    # Simple Exponential Smoothing (Holt-Winters)
    try:
        model = ExponentialSmoothing(df['request_count'], trend='add', seasonal=None).fit()
        forecast_values = model.forecast(5).tolist() # Dự báo 5 bước tiếp theo
    except:
        # Fallback nếu data quá nhiễu hoặc ít
        forecast_values = [df['request_count'].mean()] * 5
        
    return {"forecast": forecast_values}

@app.post("/recommend-scaling")
def recommend_scaling(data: MetricBatch):
    """
    Quyết định Scaling dựa trên Forecast và Hysteresis (Cooldown)
    """
    global last_scale_action_time
    
    # 1. Lấy Forecast
    forecast_resp = get_forecast(data)
    predicted_load = np.mean(forecast_resp['forecast']) if forecast_resp['forecast'] else 0
    
    current_time = pd.Timestamp.now()
    
    # 2. Tính toán số instance cần thiết
    required_instances = max(1, int(np.ceil(predicted_load / MAX_CAPACITY_PER_INSTANCE)))
    
    # 3. Anomaly Detection (DDoS/Spike) - Z-Score
    df = pd.DataFrame([vars(p) for p in data.history])
    recent_load = df['request_count'].iloc[-1]
    mean_load = df['request_count'].mean()
    std_load = df['request_count'].std()
    
    is_anomaly = False
    if std_load > 0:
        z_score = (recent_load - mean_load) / std_load
        if z_score > 3: # Ngưỡng 3 sigma
            is_anomaly = True

    # 4. Logic Hysteresis
    action = "MAINTAIN"
    reason = "Load stable"
    
    time_since_last_scale = (current_time - last_scale_action_time).total_seconds() / 60
    
    if is_anomaly:
        action = "WARNING"
        reason = f"DDoS/Spike Detected! (Z-Score: {z_score:.2f}). Auto-scaling paused to prevent cost explosion."
    elif time_since_last_scale < SCALE_COOLDOWN:
        action = "COOLDOWN"
        reason = f"In hysteresis period. Wait {SCALE_COOLDOWN - time_since_last_scale:.1f} min."
    else:
        # Giả lập current capacity (trong thực tế sẽ lấy từ Cloud provider)
        # Ở đây ta giả định hệ thống đang chạy optimal - 1 hoặc + 1
        if predicted_load > (required_instances - 1) * MAX_CAPACITY_PER_INSTANCE * 0.8: # Threshold 80%
             action = "SCALE_OUT"
             reason = f"Forecast {predicted_load:.0f} reqs exceeds capacity."
             last_scale_action_time = current_time

    # 5. Cost Estimation
    estimated_cost = required_instances * UNIT_COST_PER_INSTANCE
    
    return {
        "action": action,
        "reason": reason,
        "suggested_instances": required_instances,
        "predicted_load_avg": predicted_load,
        "is_anomaly": is_anomaly,
        "estimated_hourly_cost": estimated_cost
    }

# Run server independently: uvicorn ml_server:app --reload --port 8000