from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path

from traffic_monitor.server.utils import Predictor
from model.params import get_params

app = FastAPI(title="Traffic Analysis ML Server", version="1.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global state
last_scale_action_time = datetime.now() - timedelta(minutes=10)
current_instances = 1  # Start with 1 instance
predictor = Predictor(model_type="enhanced")
config = get_params("enhanced")


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

class ForecastRequest(BaseModel):
    data: List[Dict]
    forecast_steps: Optional[int] = 24

@app.post("/forecast")
def get_forecast(request: ForecastRequest):
    """
    Traffic forecast using trained Seq2Seq model
    
    Returns predictions for the next FORECAST_HORIZON 
    Using rolling inference strategy.
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if len(df) < config.train_window:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data. Need at least {config.train_window} records"
            )
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'timestamp' column")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Run rolling inference
        predictions, actuals = run_rolling_inference(
            predictor.model, 
            df, 
            predictor.scaler,
            config, 
            device,
            request.forecast_steps
        )
        
        # Check if predictions were made
        if len(predictions) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Could not generate predictions. Need at least {config.train_window + request.forecast_steps} data points for rolling inference."
            )
        
        # Generate future timestamps (not from existing data)
        last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
        
        # Detect time interval from data
        if len(df) >= 2:
            time_diff = pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(df['timestamp'].iloc[-2])
        else:
            time_diff = pd.Timedelta(minutes=15)  # Default 15 min interval
        
        # Generate future timestamps
        future_timestamps = [last_timestamp + time_diff * (i + 1) for i in range(len(predictions))]
        
        return {
            "status_code": 200,
            "forecast_steps": request.forecast_steps,
            "predictions": predictions.tolist(),
            "actuals": actuals.tolist() if actuals is not None else [],
            "timestamps": [str(ts) for ts in future_timestamps]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_rolling_inference(model, df_test_featured, scaler, config, device, forecast_step):
    model.eval()
    
    # Prepare input data
    input_data = df_test_featured.drop(columns=['timestamp', 'target']).values
    y_features_full = df_test_featured.drop(columns=['timestamp', 'target', 'target_scaled']).values

    all_preds_scaled = []
    lookback = config.train_window
    
    num_samples = len(df_test_featured) - lookback - forecast_step
    
    # If we don't have enough data for rolling inference, do single prediction
    if num_samples <= 0:
        num_samples = 1
        # Use the last lookback window for prediction
        start_idx = len(df_test_featured) - lookback - forecast_step
        if start_idx < 0:
            start_idx = 0
            lookback = len(df_test_featured) - forecast_step
            if lookback < 1:
                lookback = len(df_test_featured)
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, max(1, forecast_step)):
            
            # Reset x_input to initial state of current window
            curr_window = input_data[start_idx : start_idx + lookback].copy()
            x_input = torch.tensor(curr_window, dtype=torch.float32).unsqueeze(0).to(device)
            
            current_window_preds = []
            
            # Forecast rolling for forecast_step steps
            for step in range(forecast_step):
                future_idx = start_idx + lookback + step
                
                # Check if we have actual future features, otherwise use last known features
                if future_idx < len(y_features_full):
                    y_feat = y_features_full[future_idx].reshape(1, 1, -1)
                    next_row = input_data[future_idx].copy()
                else:
                    # Use last available features for true future prediction
                    y_feat = y_features_full[-1].reshape(1, 1, -1)
                    next_row = input_data[-1].copy()
                
                y_features_input = torch.tensor(y_feat, dtype=torch.float32).to(device)
                
                # Forward pass
                pred_scaled = model(x_input, y_features_input)
                if isinstance(pred_scaled, tuple): 
                    pred_scaled = pred_scaled[0]
                
                val_scaled = pred_scaled.cpu().numpy().flatten()[0]
                current_window_preds.append(val_scaled)
                
                # Rolling logic: update input with prediction
                # Overwrite target_scaled with prediction
                next_row[0] = val_scaled 
                next_row_t = torch.tensor(next_row, dtype=torch.float32).view(1, 1, -1).to(device)
                x_input = torch.cat([x_input[:, 1:, :], next_row_t], dim=1)
            
            all_preds_scaled.append(current_window_preds)

    # Safety check: if no predictions were made, return empty arrays
    if len(all_preds_scaled) == 0 or all(len(p) == 0 for p in all_preds_scaled):
        return np.array([]), np.array([])
    
    preds_scaled_final = np.array(all_preds_scaled).flatten().reshape(-1, 1)
    
    # Inverse transform: create dummy array with same shape as scaler expected
    dummy_p = np.zeros((len(preds_scaled_final), scaler.n_features_in_))
    dummy_p[:, 0] = preds_scaled_final[:, 0]
    preds = scaler.inverse_transform(dummy_p)[:, 0]
    
    # Debug logging
    print(f"[DEBUG] Predictions - Scaled mean: {preds_scaled_final.mean():.4f}, Unscaled mean: {preds.mean():.2f}")
    print(f"[DEBUG] Predictions range: [{preds.min():.2f}, {preds.max():.2f}]")
    
    targets_raw = df_test_featured['target'].values[config.train_window : config.train_window + len(preds)]
    
    return preds, targets_raw

@app.get("/health")
def health_check():
    """Check if model is loaded and ready"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_type": predictor.model_type
    }

@app.post("/anomaly-detect")
def detect_anomaly(request: ForecastRequest):
    """
    Detect anomalies in traffic data using Z-score method
    """
    try:
        df = pd.DataFrame(request.data)
        
        if 'target' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing 'target' column")
        
        recent_load = df['target'].iloc[-1]
        mean_load = df['target'].mean()
        std_load = df['target'].std()
        
        threshold_sigma = 2.5
        
        if std_load == 0:
            z_score = 0.0
            is_anomaly = False
            reason = 'Zero variance in data'
        else:
            z_score = (recent_load - mean_load) / std_load
            is_anomaly = abs(z_score) > threshold_sigma
            
            if is_anomaly:
                if z_score > 0:
                    reason = f'⚠️ Traffic SPIKE detected! Z-score: {z_score:.2f}'
                else:
                    reason = f'⬇️ Traffic DROP detected! Z-score: {z_score:.2f}'
            else:
                reason = 'Normal traffic'
        
        return {
            "is_anomaly": bool(is_anomaly),
            "z_score": float(z_score),
            "reason": reason,
            "current_load": float(recent_load),
            "mean_load": float(mean_load),
            "std_load": float(std_load),
            "threshold": threshold_sigma
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-scaling")
def recommend_scaling(request: ForecastRequest):
    """
    Recommend scaling action based on predictions
    """
    try:
        predictions = request.data  # List of prediction values
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No predictions provided")
        
        # Extract prediction values
        if isinstance(predictions[0], dict):
            pred_values = [p.get('prediction', p.get('value', 0)) for p in predictions]
        else:
            pred_values = predictions
        
        pred_array = np.array(pred_values)
        max_pred = np.max(pred_array)
        mean_pred = np.mean(pred_array)
        
        # Simple scaling logic based on thresholds
        BASE_CAPACITY = 500  # requests per instance
        current_instances = max(1, int(np.ceil(mean_pred / BASE_CAPACITY)))
        recommended_instances = max(1, int(np.ceil(max_pred / BASE_CAPACITY)))
        
        # Scaling decision
        if recommended_instances > current_instances:
            action = "SCALE_UP"
            reason = f"Peak load ({max_pred:.0f}) exceeds current capacity"
        elif recommended_instances < current_instances:
            action = "SCALE_DOWN"
            reason = f"Predicted load ({mean_pred:.0f}) allows capacity reduction"
        else:
            action = "MAINTAIN"
            reason = "Current capacity is optimal"
        
        return {
            "action": action,
            "current_instances": current_instances,
            "recommended_instances": recommended_instances,
            "reason": reason,
            "max_predicted_load": float(max_pred),
            "mean_predicted_load": float(mean_pred),
            "capacity_per_instance": BASE_CAPACITY
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cost-estimate")
def estimate_cost(request: ForecastRequest):
    """
    Estimate costs based on scaling recommendations
    """
    try:
        predictions = request.data
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No predictions provided")
        
        # Extract prediction values
        if isinstance(predictions[0], dict):
            pred_values = [p.get('prediction', p.get('value', 0)) for p in predictions]
        else:
            pred_values = predictions
        
        pred_array = np.array(pred_values)
        max_pred = np.max(pred_array)
        mean_pred = np.mean(pred_array)
        
        # Cost parameters
        COST_PER_INSTANCE_HOUR = 0.10  # $0.10 per hour per instance
        BASE_CAPACITY = 500
        
        current_instances = max(1, int(np.ceil(mean_pred / BASE_CAPACITY)))
        recommended_instances = max(1, int(np.ceil(max_pred / BASE_CAPACITY)))
        
        # Calculate costs (assuming 1 hour forecast)
        current_cost = current_instances * COST_PER_INSTANCE_HOUR
        recommended_cost = recommended_instances * COST_PER_INSTANCE_HOUR
        cost_difference = recommended_cost - current_cost
        
        return {
            "current_instances": current_instances,
            "recommended_instances": recommended_instances,
            "current_cost_per_hour": round(current_cost, 2),
            "recommended_cost_per_hour": round(recommended_cost, 2),
            "cost_difference": round(cost_difference, 2),
            "daily_cost_estimate": round(recommended_cost * 24, 2),
            "monthly_cost_estimate": round(recommended_cost * 24 * 30, 2),
            "cost_per_instance_hour": COST_PER_INSTANCE_HOUR
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server: uvicorn traffic_monitor.server.ml_server:app --reload --port 8000
