"""
Forecast & Scale Dashboard - ML-powered traffic prediction and auto-scaling
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.params import get_params
from traffic_monitor.views import settings
from traffic_monitor.modules.predict_simulator import PredictDataLoader

class ForecastScaleDashboard:
    def __init__(self):
        self.forecast_buffer = []  # Store predictions over time
        self.max_buffer_size = 1000
        self.config = get_params("enhanced")
        self.api_url = "http://localhost:8000"
        self.simulator = None
        self.is_forecasting = False
        self.data_buffer = []  # Buffer for incoming data
        self.anomaly_info = None  # Store anomaly detection results
        self.scaling_info = None  # Store scaling recommendations
        self.cost_info = None  # Store cost estimates
        
    def add_data(self, data):
        """Add incoming data to buffer"""
        self.data_buffer.append(data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]
    
    def clear_data(self):
        """Clear all data buffers"""
        self.forecast_buffer = []
        self.data_buffer = []
        self.is_forecasting = False
        
    def clear_forecast(self):
        """Clear forecast data"""
        self.forecast_buffer = []
        self.is_forecasting = False
    
    def start_simulator(self, speed_multiplier=5.0):
        """Initialize the predict simulator"""
        if self.simulator is None:
            self.simulator = PredictDataLoader(
                file_path='./cache/simulated_data.csv',
                speed_multiplier=speed_multiplier,
                shuffle=False,
                loop=False
            )
            self.is_forecasting = True
    
    def stop_simulator(self):
        """Stop the simulator"""
        self.simulator = None
        self.is_forecasting = False
    
    def stream_next_forecast(self, forecast_steps=24):
        """Stream next batch of data and make predictions"""
        if self.simulator is None:
            return None, "Simulator not started"
        
        try:
            # Get next batch from simulator
            batch = self.simulator.get_next_batch(batch_size=forecast_steps)
            
            if not batch:
                self.is_forecasting = False
                return None, "End of data stream"
            
            # Add batch data to buffer
            self.add_data(batch)
            
            # Need enough historical data for prediction (lookback window)
            if len(self.data_buffer) < self.config.train_window:
                return None, f"Accumulating data... {len(self.data_buffer)}/{self.config.train_window}"
            
            # Call API for predictions when we have enough data
            data = self.data_buffer[-self.config.train_window:]
            result, error = self.call_forecast_api(data, forecast_steps)
            
            if error:
                return None, error
            
            # Store predictions
            self.forecast_buffer = result
            
            return True, None
            
        except StopIteration:
            self.is_forecasting = False
            return None, "End of data stream"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def call_forecast_api(self, data, forecast_steps=24):
        """Call ML server API for forecast"""
        try:
            # Check API health
            health_response = requests.get(f"{self.api_url}/health", timeout=2)
            if health_response.status_code != 200:
                return None, "ML Server is not healthy"
            
            # Convert data to proper format (list of dicts)
            if isinstance(data, list) and len(data) > 0:
                if not isinstance(data[0], dict):
                    return None, "Data format error: Expected list of dictionaries"
                api_data = data
            else:
                return None, "Data format error: Expected non-empty list"
            
            # Call forecast endpoint
            response = requests.post(
                f"{self.api_url}/forecast",
                json={"data": api_data, "forecast_steps": forecast_steps},
                timeout=30
            )
            
            if response.status_code == 200:
                # Get the List[Dict]: predictions with timestamps from response json
                preds = response.json().get("predictions", [])
                timestamps = response.json().get("timestamps", [])
                result = [{"timestamp": ts, "prediction": pred} for ts, pred in zip(timestamps, preds)]
                return result, None
            else:
                return None, f"API Error: {response.status_code} - {response.text}"
        
        except requests.exceptions.ConnectionError:
            return None, "Cannot connect to ML Server. Please start it with: uvicorn traffic_monitor.server.ml_server:app --reload --port 8000"
        except Exception as e:
            return None, f"Error calling API: {str(e)}"
    
    def call_anomaly_api(self, data):
        """Call anomaly detection API"""
        try:
            response = requests.post(
                f"{self.api_url}/anomaly-detect",
                json={"data": data},
                timeout=10
            )
            if response.status_code == 200:
                return response.json(), None
            return None, f"API Error: {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    def call_scaling_api(self, predictions):
        """Call scaling recommendation API"""
        try:
            response = requests.post(
                f"{self.api_url}/recommend-scaling",
                json={"data": predictions},
                timeout=10
            )
            if response.status_code == 200:
                return response.json(), None
            return None, f"API Error: {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    def call_cost_api(self, predictions):
        """Call cost estimation API"""
        try:
            response = requests.post(
                f"{self.api_url}/cost-estimate",
                json={"data": predictions},
                timeout=10
            )
            if response.status_code == 200:
                return response.json(), None
            return None, f"API Error: {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    def plot_forecast(self):
        """
        Plot the data buffer (historical/current) and future predictions separately
        Historical data in green, Future predictions in red (not overlapping)
        """
        if len(self.data_buffer) == 0:
            return None
            
        fig = go.Figure()
        
        # Plot historical/current data (actual traffic) in green
        known_df = pd.DataFrame(self.data_buffer)
        if 'timestamp' in known_df.columns and 'target' in known_df.columns:
            known_df['timestamp'] = pd.to_datetime(known_df['timestamp'])
            # Show last 200 points for better visibility
            display_df = known_df.tail(200)
            fig.add_trace(go.Scatter(
                x=display_df['timestamp'],
                y=display_df['target'],
                mode='lines+markers',
                name='Historical Traffic',
                line=dict(color='#2ECC71', width=2),
                marker=dict(size=4, color='#2ECC71')
            ))
            
            # Get the last timestamp for connecting the forecast
            last_timestamp = display_df['timestamp'].iloc[-1]
            last_value = display_df['target'].iloc[-1]
            
            # Highlight anomaly if detected
            if self.anomaly_info and self.anomaly_info.get('is_anomaly'):
                fig.add_trace(go.Scatter(
                    x=[last_timestamp],
                    y=[last_value],
                    mode='markers',
                    name='⚠️ Anomaly',
                    marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='darkred'))
                ))
        
        # Plot future predictions (not overlapping with actual data)
        if len(self.forecast_buffer) > 0:
            pred_df = pd.DataFrame(self.forecast_buffer)
            if 'timestamp' in pred_df.columns and 'prediction' in pred_df.columns:
                pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
                # Remove duplicates and sort
                pred_df = pred_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # Only show predictions that are AFTER the current data
                if 'last_timestamp' in locals():
                    pred_df = pred_df[pred_df['timestamp'] > last_timestamp]
                
                if len(pred_df) > 0:
                    # Add connecting point from last actual to first prediction
                    first_pred_time = pred_df['timestamp'].iloc[0]
                    first_pred_val = pred_df['prediction'].iloc[0]
                    
                    # Connection line
                    fig.add_trace(go.Scatter(
                        x=[last_timestamp, first_pred_time],
                        y=[last_value, first_pred_val],
                        mode='lines',
                        name='Transition',
                        line=dict(color='#FFA500', width=1, dash='dot'),
                        showlegend=False
                    ))
                    
                    # Future predictions
                    fig.add_trace(go.Scatter(
                        x=pred_df['timestamp'],
                        y=pred_df['prediction'],
                        mode='lines+markers',
                        name='Future Forecast',
                        line=dict(color='#E74C3C', width=2, dash='dash'),
                        marker=dict(size=5, color='#E74C3C', symbol='diamond')
                    ))
                    
                    # Add anomaly threshold bands if available
                    if self.anomaly_info:
                        mean = self.anomaly_info.get('mean_load', 0)
                        std = self.anomaly_info.get('std_load', 0)
                        threshold = self.anomaly_info.get('threshold', 2.5)
                        
                        upper_bound = mean + threshold * std
                        lower_bound = mean - threshold * std
                        
                        # Add threshold bands
                        all_timestamps = list(display_df['timestamp']) + list(pred_df['timestamp'])
                        
                        fig.add_trace(go.Scatter(
                            x=all_timestamps,
                            y=[upper_bound] * len(all_timestamps),
                            mode='lines',
                            name='Upper Threshold',
                            line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dot'),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=all_timestamps,
                            y=[lower_bound] * len(all_timestamps),
                            mode='lines',
                            name='Lower Threshold',
                            line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dot'),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.05)',
                            showlegend=False
                        ))
                    
                    # Show statistics
                    pred_mean = pred_df['prediction'].mean()
                    pred_std = pred_df['prediction'].std()
                    st.caption(f"📊 Forecast Stats: Mean={pred_mean:.2f}, Std={pred_std:.2f}, Min={pred_df['prediction'].min():.2f}, Max={pred_df['prediction'].max():.2f}")
        
        fig.update_layout(
            title='Traffic Forecast - Historical Data → Future Predictions',
            xaxis_title='Timestamp',
            yaxis_title='Traffic Volume (Requests)',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    
    def render(self):
        """Render the forecast dashboard"""
        st.title("🔮 Traffic Forecast & Auto-Scaling")
        st.markdown("ML-powered traffic prediction using Seq2Seq model with streaming inference")
        
        # Check if simulator is running (controlled by app.py)
        is_active = st.session_state.get('predict_simulator_running', False)
        
        # Status section
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            if is_active:
                st.metric("Status", "🟢 Streaming")
            else:
                st.metric("Status", "⚪ Idle")
        with status_col2:
            st.metric("Data Points", len(self.data_buffer))
        with status_col3:
            st.metric("Predictions", len(self.forecast_buffer))
        
        # Show data range info
        if len(self.data_buffer) > 0:
            known_df = pd.DataFrame(self.data_buffer)
            if 'target' in known_df.columns:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Actual - Mean", f"{known_df['target'].mean():.2f}")
                with col2:
                    st.metric("Actual - Min", f"{known_df['target'].min():.2f}")
                with col3:
                    st.metric("Actual - Max", f"{known_df['target'].max():.2f}")
        
        # Main content area
        if is_active:
            # Check if we have enough data to make predictions
            if len(self.data_buffer) >= self.config.train_window:
                # Show control buttons
                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    auto_update = st.checkbox("Auto-update", value=True, key="auto_update_preds")
                with col3:
                    if st.button("🔄 Refresh Now", key="refresh_predictions"):
                        st.session_state['force_repredict'] = True
                        st.rerun()
                
                # Auto-update predictions as data streams (every 20 new data points to allow model inference time)
                should_update = (
                    len(self.forecast_buffer) == 0 or 
                    st.session_state.get('force_repredict', False) or
                    (auto_update and len(self.data_buffer) % 20 == 0)
                )
                
                # Try to make predictions
                if should_update:
                    with st.spinner("Calling ML API for predictions..."):
                        # Send at least train_window + forecast_horizon data for rolling inference
                        # If we have more, send up to train_window + 100 for better context
                        required_len = self.config.train_window + settings.FORECAST_HORIZON
                        max_len = self.config.train_window + 200
                        data_len = min(len(self.data_buffer), max_len)
                        data_len = max(data_len, required_len) if len(self.data_buffer) >= required_len else self.config.train_window
                        data = self.data_buffer[-data_len:]
                        result, error = self.call_forecast_api(data, settings.FORECAST_HORIZON)
                        
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            # Replace predictions (showing future forecast from current point)
                            self.forecast_buffer = result
                            
                            st.session_state['force_repredict'] = False
                            st.success(f"✅ Future forecast updated! ({len(result)} points ahead)")
                            
                            # Call additional APIs for anomaly, scaling, and cost
                            if len(self.data_buffer) > 10:
                                anomaly_result, _ = self.call_anomaly_api(self.data_buffer[-50:])
                                if anomaly_result:
                                    self.anomaly_info = anomaly_result
                            
                            if len(self.forecast_buffer) > 0:
                                scaling_result, _ = self.call_scaling_api(self.forecast_buffer)
                                if scaling_result:
                                    self.scaling_info = scaling_result
                                
                                cost_result, _ = self.call_cost_api(self.forecast_buffer)
                                if cost_result:
                                    self.cost_info = cost_result
            else:
                progress = len(self.data_buffer) / self.config.train_window
                st.info(f"📊 Accumulating data... {len(self.data_buffer)}/{self.config.train_window} ({progress*100:.1f}%)")
                st.progress(progress)
            
            st.divider()
            
            # Create a placeholder for the plot to avoid duplicates
            plot_placeholder = st.empty()
            
            # Show plot
            if len(self.forecast_buffer) > 0:
                fig = self.plot_forecast()
                if fig:
                    with plot_placeholder:
                        st.plotly_chart(fig, width='stretch', key='forecast_with_predictions')
            elif len(self.data_buffer) > 0:
                # Show only actual data if no predictions yet
                fig = self._plot_data_only()
                if fig:
                    with plot_placeholder:
                        st.plotly_chart(fig, width='stretch', key='forecast_data_only')
            
            # Show insights section
            if self.anomaly_info or self.scaling_info or self.cost_info:
                st.divider()
                st.subheader("📊 Insights & Recommendations")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🔍 Anomaly Detection")
                    if self.anomaly_info:
                        if self.anomaly_info.get('is_anomaly'):
                            st.error(self.anomaly_info.get('reason'))
                            st.metric("Z-Score", f"{self.anomaly_info.get('z_score', 0):.2f}")
                        else:
                            st.success("✅ Normal traffic pattern")
                            st.metric("Z-Score", f"{self.anomaly_info.get('z_score', 0):.2f}")
                        st.caption(f"Current: {self.anomaly_info.get('current_load', 0):.0f} | Mean: {self.anomaly_info.get('mean_load', 0):.0f}")
                    else:
                        st.info("Waiting for data...")
                
                with col2:
                    st.markdown("#### ⚙️ Scaling Recommendation")
                    if self.scaling_info:
                        action = self.scaling_info.get('action')
                        if action == "SCALE_UP":
                            st.warning(f"🔼 {action}")
                        elif action == "SCALE_DOWN":
                            st.info(f"🔽 {action}")
                        else:
                            st.success(f"✅ {action}")
                        
                        st.metric(
                            "Recommended Instances", 
                            self.scaling_info.get('recommended_instances'),
                            delta=self.scaling_info.get('recommended_instances') - self.scaling_info.get('current_instances')
                        )
                        st.caption(self.scaling_info.get('reason'))
                    else:
                        st.info("Waiting for predictions...")
                
                with col3:
                    st.markdown("#### 💰 Cost Estimation")
                    if self.cost_info:
                        st.metric("Hourly Cost", f"${self.cost_info.get('recommended_cost_per_hour', 0):.2f}")
                        st.metric("Daily Cost", f"${self.cost_info.get('daily_cost_estimate', 0):.2f}")
                        st.metric("Monthly Cost", f"${self.cost_info.get('monthly_cost_estimate', 0):.2f}")
                        
                        diff = self.cost_info.get('cost_difference', 0)
                        if diff > 0:
                            st.caption(f"⬆️ +${diff:.2f}/hr vs current")
                        elif diff < 0:
                            st.caption(f"⬇️ ${diff:.2f}/hr vs current")
                    else:
                        st.info("Waiting for predictions...")
                        
        else:
            # Initial state - show instructions
            st.info("👈 Start the **Forecast Simulator** from the sidebar to begin streaming data and making predictions")
            
            with st.expander("ℹ️ How it works"):
                st.markdown("""
                1. **Start Simulator**: Click 'Start' under 'Forecast Simulator' in the sidebar
                2. **Data Accumulation**: The system will collect historical data for the prediction window
                3. **ML Predictions**: Once enough data is collected, predictions are made via the ML API
                4. **Real-time Updates**: The chart updates as new data streams in
                
                **Requirements**:
                - ML Server must be running on port 8000
                - Minimum {} data points needed for predictions
                """.format(self.config.train_window))
    
    def _plot_data_only(self):
        """Plot only the actual data without predictions"""
        if len(self.data_buffer) == 0:
            return None
            
        known_df = pd.DataFrame(self.data_buffer)
        known_df['timestamp'] = pd.to_datetime(known_df['timestamp'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=known_df['timestamp'],
            y=known_df['target'],
            mode='lines',
            name='Actual Traffic',
            line=dict(color='#2ECC71', width=2)
        ))
        
        fig.update_layout(
            title='Traffic Data - Actual',
            xaxis_title='Timestamp',
            yaxis_title='Traffic Volume',
            hovermode='x unified',
            template='plotly_dark',
            height=500
        )
        
        return fig
