"""
Forecast & Scale Dashboard - ML-powered traffic prediction and auto-scaling
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

from model.params import get_params
import views.settings as settings

# Import predictor utilities
try:
    from traffic_monitor.server.utils import Predictor
except ImportError:
    Predictor = None
    st.warning("⚠️ ML Server utilities not available. Forecast features may be limited.")


class ForecastScaleDashboard:
    def __init__(self):
        self.logs_buffer = []
        self.max_buffer_size = settings.MAX_LOGS_DISPLAY
        self.config = get_params("baseline")
        
        # Load padding data from cached CSV
        self._load_padding_data()
        
        # Initialize predictor
        if Predictor:
            try:
                self.predictor = Predictor(model_type="baseline")
            except Exception as e:
                st.error(f"Failed to load predictor: {e}")
                self.predictor = None
        else:
            self.predictor = None
    
    def _load_padding_data(self):
        """Load padding data from cached CSV file"""
        try:
            cache_path = Path('cache/padding_train_15min.csv')
            if cache_path.exists():
                self.padding_df_resampled = pd.read_csv(cache_path)
                self.padding_df_resampled['timestamp'] = pd.to_datetime(self.padding_df_resampled['timestamp'])
                st.sidebar.success(f"✅ Loaded {len(self.padding_df_resampled)} 15-min intervals from cache")
            else:
                st.sidebar.error(f"❌ Cache file not found: {cache_path}")
                # Create empty padding dataframe as fallback
                self.padding_df_resampled = pd.DataFrame({
                    'timestamp': [],
                    'request_count': [],
                    'bytes': []
                })
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load padding data: {e}")
            # Create empty padding dataframe as fallback
            self.padding_df_resampled = pd.DataFrame({
                'timestamp': [],
                'request_count': [],
                'bytes': []
            })
    
    def add_log(self, log):
        """Add a new log entry to the buffer"""
        self.logs_buffer.append(log)
        if len(self.logs_buffer) > self.max_buffer_size:
            self.logs_buffer.pop(0)
    
    def clear_logs(self):
        """Clear all logs from buffer"""
        self.logs_buffer = []
    
    def _prepare_data_for_forecast(self):
        """
        Prepare resampled data with padding for forecasting
        Returns: DataFrame with at least 64 15-min intervals (using padding if needed)
        """
        if not self.logs_buffer:
            return None, "No data available"
        
        # Convert logs to DataFrame
        df = pd.DataFrame(self.logs_buffer)
        if 'timestamp' not in df.columns:
            return None, "Missing timestamp column"
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['request_count'] = 1  # Each log is 1 request
        
        # Resample to 15-minute intervals
        df_resampled = df.set_index('timestamp').resample('15min').agg({
            'request_count': 'sum',
            'bytes': 'sum'
        }).reset_index()
        
        # Check if we have at least 1 15-min interval
        if len(df_resampled) < 1:
            return None, "Need at least 1 complete 15-min interval"
        
        # Pad with training data if needed (require 64 intervals for model)
        if len(df_resampled) < self.config.train_window:
            needed = self.config.train_window - len(df_resampled)
            
            if len(self.padding_df_resampled) >= needed:
                padding_slice = self.padding_df_resampled.tail(needed).copy()
                
                # Shift padding timestamps to align before current data
                time_shift = df_resampled['timestamp'].iloc[0] - padding_slice['timestamp'].iloc[-1] - timedelta(minutes=15)
                padding_slice['timestamp'] = padding_slice['timestamp'] + time_shift
                
                # Combine padding + current data
                df_combined = pd.concat([padding_slice, df_resampled]).reset_index(drop=True)
                
                return df_combined, None
            else:
                return None, f"Insufficient padding data. Need {needed} intervals but only have {len(self.padding_df_resampled)}"
        
        return df_resampled, None
    
    def _generate_forecast(self, df_resampled, forecast_horizon=10):
        """
        Generate traffic forecast using trained model
        
        Args:
            df_resampled: DataFrame with 15-min aggregated data
            forecast_horizon: Number of 15-min intervals to predict
        
        Returns:
            tuple: (forecast_values, future_timestamps, confidence_interval)
        """
        if self.predictor is None:
            return None, None, None
        
        try:
            # Predict using trained model
            forecast_values = self.predictor.predict(
                df_resampled['request_count'].values,
                predict_horizon=forecast_horizon
            )
            
            # Generate future timestamps
            last_timestamp = df_resampled['timestamp'].iloc[-1]
            future_timestamps = [
                last_timestamp + timedelta(minutes=15*(i+1))
                for i in range(forecast_horizon)
            ]
            
            # Calculate confidence interval using recent standard deviation
            recent_std = df_resampled['request_count'].tail(30).std()
            confidence_interval = {
                'lower': (forecast_values - 1.96 * recent_std).clip(min=0),
                'upper': (forecast_values + 1.96 * recent_std)
            }
            
            return forecast_values, future_timestamps, confidence_interval
            
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            return None, None, None
    
    def render_forecast_view(self, time_window_key="15min"):
        """Render the forecast dashboard with predictions"""
        st.subheader("🔮 Traffic Forecast Dashboard")
        
        if not self.logs_buffer:
            st.info("⏳ Waiting for data... Start the simulator to see predictions.")
            return
        
        # Prepare data
        df_prepared, error = self._prepare_data_for_forecast()
        
        if error:
            st.warning(f"⚠️ {error}")
            st.info("Continue collecting data to enable forecasting...")
            return
        
        # Display data preparation info
        col1, col2, col3 = st.columns(3)
        with col1:
            actual_intervals = len([ts for ts in df_prepared['timestamp'] 
                                   if ts >= pd.Timestamp(self.logs_buffer[0]['timestamp'])])
            st.metric("Actual 15-min Intervals", actual_intervals)
        with col2:
            padded_intervals = len(df_prepared) - actual_intervals
            st.metric("Padded Intervals (from training)", padded_intervals)
        with col3:
            st.metric("Total Intervals for Model", len(df_prepared))
        
        # Generate forecast
        forecast_horizon = 10  # Predict next 10 x 15-min intervals
        forecast_values, future_timestamps, confidence_interval = self._generate_forecast(
            df_prepared, forecast_horizon
        )
        
        if forecast_values is None:
            st.error("❌ Failed to generate forecast")
            return
        
        # Create visualization
        st.subheader("📈 Request Count: Historical + Predicted")
        
        fig = go.Figure()
        
        # Separate actual and padded historical data
        split_idx = len(df_prepared) - actual_intervals
        df_padded = df_prepared.iloc[:split_idx]
        df_actual = df_prepared.iloc[split_idx:]
        
        # Plot padded data (lighter color)
        if len(df_padded) > 0:
            fig.add_trace(go.Scatter(
                x=df_padded['timestamp'],
                y=df_padded['request_count'],
                mode='lines',
                name='Historical (Padding)',
                line=dict(color='rgba(100, 100, 100, 0.4)', width=1, dash='dot'),
                showlegend=True
            ))
        
        # Plot actual historical data
        fig.add_trace(go.Scatter(
            x=df_actual['timestamp'],
            y=df_actual['request_count'],
            mode='lines+markers',
            name='Current Data',
            line=dict(color='#00BFFF', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(0, 191, 255, 0.2)'
        ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            marker=dict(size=8, symbol='star')
        ))
        
        # Plot confidence interval
        if confidence_interval:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=confidence_interval['upper'],
                mode='lines',
                name='95% Confidence Upper',
                line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
                showlegend=False
            ))
            
            # Lower bound
            fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=confidence_interval['lower'],
                mode='lines',
                name='95% Confidence',
                line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.2)'
            ))
        
        # Add vertical line to separate actual and forecast
        if len(df_actual) > 0:
            last_actual_time = df_actual['timestamp'].iloc[-1]
            # Use shape instead of add_vline to avoid timestamp issues
            fig.add_shape(
                type="line",
                x0=last_actual_time, x1=last_actual_time,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="yellow", width=2, dash="dash")
            )
            # Add annotation separately
            fig.add_annotation(
                x=last_actual_time,
                y=1,
                yref="paper",
                text="Now",
                showarrow=False,
                yshift=10,
                font=dict(color="yellow", size=12)
            )
        
        fig.update_layout(
            title='Traffic Forecast (15-minute intervals)',
            xaxis_title='Time',
            yaxis_title='Request Count per 15 min',
            height=500,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Forecast statistics
        st.subheader("📊 Forecast Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_load = int(df_actual['request_count'].iloc[-1]) if len(df_actual) > 0 else 0
            st.metric("Current Load (15 min)", current_load)
        
        with col2:
            avg_forecast = float(np.mean(forecast_values))
            change_pct = ((avg_forecast - current_load) / current_load * 100) if current_load > 0 else 0
            st.metric("Avg Predicted Load", f"{avg_forecast:.0f}", f"{change_pct:+.1f}%")
        
        with col3:
            max_forecast = float(np.max(forecast_values))
            st.metric("Peak Predicted Load", f"{max_forecast:.0f}")
        
        with col4:
            time_to_peak = future_timestamps[np.argmax(forecast_values)]
            minutes_to_peak = int((time_to_peak - df_actual['timestamp'].iloc[-1]).total_seconds() / 60)
            st.metric("Time to Peak", f"{minutes_to_peak} min")
        
        # Show forecast details table
        with st.expander("📋 Detailed Forecast Values"):
            forecast_df = pd.DataFrame({
                'Time': future_timestamps,
                'Predicted Requests': forecast_values.round(0).astype(int),
                'Lower Bound (95%)': confidence_interval['lower'].round(0).astype(int) if confidence_interval else None,
                'Upper Bound (95%)': confidence_interval['upper'].round(0).astype(int) if confidence_interval else None
            })
            st.dataframe(forecast_df, width='stretch')
        
        # Insights and recommendations
        st.subheader("💡 Insights & Recommendations")
        
        # Trend analysis
        if avg_forecast > current_load * 1.2:
            st.warning("📈 **Traffic is expected to increase significantly (+20%)** in the next 2.5 hours. Consider scaling up resources.")
        elif avg_forecast < current_load * 0.8:
            st.info("📉 **Traffic is expected to decrease (-20%)** in the next 2.5 hours. Potential opportunity to scale down.")
        else:
            st.success("➡️ **Traffic is expected to remain stable** in the next 2.5 hours. Current capacity should be sufficient.")
        
        # Capacity planning
        avg_per_minute = avg_forecast / 15  # Convert 15-min to per-minute rate
        max_per_minute = max_forecast / 15
        
        st.info(f"""
        **Capacity Planning:**
        - Average expected rate: **{avg_per_minute:.1f} requests/min**
        - Peak expected rate: **{max_per_minute:.1f} requests/min**
        - Forecast horizon: **{forecast_horizon * 15} minutes** (next 2.5 hours)
        """)
        
        # Scaling Recommendation
        self._render_scaling_recommendation(df_actual, avg_per_minute, max_per_minute, current_load)
    
    def _render_scaling_recommendation(self, df_actual, avg_per_minute, max_per_minute, current_load):
        """Render auto-scaling recommendation based on forecast"""
        st.subheader("⚙️ Auto-Scaling Recommendation")
        
        # Configuration constants
        MAX_CAPACITY_PER_INSTANCE = 1000  # requests per minute
        UNIT_COST_PER_INSTANCE = 0.05  # $ per hour per instance
        SCALE_UP_THRESHOLD = 0.75  # Scale up at 75% capacity
        SCALE_DOWN_THRESHOLD = 0.30  # Scale down below 30% capacity
        ANOMALY_THRESHOLD_SIGMA = 3.0
        
        # Initialize global state (simulate server state)
        if 'current_instances' not in st.session_state:
            st.session_state.current_instances = 1
        if 'last_scale_time' not in st.session_state:
            st.session_state.last_scale_time = datetime.now() - timedelta(minutes=10)
        
        current_instances = st.session_state.current_instances
        
        # Calculate required instances based on max predicted load
        required_instances = max(1, int(np.ceil(max_per_minute / MAX_CAPACITY_PER_INSTANCE)))
        
        # Calculate capacity utilization
        current_capacity = current_instances * MAX_CAPACITY_PER_INSTANCE
        current_utilization = (current_load / 15) / current_capacity if current_capacity > 0 else 0
        predicted_utilization = avg_per_minute / current_capacity if current_capacity > 0 else 0
        
        # Anomaly detection
        try:
            from traffic_monitor.server.utils import detect_anomaly
            anomaly_result = detect_anomaly(df_actual, threshold_sigma=ANOMALY_THRESHOLD_SIGMA)
            is_anomaly = anomaly_result['is_anomaly']
        except:
            is_anomaly = False
            anomaly_result = {'reason': 'Anomaly detection unavailable', 'z_score': 0}
        
        # Cooldown logic
        time_since_last_scale = (datetime.now() - st.session_state.last_scale_time).total_seconds() / 60
        cooldown_remaining = max(0, 5 - time_since_last_scale)  # 5 min cooldown
        in_cooldown = cooldown_remaining > 0
        
        # Decision logic
        action = "MAINTAIN"
        reason = "Load stable within normal range"
        new_instances = current_instances
        action_color = "green"
        
        if is_anomaly and anomaly_result.get('z_score', 0) > ANOMALY_THRESHOLD_SIGMA:
            action = "WARNING"
            reason = f"⚠️ ANOMALY DETECTED: {anomaly_result['reason']}. Auto-scaling paused to prevent cost explosion."
            action_color = "red"
        elif in_cooldown:
            action = "COOLDOWN"
            reason = f"⏳ Hysteresis period active. {cooldown_remaining:.1f} minutes remaining."
            action_color = "orange"
        else:
            if predicted_utilization > SCALE_UP_THRESHOLD:
                action = "SCALE_OUT"
                new_instances = required_instances
                reason = f"📈 Predicted load ({avg_per_minute:.0f} req/min) exceeds {SCALE_UP_THRESHOLD*100:.0f}% capacity. Scale from {current_instances} to {new_instances} instances."
                action_color = "blue"
            elif predicted_utilization < SCALE_DOWN_THRESHOLD and current_instances > 1:
                action = "SCALE_IN"
                new_instances = max(1, required_instances)
                reason = f"📉 Predicted load ({avg_per_minute:.0f} req/min) below {SCALE_DOWN_THRESHOLD*100:.0f}% capacity. Scale from {current_instances} to {new_instances} instances."
                action_color = "blue"
            else:
                action = "MAINTAIN"
                reason = f"✅ Current capacity ({current_instances} instances) sufficient. Utilization: {predicted_utilization*100:.1f}%"
                action_color = "green"
        
        # Display recommendation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric("Current Instances", current_instances)
        
        with col2:
            # Color-coded action badge
            if action_color == "red":
                st.error(f"**Action: {action}**")
            elif action_color == "orange":
                st.warning(f"**Action: {action}**")
            elif action_color == "blue":
                st.info(f"**Action: {action}**")
            else:
                st.success(f"**Action: {action}**")
        
        with col3:
            st.metric("Recommended Instances", new_instances, delta=new_instances - current_instances)
        
        st.write(reason)
        
        # Detailed metrics
        with st.expander("📊 Detailed Scaling Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Capacity Analysis**")
                st.write(f"- Current capacity: {current_capacity} req/min")
                st.write(f"- Current utilization: {current_utilization*100:.1f}%")
                st.write(f"- Predicted utilization: {predicted_utilization*100:.1f}%")
                st.write(f"- Required instances: {required_instances}")
            
            with col2:
                st.write("**Cost Estimation**")
                hourly_cost_current = current_instances * UNIT_COST_PER_INSTANCE
                hourly_cost_new = new_instances * UNIT_COST_PER_INSTANCE
                st.write(f"- Current hourly cost: ${hourly_cost_current:.3f}")
                st.write(f"- Projected hourly cost: ${hourly_cost_new:.3f}")
                st.write(f"- Daily cost (projected): ${hourly_cost_new * 24:.2f}")
                st.write(f"- Monthly cost (projected): ${hourly_cost_new * 24 * 30:.2f}")
        
        # Action buttons (simulation)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔼 Scale Up (+1)", disabled=in_cooldown, width='stretch'):
                st.session_state.current_instances += 1
                st.session_state.last_scale_time = datetime.now()
                st.success(f"Scaled up to {st.session_state.current_instances} instances")
                st.rerun()
        
        with col2:
            if st.button("✅ Apply Recommendation", disabled=(action in ["MAINTAIN", "COOLDOWN", "WARNING"]), width='stretch'):
                st.session_state.current_instances = new_instances
                st.session_state.last_scale_time = datetime.now()
                st.success(f"Applied recommendation: {new_instances} instances")
                st.rerun()
        
        with col3:
            if st.button("🔽 Scale Down (-1)", disabled=(current_instances <= 1 or in_cooldown), width='stretch'):
                st.session_state.current_instances = max(1, current_instances - 1)
                st.session_state.last_scale_time = datetime.now()
                st.success(f"Scaled down to {st.session_state.current_instances} instances")
                st.rerun()
    
    def render(self, logs_buffer=None):
        """Main render method for the dashboard"""
        # Sync logs buffer if provided
        if logs_buffer is not None:
            self.logs_buffer = logs_buffer
        
        # Render forecast view
        self.render_forecast_view(time_window_key="15min")
