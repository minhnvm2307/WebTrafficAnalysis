"""
Main Dashboard Component - Advanced Analytics and Monitoring
Includes real-time metrics, charts, and system monitoring
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

import views.settings as settings
from modules.utils import calculate_cpu_usage, calculate_memory_usage

class MainDashboard:
    def __init__(self):
        self.logs_buffer = []
        self.max_buffer_size = settings.MAX_LOGS_DASHBOARD
        
    def add_log(self, log):
        """Add a new log entry to the buffer"""
        self.logs_buffer.append(log)
        # Keep only the most recent logs
        if len(self.logs_buffer) > self.max_buffer_size:
            self.logs_buffer.pop(0)
    
    def add_logs_batch(self, logs):
        """Add multiple log entries at once"""
        for log in logs:
            self.add_log(log)
    
    def clear_logs(self):
        """Clear all logs from buffer"""
        self.logs_buffer = []
    
    def render_current_metrics(self):
        """Render current state metrics at the top"""
        if not self.logs_buffer:
            st.info("⏳ Waiting for data...")
            return
        
        # Calculate current metrics
        df = pd.DataFrame(self.logs_buffer)
        
        # Get last 60 seconds of data for "current" metrics
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            latest_time = df['timestamp'].max()
            one_min_ago = latest_time - timedelta(seconds=60)
            recent_df = df[df['timestamp'] >= one_min_ago]
        else:
            recent_df = df.tail(60)
        
        # Calculate metrics
        total_requests = len(self.logs_buffer)
        current_rps = len(recent_df) / 60 if len(recent_df) > 0 else 0
        
        if 'bytes' in recent_df.columns:
            current_bytes = recent_df['bytes'].sum()
            current_bps = current_bytes / 60 if len(recent_df) > 0 else 0
        else:
            current_bytes = 0
            current_bps = 0
        
        if 'status' in recent_df.columns:
            error_count = len(recent_df[recent_df['status'].astype(str).str.match(r'^[45]')])
            error_rate = (error_count / len(recent_df) * 100) if len(recent_df) > 0 else 0
        else:
            error_count = 0
            error_rate = 0
        
        # Simulate system metrics
        cpu_usage = self._calculate_cpu_usage(len(recent_df), 60)
        memory_usage = self._calculate_memory_usage(total_requests)
        
        # Display metrics in columns
        st.subheader("Current State (Last 60 seconds)")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Requests/sec",
                f"{current_rps:.2f}",
                delta=None,
                help="Current request rate per second"
            )
        
        with col2:
            # Convert bytes to appropriate unit
            if current_bps > 1024*1024:
                bytes_display = f"{current_bps/(1024*1024):.2f} MB/s"
            elif current_bps > 1024:
                bytes_display = f"{current_bps/1024:.2f} KB/s"
            else:
                bytes_display = f"{current_bps:.2f} B/s"
            
            st.metric(
                "Bandwidth",
                bytes_display,
                delta=None,
                help="Current data transfer rate"
            )
        
        with col3:
            # Color code based on error rate
            error_delta_color = "inverse" if error_rate > 5 else "normal"
            st.metric(
                "Error Rate",
                f"{error_rate:.1f}%",
                delta=f"{error_count} errors",
                delta_color=error_delta_color,
                help="Percentage of 4xx and 5xx responses"
            )
        
        with col4:
            st.metric(
                "Total Requests",
                f"{total_requests:,}",
                delta=None,
                help="Total requests processed"
            )
        
        with col5:
            # CPU usage with color indicator
            cpu_color = "🟢" if cpu_usage < 50 else "🟡" if cpu_usage < 80 else "🔴"
            st.metric(
                f"{cpu_color} CPU Usage",
                f"{cpu_usage:.1f}%",
                delta=None,
                help="Simulated CPU usage"
            )
        
        with col6:
            # Memory usage with color indicator
            mem_color = "🟢" if memory_usage < 50 else "🟡" if memory_usage < 80 else "🔴"
            st.metric(
                f"{mem_color} Memory",
                f"{memory_usage:.1f}%",
                delta=None,
                help="Simulated memory usage"
            )
    
    def render_requests_per_second_chart(self):
        """Render real-time requests per second chart"""
        if not self.logs_buffer:
            st.warning("No data available for Requests/Second chart")
            return
        
        df = pd.DataFrame(self.logs_buffer)
        
        if 'timestamp' not in df.columns:
            st.warning("Timestamp data not available")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by second
        df_rps = df.set_index('timestamp').resample('1s').size().reset_index(name='requests')
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_rps['timestamp'],
            y=df_rps['requests'],
            mode='lines',
            name='Requests/sec',
            line=dict(color='#00D9FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.2)'
        ))
        
        # Add average line
        avg_rps = df_rps['requests'].mean()
        fig.add_hline(
            y=avg_rps,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Avg: {avg_rps:.2f} req/s",
            annotation_position="right"
        )
        
        fig.update_layout(
            title='Requests Per Second',
            xaxis_title='Time',
            yaxis_title='Requests/sec',
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Current state below chart
        current_rps = df_rps['requests'].iloc[-1] if not df_rps.empty else 0
        peak_rps = df_rps['requests'].max() if not df_rps.empty else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current", f"{current_rps:.0f} req/s")
        with col2:
            st.metric("Average", f"{avg_rps:.2f} req/s")
        with col3:
            st.metric("Peak", f"{peak_rps:.0f} req/s")
    
    def render_bytes_chart(self):
        """Render data transfer (bytes) chart"""
        if not self.logs_buffer:
            st.warning("No data available for Bytes chart")
            return
        
        df = pd.DataFrame(self.logs_buffer)
        
        if 'timestamp' not in df.columns or 'bytes' not in df.columns:
            st.warning("Required data not available for Bytes chart")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by second and sum bytes
        df_bytes = df.set_index('timestamp').resample('1s')['bytes'].sum().reset_index(name='bytes')
        
        # Convert to KB
        df_bytes['kb'] = df_bytes['bytes'] / 1024
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_bytes['timestamp'],
            y=df_bytes['kb'],
            mode='lines',
            name='KB/sec',
            line=dict(color='#00FF85', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 133, 0.2)'
        ))
        
        # Add average line
        avg_kb = df_bytes['kb'].mean()
        fig.add_hline(
            y=avg_kb,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Avg: {avg_kb:.2f} KB/s",
            annotation_position="right"
        )
        
        fig.update_layout(
            title='Data Transfer Rate',
            xaxis_title='Time',
            yaxis_title='KB/sec',
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Current state below chart
        current_kb = df_bytes['kb'].iloc[-1] if not df_bytes.empty else 0
        peak_kb = df_bytes['kb'].max() if not df_bytes.empty else 0
        total_mb = df_bytes['bytes'].sum() / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current", f"{current_kb:.2f} KB/s")
        with col2:
            st.metric("Average", f"{avg_kb:.2f} KB/s")
        with col3:
            st.metric("Total Transfer", f"{total_mb:.2f} MB")
    
    def render_error_requests_chart(self):
        """Render error requests chart (4xx and 5xx)"""
        if not self.logs_buffer:
            st.warning("No data available for Error chart")
            return
        
        df = pd.DataFrame(self.logs_buffer)
        
        if 'timestamp' not in df.columns or 'status' not in df.columns:
            st.warning("Required data not available for Error chart")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['status_str'] = df['status'].astype(str)
        
        # Classify status codes
        df['status_type'] = 'Success'
        df.loc[df['status_str'].str.match(r'^4'), 'status_type'] = '4xx Client Error'
        df.loc[df['status_str'].str.match(r'^5'), 'status_type'] = '5xx Server Error'
        
        # Group by second and status type
        df_errors = df.set_index('timestamp').groupby([pd.Grouper(freq='1s'), 'status_type']).size().reset_index(name='count')
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Add traces for each status type
        # Define beautiful color palette with gradients
        color_map = {
            'Success': {
                'line': '#00D4AA',
                'fill': 'rgba(0, 212, 170, 0.15)',
                'marker': '#00FFD1'
            },
            '4xx Client Error': {
                'line': '#FFB347',
                'fill': 'rgba(255, 179, 71, 0.15)',
                'marker': '#FFC870'
            },
            '5xx Server Error': {
                'line': '#FF6B6B',
                'fill': 'rgba(255, 107, 107, 0.15)',
                'marker': '#FF8E8E'
            }
        }
        
        for status_type in ['Success', '4xx Client Error', '5xx Server Error']:
            df_type = df_errors[df_errors['status_type'] == status_type]
            
            if not df_type.empty:
                # Sort by timestamp for proper line connection
                df_type = df_type.sort_values('timestamp')
                
                fig.add_trace(go.Scatter(
                    x=df_type['timestamp'],
                    y=df_type['count'],
                    mode='lines+markers',
                    name=status_type,
                    line=dict(
                        color=color_map[status_type]['line'],
                        width=2.5,
                        shape='spline',  # Smooth curved lines
                        smoothing=1.3
                    ),
                    marker=dict(
                        color=color_map[status_type]['marker'],
                        size=6,
                        symbol='circle',
                        line=dict(color='white', width=1)
                    ),
                    fill='tozeroy',
                    fillcolor=color_map[status_type]['fill'],
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Count: %{y}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text='Request Status Distribution',
                font=dict(size=18, color='#E0E0E0'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Time',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                showline=True,
                linewidth=1,
                linecolor='rgba(128, 128, 128, 0.3)'
            ),
            yaxis=dict(
                title='Number of Requests',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                showline=True,
                linewidth=1,
                linecolor='rgba(128, 128, 128, 0.3)'
            ),
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(128,128,128,0.3)',
                borderwidth=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Current state below chart
        total_requests = len(df)
        error_4xx = len(df[df['status_str'].str.match(r'^4')])
        error_5xx = len(df[df['status_str'].str.match(r'^5')])
        success = total_requests - error_4xx - error_5xx
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            success_rate = (success / total_requests * 100) if total_requests > 0 else 0
            st.metric("Success", f"{success_rate:.1f}%", f"{success} requests")
        with col2:
            error_4xx_rate = (error_4xx / total_requests * 100) if total_requests > 0 else 0
            st.metric("4xx Errors", f"{error_4xx_rate:.1f}%", f"{error_4xx} requests")
        with col3:
            error_5xx_rate = (error_5xx / total_requests * 100) if total_requests > 0 else 0
            st.metric("5xx Errors", f"{error_5xx_rate:.1f}%", f"{error_5xx} requests")
        with col4:
            total_error_rate = ((error_4xx + error_5xx) / total_requests * 100) if total_requests > 0 else 0
            st.metric("Total Error Rate", f"{total_error_rate:.1f}%")
    
    def render_system_metrics_chart(self):
        """Render CPU and Memory usage charts"""
        if not self.logs_buffer:
            st.warning("No data available for System Metrics")
            return
        
        df = pd.DataFrame(self.logs_buffer)
        
        if 'timestamp' not in df.columns:
            st.warning("Timestamp data not available")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by second to get request count
        df_seconds = df.set_index('timestamp').resample('1s').size().reset_index(name='requests')
        
        # Calculate CPU and Memory for each second
        cpu_values = []
        memory_values = []
        
        for idx, row in df_seconds.iterrows():
            # Get cumulative logs up to this point
            logs_until_now = idx + 1
            
            # Calculate metrics
            cpu = calculate_cpu_usage(self.logs_buffer, row['requests'], 1)
            memory = calculate_memory_usage(self.logs_buffer, logs_until_now * 10)  # Approximate
            
            cpu_values.append(cpu)
            memory_values.append(memory)
        
        df_seconds['cpu'] = cpu_values
        df_seconds['memory'] = memory_values
        
        # Create subplot with 2 rows
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'),
            vertical_spacing=0.12
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(
                x=df_seconds['timestamp'],
                y=df_seconds['cpu'],
                mode='lines',
                name='CPU %',
                line=dict(color='#FF6B6B', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.2)'
            ),
            row=1, col=1
        )
        
        # Add CPU threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                     annotation_text="High", row=1, col=1)
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(
                x=df_seconds['timestamp'],
                y=df_seconds['memory'],
                mode='lines',
                name='Memory %',
                line=dict(color='#4ECDC4', width=2),
                fill='tozeroy',
                fillcolor='rgba(78, 205, 196, 0.2)'
            ),
            row=2, col=1
        )
        
        # Add Memory threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                     annotation_text="High", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="CPU %", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="Memory %", range=[0, 100], row=2, col=1)
        
        fig.update_layout(
            height=600,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Current state below chart
        current_cpu = df_seconds['cpu'].iloc[-1] if not df_seconds.empty else 0
        avg_cpu = df_seconds['cpu'].mean() if not df_seconds.empty else 0
        current_memory = df_seconds['memory'].iloc[-1] if not df_seconds.empty else 0
        avg_memory = df_seconds['memory'].mean() if not df_seconds.empty else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cpu_status = "🟢 Normal" if current_cpu < 50 else "🟡 Moderate" if current_cpu < 80 else "🔴 High"
            st.metric("Current CPU", f"{current_cpu:.1f}%", cpu_status)
        with col2:
            st.metric("Avg CPU", f"{avg_cpu:.1f}%")
        with col3:
            mem_status = "🟢 Normal" if current_memory < 50 else "🟡 Moderate" if current_memory < 80 else "🔴 High"
            st.metric("Current Memory", f"{current_memory:.1f}%", mem_status)
        with col4:
            st.metric("Avg Memory", f"{avg_memory:.1f}%")
    
    def render(self):
        """Render the complete main dashboard"""
        st.header("📊 Main Dashboard - Real-time Analytics")
        
        # Current metrics at top
        self.render_current_metrics()
        
        st.divider()
        
        # Charts in 2 columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Request Rate")
            self.render_requests_per_second_chart()
            
            st.divider()
            
            st.subheader("📊 Error Distribution")
            self.render_error_requests_chart()
        
        with col2:
            st.subheader("💾 Data Transfer")
            self.render_bytes_chart()
            
            st.divider()
            
            st.subheader("🖥️ System Resources")
            self.render_system_metrics_chart()