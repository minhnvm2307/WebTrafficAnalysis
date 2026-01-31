"""
Log Tracer Component - Display raw logs in multiple formats
"""
import streamlit as st
import os
import sys
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import views.settings as settings

class LogTracer:
    def __init__(self):
        self.logs_buffer = []
        self.max_buffer_size = settings.MAX_LOGS_DISPLAY
        
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
    
    def render_terminal_view(self):
        """Render logs in terminal-style format"""
        st.subheader("🖥️ Terminal View (NASA Log Format)")
        
        if not self.logs_buffer:
            st.info("⏳ Waiting for logs...")
            return
        
        # Terminal style container
        terminal_html = f"""
        <div style="
            background-color: {settings.TERMINAL_BG_COLOR};
            color: {settings.TERMINAL_TEXT_COLOR};
            font-family: 'Courier New', monospace;
            padding: 15px;
            border-radius: 5px;
            height: 500px;
            overflow-y: auto;
            font-size: 12px;
            line-height: 1.4;
        ">
        """
        
        # Display logs in reverse order (newest first)
        for log in reversed(self.logs_buffer[-settings.TERMINAL_MAX_LINES:]):
            timestamp = log.get('timestamp', 'N/A')
            request_src = log.get('request_src', 'N/A')
            method = log.get('method', 'N/A')
            dest_path = log.get('dest_path', 'N/A')
            status = log.get('status', 'N/A')
            bytes_size = log.get('bytes', 'N/A')
            
            # Color code based on status
            status_str = str(status)
            if status_str.startswith('2'):
                color = settings.STATUS_COLORS['2xx']
            elif status_str.startswith('3'):
                color = settings.STATUS_COLORS['3xx']
            elif status_str.startswith('4'):
                color = settings.STATUS_COLORS['4xx']
            elif status_str.startswith('5'):
                color = settings.STATUS_COLORS['5xx']
            else:
                color = settings.TERMINAL_TEXT_COLOR
            
            log_line = f'<span style="color: #888;">[{timestamp}]</span> '
            log_line += f'<span style="color: #00BFFF;">{request_src}</span> '
            log_line += f'<span style="color: #FFD700;">{method}</span> '
            log_line += f'<span>{dest_path}</span> '
            log_line += f'<span style="color: {color}; font-weight: bold;">[{status}]</span> '
            log_line += f'<span style="color: #888;">{bytes_size} bytes</span>'
            
            terminal_html += f'{log_line}<br>'
        
        terminal_html += "</div>"
        st.markdown(terminal_html, unsafe_allow_html=True)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Logs", len(self.logs_buffer))
        with col2:
            if self.logs_buffer:
                success_rate = len([l for l in self.logs_buffer if str(l.get('status', '')).startswith('2')]) / len(self.logs_buffer) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            if self.logs_buffer:
                avg_size = sum([l.get('bytes', 0) for l in self.logs_buffer]) / len(self.logs_buffer)
                st.metric("Avg Size", f"{avg_size:.0f} bytes")
    
    def render_table_view(self):
        """Render logs in table format"""
        st.subheader("📊 Table View")
        
        if not self.logs_buffer:
            st.info("⏳ Waiting for logs...")
            return
        
        # Convert logs to DataFrame
        df = pd.DataFrame(self.logs_buffer)
        
        # Select and reorder columns
        display_columns = ['timestamp', 'request_src', 'method', 'dest_path', 'status', 'bytes']
        available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            df_display = df[available_columns].copy()
            
            # Format timestamp if it exists
            if 'timestamp' in df_display.columns:
                df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display with color coding for status
            def color_status(val):
                val_str = str(val)
                if val_str.startswith('2'):
                    return 'background-color: #d4edda'
                elif val_str.startswith('3'):
                    return 'background-color: #d1ecf1'
                elif val_str.startswith('4'):
                    return 'background-color: #fff3cd'
                elif val_str.startswith('5'):
                    return 'background-color: #f8d7da'
                return ''
            
            if 'status' in df_display.columns:
                styled_df = df_display.style.map(color_status, subset=['status'])
                st.dataframe(styled_df, width='stretch', height=500)
            else:
                st.dataframe(df_display, width='stretch', height=500)
            
            # Download button
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def render_timeseries_view(self, time_window_key):
        """Render logs in time-series chart format"""
        st.subheader("📈 Time-Series View")
        
        if not self.logs_buffer:
            st.info("⏳ Waiting for logs...")
            return
        
        # Convert logs to DataFrame
        df = pd.DataFrame(self.logs_buffer)
        
        if df.empty or 'timestamp' not in df.columns:
            st.warning("No timestamp data available")
            return
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get time window
        bin_size = settings.TIME_CHUNKS.get(time_window_key, "1min")
        
        # Group by time bins
        df_grouped = df.set_index('timestamp').resample(bin_size).size().reset_index(name='count')
        
        # Create time-series chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_grouped['timestamp'],
            y=df_grouped['count'],
            mode='lines+markers',
            name='Request Count',
            line=dict(color='#00BFFF', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(0, 191, 255, 0.2)'
        ))
        
        fig.update_layout(
            title=f'Request Rate (per {bin_size})',
            xaxis_title='Time',
            yaxis_title='Number of Requests',
            height=settings.CHART_HEIGHT,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # # Status code distribution over time
        # if 'status' in df.columns:
        #     st.subheader("📊 Status Code Distribution")
            
        #     # Create status groups
        #     df['status_group'] = df['status'].astype(str).str[0] + 'xx'
            
        #     # Group by time and status
        #     df_status = df.set_index('timestamp').groupby([pd.Grouper(freq=bin_size), 'status_group']).size().reset_index(name='count')
            
        #     fig_status = px.bar(
        #         df_status,
        #         x='timestamp',
        #         y='count',
        #         color='status_group',
        #         title=f'Status Codes Over Time (per {bin_size})',
        #         labels={'count': 'Number of Requests', 'timestamp': 'Time'},
        #         color_discrete_map={
        #             '2xx': settings.STATUS_COLORS['2xx'],
        #             '3xx': settings.STATUS_COLORS['3xx'],
        #             '4xx': settings.STATUS_COLORS['4xx'],
        #             '5xx': settings.STATUS_COLORS['5xx']
        #         },
        #         height=settings.CHART_HEIGHT
        #     )
            
        #     fig_status.update_layout(template='plotly_dark')
        #     st.plotly_chart(fig_status, width='stretch')
        
        # # Summary statistics
        # st.subheader("📋 Summary Statistics")
        # col1, col2, col3, col4 = st.columns(4)
        
        # with col1:
        #     total_requests = len(df)
        #     st.metric("Total Requests", total_requests)
        
        # with col2:
        #     if not df_grouped.empty:
        #         avg_rate = df_grouped['count'].mean()
        #         st.metric(f"Avg Rate ({bin_size})", f"{avg_rate:.1f}")
        
        # with col3:
        #     if not df_grouped.empty:
        #         max_rate = df_grouped['count'].max()
        #         st.metric(f"Peak Rate ({bin_size})", int(max_rate))
        
        # with col4:
        #     if 'status' in df.columns:
        #         error_rate = len(df[df['status'].astype(str).str.startswith(('4', '5'))]) / len(df) * 100
        #         st.metric("Error Rate", f"{error_rate:.1f}%")