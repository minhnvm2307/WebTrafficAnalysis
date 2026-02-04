"""
Log Monitor Dashboard - Main Streamlit Application
AWS CloudWatch-style log monitoring with real-time simulation
"""
import streamlit as st
import sys
import os
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components
from views.tracer import LogTracer
from views.dashboard import MainDashboard
from views.forecast_scale import ForecastScaleDashboard
import views.settings as settings
from modules.log_simulator import LogDataLoader
from modules.predict_simulator import PredictDataLoader

# Page configuration
st.set_page_config(
    page_title="Log Monitor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00BFFF;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #0C0C0C 0%, #1a1a1a 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .status-badge {
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .status-running {
        background-color: #28a745;
        color: white;
    }
    .status-stopped {
        background-color: #dc3545;
        color: white;
    }
    .status-paused {
        background-color: #ffc107;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'tracer' not in st.session_state:
        st.session_state.tracer = LogTracer()
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = MainDashboard()
    if 'forecast_dashboard' not in st.session_state:
        st.session_state.forecast_dashboard = ForecastScaleDashboard()
    
    # Log Simulator state
    if 'log_loader' not in st.session_state:
        st.session_state.log_loader = None
    if 'log_simulator_running' not in st.session_state:
        st.session_state.log_simulator_running = False
    if 'log_total_processed' not in st.session_state:
        st.session_state.log_total_processed = 0
    # Always use settings values (not cached in session_state)
    if 'log_file_path' not in st.session_state:
        st.session_state.log_file_path = settings.DEFAULT_LOG_FILE
    if 'log_speed_multiplier' not in st.session_state:
        st.session_state.log_speed_multiplier = settings.DEFAULT_SPEED_MULTIPLIER
    
    # Forecast Simulator state
    if 'predict_loader' not in st.session_state:
        st.session_state.predict_loader = None
    if 'predict_simulator_running' not in st.session_state:
        st.session_state.predict_simulator_running = False
    if 'predict_total_processed' not in st.session_state:
        st.session_state.predict_total_processed = 0
    if 'predict_file_path' not in st.session_state:
        st.session_state.predict_file_path = "cache/simulated_data.csv"
    if 'predict_speed_multiplier' not in st.session_state:
        st.session_state.predict_speed_multiplier = settings.DEFAULT_SPEED_MULTIPLIER

def stream_next_log():
    """Stream the next batch of logs from the loader and add to tracer"""
    if st.session_state.log_loader is None:
        return False
    
    try:
        # Use settings.BATCH_SIZE directly instead of cached value
        batch = st.session_state.log_loader.get_next_batch(settings.BATCH_SIZE)
        if batch:
            for log in batch:
                st.session_state.tracer.add_log(log)
                st.session_state.dashboard.add_log(log)
                st.session_state.log_total_processed += 1
            return True
        else:
            st.session_state.log_simulator_running = False
            return False
    except Exception as e:
        st.error(f"Stream error: {e}")
        st.session_state.log_simulator_running = False
        return False

def stream_next_forecast():
    """Stream the next batch of forecast data"""
    if st.session_state.predict_loader is None:
        return False
    
    try:
        # Use settings.BATCH_SIZE directly instead of cached value
        batch = st.session_state.predict_loader.get_next_batch(settings.BATCH_SIZE)
        if batch:
            for data in batch:
                st.session_state.forecast_dashboard.add_data(data)
                st.session_state.predict_total_processed += 1
            return True
        else:
            st.session_state.predict_simulator_running = False
            return False
    except Exception as e:
        st.error(f"Forecast stream error: {e}")
        st.session_state.predict_simulator_running = False
        return False

def sidebar_controls():
    """Render sidebar controls"""
    st.sidebar.title("Control Panel")
    
    # Log Simulator Control
    st.sidebar.subheader("Log Simulator")
    
    log_status = "Running" if st.session_state.log_simulator_running else "Stopped"
    st.sidebar.text(f"Status: {log_status}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start", key="log_start", disabled=st.session_state.log_simulator_running):
            st.session_state.log_loader = LogDataLoader(
                st.session_state.log_file_path,
                speed_multiplier=st.session_state.log_speed_multiplier,
                shuffle=False,
                loop=True
            )
            st.session_state.log_simulator_running = True
            st.rerun()
    
    with col2:
        if st.button("Stop", key="log_stop", disabled=not st.session_state.log_simulator_running):
            st.session_state.log_simulator_running = False
            st.session_state.log_loader = None
            st.rerun()
    
    if st.sidebar.button("Clear", key="log_clear"):
        st.session_state.tracer.clear_logs()
        st.session_state.dashboard.clear_logs()
        st.session_state.log_total_processed = 0
        st.rerun()
    
    st.sidebar.text(f"Processed: {st.session_state.log_total_processed}")
    
    st.sidebar.divider()
    
    # Forecast Simulator Control
    st.sidebar.subheader("Forecast Simulator")
    
    predict_status = "Running" if st.session_state.predict_simulator_running else "Stopped"
    st.sidebar.text(f"Status: {predict_status}")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("Start", key="predict_start", disabled=st.session_state.predict_simulator_running):
            st.session_state.predict_loader = PredictDataLoader(
                st.session_state.predict_file_path,
                speed_multiplier=st.session_state.predict_speed_multiplier,
                shuffle=False,
                loop=True
            )
            st.session_state.predict_simulator_running = True
            st.rerun()
    
    with col4:
        if st.button("Stop", key="predict_stop", disabled=not st.session_state.predict_simulator_running):
            st.session_state.predict_simulator_running = False
            st.session_state.predict_loader = None
            st.rerun()
    
    if st.sidebar.button("Clear", key="predict_clear"):
        st.session_state.forecast_dashboard.clear_data()
        st.session_state.predict_total_processed = 0
        st.rerun()
    
    st.sidebar.text(f"Processed: {st.session_state.predict_total_processed}")
    
    st.sidebar.divider()
    
    # Settings
    with st.sidebar.expander("Settings"):
        st.text("Log Speed")
        
        # Find current speed key for the value
        current_log_speed_key = None
        for key, val in settings.SPEED_OPTIONS.items():
            if val == st.session_state.log_speed_multiplier:
                current_log_speed_key = key
                break
        if current_log_speed_key is None:
            current_log_speed_key = "5x (Fast)"
        
        log_speed = st.select_slider(
            "log_speed",
            options=list(settings.SPEED_OPTIONS.keys()),
            value=current_log_speed_key,
            label_visibility="collapsed"
        )
        # Update speed multiplier in session state
        st.session_state.log_speed_multiplier = settings.SPEED_OPTIONS[log_speed]
        # Update running loader if exists
        if st.session_state.log_loader:
            st.session_state.log_loader.set_speed(st.session_state.log_speed_multiplier)
        
        st.text("Forecast Speed")
        
        # Find current speed key for the value
        current_predict_speed_key = None
        for key, val in settings.SPEED_OPTIONS.items():
            if val == st.session_state.predict_speed_multiplier:
                current_predict_speed_key = key
                break
        if current_predict_speed_key is None:
            current_predict_speed_key = "5x (Fast)"
        
        predict_speed = st.select_slider(
            "predict_speed",
            options=list(settings.SPEED_OPTIONS.keys()),
            value=current_predict_speed_key,
            label_visibility="collapsed"
        )
        # Update speed multiplier in session state
        st.session_state.predict_speed_multiplier = settings.SPEED_OPTIONS[predict_speed]
        # Update running loader if exists
        if st.session_state.predict_loader:
            st.session_state.predict_loader.set_speed(st.session_state.predict_speed_multiplier)

def main():
    """Main application function"""
    # Initialize
    init_session_state()
    
    # Sidebar controls
    sidebar_controls()
    
    # Header
    st.markdown('<div class="main-header">Log Monitor Dashboard</div>', unsafe_allow_html=True)

    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["Raw Log View", "Main Dashboard", "Forecast & Scaling"])
    
    with tab1:
        # Raw Log View - Sub-tabs for different views
        view_tab1, view_tab2, view_tab3 = st.tabs(["List View", "Table View", "Time-Series"])
        
        with view_tab1:
            st.session_state.tracer.render_terminal_view()
        
        with view_tab2:
            st.session_state.tracer.render_table_view()
        
        with view_tab3:
            # Time window selector
            time_window = st.selectbox(
                "Select Time Window",
                options=list(settings.TIME_CHUNKS.keys()),
                index=list(settings.TIME_CHUNKS.keys()).index(settings.DEFAULT_TIME_WINDOW)
            )
            st.session_state.tracer.render_timeseries_view(time_window)
    
    with tab2:
        # Main Dashboard - Placeholder for future analytics
        st.session_state.dashboard.render()
    with tab3:
        # Forecast & Auto-scaling Dashboard
        st.session_state.forecast_dashboard.render()


if __name__ == "__main__":
    main()
    should_refresh = False
    
    if st.session_state.log_simulator_running:
        stream_next_log()
        should_refresh = True
    
    if st.session_state.predict_simulator_running:
        stream_next_forecast()
        should_refresh = True
    
    if should_refresh:
        time.sleep(settings.REFRESH_INTERVAL)
        st.rerun()