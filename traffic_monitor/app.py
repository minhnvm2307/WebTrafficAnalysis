"""
Log Monitor Dashboard - Main Streamlit Application
AWS CloudWatch-style log monitoring with real-time simulation
"""
import streamlit as st
import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components
from views.tracer import LogTracer
from views.dashboard import MainDashboard
import views.settings as settings
from modules.log_simulator import LogDataLoader

# Page configuration
st.set_page_config(
    page_title="Log Monitor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    
    if 'loader' not in st.session_state:
        st.session_state.loader = None
    
    if 'log_iter' not in st.session_state:
        st.session_state.log_iter = None
    
    if 'simulator_running' not in st.session_state:
        st.session_state.simulator_running = False
    
    if 'total_logs_processed' not in st.session_state:
        st.session_state.total_logs_processed = 0
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    
    if 'speed_multiplier' not in st.session_state:
        st.session_state.speed_multiplier = settings.DEFAULT_SPEED_MULTIPLIER
    
    if 'log_file_path' not in st.session_state:
        st.session_state.log_file_path = settings.DEFAULT_LOG_FILE
    
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = settings.BATCH_SIZE

def stream_next_log():
    """Stream the next batch of logs from the loader and add to tracer"""
    if st.session_state.loader is None:
        return False
    
    try:
        batch = st.session_state.loader.get_next_batch(st.session_state.batch_size)
        if batch:
            for log in batch:
                st.session_state.tracer.add_log(log)
                st.session_state.dashboard.add_log(log)
                st.session_state.total_logs_processed += 1
            return True
        else:
            # End of stream
            st.session_state.simulator_running = False
            return False
    except Exception as e:
        st.error(f"Stream error: {e}")
        st.session_state.simulator_running = False
        return False

def sidebar_controls():
    """Render sidebar controls"""
    st.sidebar.title("⚙️ Control Panel")
    
    # Simulator status
    status = "🟢 Running" if st.session_state.simulator_running else "🔴 Stopped"
    st.sidebar.markdown(f"### Status: {status}")
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.simulator_running):
            # Create loader
            st.session_state.loader = LogDataLoader(
                st.session_state.log_file_path,
                speed_multiplier=st.session_state.speed_multiplier,
                shuffle=False,
                loop=True
            )
            st.session_state.simulator_running = True
            st.session_state.start_time = time.time()
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop", use_container_width=True, disabled=not st.session_state.simulator_running):
            st.session_state.simulator_running = False
            st.session_state.loader = None
            st.rerun()
    
    # Clear logs button
    if st.sidebar.button("🗑️ Clear Logs", use_container_width=True):
        st.session_state.tracer.clear_logs()
        st.session_state.dashboard.clear_logs()
        st.session_state.total_logs_processed = 0
        st.rerun()
    
    st.sidebar.divider()
    
    # Speed control
    st.sidebar.subheader("🚀 Simulation Speed")
    
    selected_speed_label = st.sidebar.select_slider(
        "Speed Multiplier",
        options=list(settings.SPEED_OPTIONS.keys()),
        value="5x (Fast)",
        help="Control how fast logs are simulated"
    )
    
    new_speed = settings.SPEED_OPTIONS[selected_speed_label]
    
    # Update speed if changed
    if new_speed != st.session_state.speed_multiplier:
        st.session_state.speed_multiplier = new_speed
        if st.session_state.loader is not None:
            st.session_state.loader.set_speed(new_speed)
        st.sidebar.success(f"Speed updated to {selected_speed_label}")
    
    # Batch size control
    st.sidebar.subheader("📦 Batch Size")
    
    batch_size = st.sidebar.slider(
        "Logs per batch",
        min_value=1,
        max_value=100,
        value=st.session_state.batch_size,
        step=1,
        help="Number of logs to load per refresh cycle"
    )
    
    if batch_size != st.session_state.batch_size:
        st.session_state.batch_size = batch_size
    
    st.sidebar.divider()
    
    # File selection
    st.sidebar.subheader("📁 Log File")
    
    log_file = st.sidebar.text_input(
        "File Path",
        value=st.session_state.log_file_path,
        help="Path to log file"
    )
    
    if log_file != st.session_state.log_file_path:
        st.session_state.log_file_path = log_file
        st.session_state.loader = None  # Reset loader
        st.sidebar.info("File path updated. Stop and restart to load new file.")
    
    st.sidebar.divider()
    
    # Statistics
    st.sidebar.subheader("📊 Statistics")
    st.sidebar.metric("Logs Processed", st.session_state.total_logs_processed)
    
    if st.session_state.start_time is not None:
        elapsed = time.time() - st.session_state.start_time
        st.sidebar.metric("Runtime", f"{elapsed:.1f}s")
        
        if st.session_state.total_logs_processed > 0 and elapsed > 0:
            rate = st.session_state.total_logs_processed / elapsed
            st.sidebar.metric("Processing Rate", f"{rate:.1f} logs/s")
    
    st.sidebar.divider()
    
    # About
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **Log Monitor Dashboard**
        
        A CloudWatch-style dashboard for monitoring server request logs in real-time.
        
        **Features:**
        - Real-time log simulation
        - Multiple view modes
        - Time-series analysis
        - Adjustable playback speed
        
        **Views:**
        1. Terminal View - Raw log format
        2. Table View - Structured data
        3. Time-Series - Statistical analysis
        """)

def main():
    """Main application function"""
    # Initialize
    init_session_state()
    
    # Sidebar controls
    sidebar_controls()
    
    # Header
    st.markdown('<div class="main-header">📊 Log Monitor Dashboard</div>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["📋 Raw Log View", "📊 Main Dashboard"])
    
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
    
    # Auto-refresh when simulator is running
    if st.session_state.simulator_running:
        # Stream next log
        stream_next_log()
        time.sleep(settings.REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()