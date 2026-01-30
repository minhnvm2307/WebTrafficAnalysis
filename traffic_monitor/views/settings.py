"""
Settings and configuration for the Log Monitor Dashboard
"""

# Simulation settings
DEFAULT_SPEED_MULTIPLIER = 5.0
SPEED_OPTIONS = {
    "1x (Real-time)": 1.0,
    "5x (Fast)": 5.0,
    "20x (Very Fast)": 20.0,
    "100x (Ultra Fast)": 100.0,
    "200x (Maximum)": 200.0
}

# Display settings
MAX_LOGS_DISPLAY = 5000  
MAX_LOGS_DASHBOARD = 500
BATCH_SIZE = 30  
REFRESH_INTERVAL = 2.3  

# Time window settings for time-series analysis
TIME_CHUNKS = {
    "1 minute": '1min',
    "5 minutes": '5min',
    "15 minutes": '15min',
    "30 minutes": '30min',
    "1 hour": '1h'
}
DEFAULT_TIME_WINDOW = "5 minutes"
CHART_HEIGHT = 500
BASE_CPU = 10.0
BASE_MEMORY = 512

# Terminal style settings
TERMINAL_BG_COLOR = "#0C0C0C"
TERMINAL_TEXT_COLOR = "#557955"
TERMINAL_MAX_LINES = 50

# Status code colors
STATUS_COLORS = {
    "2xx": "#00fa3a",  
    "3xx": "#17a2b8",  
    "4xx": "#ff8b07",  
    "5xx": "#dc3545",  
}

# Log file path (default)
DEFAULT_LOG_FILE = "dataset/test.txt"