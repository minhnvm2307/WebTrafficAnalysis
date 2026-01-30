import views.settings as settings
import random

def calculate_cpu_usage(logs_buffer,logs_count, time_window=60):
    """
    Simulate CPU usage based on request load
    Higher request rate = higher CPU usage
    """
    if not logs_buffer:
        return 0.0
    
    # Base CPU usage
    base_cpu = settings.BASE_CPU  
    
    # Calculate request rate
    recent_logs = logs_buffer[-logs_count:] if logs_count < len(logs_buffer) else logs_buffer
    request_rate = len(recent_logs) / time_window if time_window > 0 else 0
    
    # CPU scales with request rate (simulate)
    cpu_load = min(base_cpu + (request_rate * 2), 100.0)
    
    # Add some random variation
    cpu_load += random.uniform(-5, 5)
    cpu_load = max(0, min(100, cpu_load))
    
    return cpu_load

def calculate_memory_usage(logs_buffer, logs_count):
    """
    Simulate memory usage based on number of logs in buffer
    More logs in memory = higher memory usage
    """
    if not logs_buffer:
        return 0.0
    
    # Base memory usage (MB)
    base_memory = settings.BASE_MEMORY  
    
    # Memory scales with buffer size
    memory_per_log = 0.1  # KB per log
    buffer_memory = logs_count * memory_per_log / 1024  # Convert to MB
    
    total_memory = base_memory + buffer_memory
    
    # Simulate as percentage of available memory (assume 4GB)
    memory_percent = (total_memory / 4096) * 100
    
    # Add some random variation
    memory_percent += random.uniform(-2, 2)
    memory_percent = max(0, min(100, memory_percent))
    
    return memory_percent