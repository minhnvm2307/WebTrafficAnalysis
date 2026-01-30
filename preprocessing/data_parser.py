import pandas as pd
import csv
import re
from datetime import datetime

'''
Raw data format
1. User Host IP/Domain
2. Timestampe
3. Request URL
4. Status
5. Return payload (byte)
'''

def load_raw_data(file_path):
    """Load raw data from the specified TXT file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return lines

def preprocess_data(lines, verbose=False):
    """Preprocess the raw data and return the processed DataFrame."""
    # Regex pattern to parse Apache/NASA log format
    log_pattern = r'^(\S+) - - \[([^\]]+)\] "(.*)" (\d+) (\S+)'
    
    parsed_data = []
    skipped_lines = []
    
    for line in lines:
        match = re.match(log_pattern, line)
        if match:
            request_src = match.group(1)
            timestamp = match.group(2)
            request = match.group(3)
            status = int(match.group(4))
            bytes_sent = match.group(5)
            
            # Extract method, URL, and HTTP type
            method = ''
            url = ''
            http_type = ''
            
            # Match HTTP method (GET, POST, PUT, DELETE, HEAD, OPTIONS, PATCH, etc.)
            method_match = re.match(r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|CONNECT|TRACE)\s*', request, re.IGNORECASE)
            if method_match:
                method = method_match.group(1).upper()
                remaining = request[method_match.end():]
                
                # Match PATH
                path_match = re.match(r'(\S+)\s*', remaining)
                if path_match:
                    path = path_match.group(1)
                    remaining = remaining[path_match.end():]
                    
                    # Match HTTP type (HTTP/x.x format)
                    http_match = re.match(r'(HTTP/\d+\.\d+)', remaining, re.IGNORECASE)
                    if http_match:
                        http_type = http_match.group(1).upper()
            
            # Convert bytes to integer, handle '-' as 0
            try:
                bytes_sent = int(bytes_sent) if bytes_sent != '-' else 0
            except ValueError:
                bytes_sent = 0
            
            parsed_data.append({
                'request_src': request_src,
                'timestamp': timestamp,
                'method': method,
                'dest_path': path,
                'http_type': http_type,
                'status': status,
                'bytes': bytes_sent
            })
        else:
            skipped_lines.append(line)
    
    if verbose:
        print(f"Parsed {len(parsed_data)} lines, skipped {len(skipped_lines)} lines.")
    
    df = pd.DataFrame(parsed_data)
    return df, skipped_lines

def save_processed_data(df, file_path):
    """Save the processed data to the specified CSV file."""
    df.to_csv(file_path, index=False)