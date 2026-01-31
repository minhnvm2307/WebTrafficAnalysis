import random
import time
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocessing.data_parser import load_raw_data, preprocess_data, load_raw_data_online

class LogDataLoader:
    def __init__(self, file_path, speed_multiplier=1.0, shuffle=False, loop=False):
        """
        Args:
            file_path: Path to the log file
            speed_multiplier: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed)
            shuffle: If True, randomize the order of logs (warning: breaks timestamp order)
            loop: If True, repeat dataset infinitely; if False, stop after one pass
        """
        self.file_path = file_path
        self.speed_multiplier = speed_multiplier
        self.shuffle = shuffle
        self.loop = loop
        
        # Load raw lines
        self.lines = load_raw_data_online(file_path)
        self.total_lines = len(self.lines)
        self.current_index = 0
        
        # Preprocess all lines once for efficiency
        self._preprocess_all()
        
    def _preprocess_all(self):
        """Preprocess all log lines once and calculate time deltas."""
        self.parsed_logs = []
        df, _ = preprocess_data(self.lines)
        if not df.empty:
            # Convert timestamp to datetime (vectorized operation)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                format="%d/%b/%Y:%H:%M:%S %z"
            )

            self.parsed_logs = df.to_dict('records')
        self.total_logs = len(self.parsed_logs)
    
    def _parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object."""
        try:
            return datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
        except:
            return datetime.now()
    
    def __len__(self):
        """Return the number of logs in the dataset."""
        return self.total_logs
    
    def __iter__(self):
        """Return the iterator object (self)."""
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.parsed_logs)
        return self
    
    def __next__(self):
        """Yield the next log entry with timestamp-based delay."""
        if self.current_index >= self.total_logs:
            if self.loop:
                self.current_index = 0
                if self.shuffle:
                    random.shuffle(self.parsed_logs)
            else:
                raise StopIteration
        
        log = self.parsed_logs[self.current_index]
        
        # Apply delay based on timestamp difference (scaled by speed multiplier)
        if self.current_index > 0:
            delay = 1 / self.speed_multiplier
            if delay > 0:
                time.sleep(delay)
        
        self.current_index += 1
        return log
    
    def _get_next_no_delay(self):
        """Get next log without applying delay (for batch streaming)."""
        if self.current_index >= self.total_logs:
            if self.loop:
                self.current_index = 0
                if self.shuffle:
                    random.shuffle(self.parsed_logs)
            else:
                raise StopIteration
        
        log = self.parsed_logs[self.current_index]
        self.current_index += 1
        return log

    def stream(self, max_logs=None):
        """
        Generator that yields logs one by one based on their timestamp intervals.
        
        Args:
            max_logs: Maximum number of logs to yield (None for unlimited in loop mode)
        
        Yields:
            dict: Parsed log entry
        """
        count = 0
        for log in self:
            yield log
            count += 1
            if max_logs is not None and count >= max_logs:
                break
    
    def batch_stream(self, batch_size=10, max_batches=None):
        """
        Generator that yields logs in batches with timestamp-based delays.
        Delay is applied once per batch, not per individual log.
        
        Args:
            batch_size: Number of logs per batch
            max_batches: Maximum number of batches to yield (None for unlimited)
        
        Yields:
            list: Batch of parsed log entries
        """
        batch = []
        batch_count = 0
        
        while True:
            try:
                log = self._get_next_no_delay()
                batch.append(log)
                
                if len(batch) >= batch_size:
                    # Apply delay once per batch
                    if batch_count > 0:
                        delay = 1 / self.speed_multiplier
                        if delay > 0:
                            time.sleep(delay)
                    
                    yield batch
                    batch = []
                    batch_count += 1
                    
                    if max_batches is not None and batch_count >= max_batches:
                        break
            except StopIteration:
                break
        
        # Yield remaining logs if any
        if batch:
            yield batch
    
    def set_speed(self, speed_multiplier):
        """
        Dynamically change the playback speed.
        
        Args:
            speed_multiplier: Speed multiplier (1.0 = real-time, 2.0 = 2x, 10.0 = 10x)
        """
        self.speed_multiplier = speed_multiplier
    
    def get_next_batch(self, batch_size=10):
        """
        Get the next batch of logs without using a generator.
        Applies delay once per batch call.
        
        Args:
            batch_size: Number of logs to return
            
        Returns:
            list: Batch of parsed log entries, or empty list if no more logs
        """
        batch = []
        
        for _ in range(batch_size):
            if self.current_index >= self.total_logs:
                if self.loop:
                    self.current_index = 0
                    if self.shuffle:
                        random.shuffle(self.parsed_logs)
                else:
                    break
            
            if self.current_index < self.total_logs:
                batch.append(self.parsed_logs[self.current_index])
                self.current_index += 1
        
        return batch

    
# Example usage
if __name__ == "__main__":
    # Real-time playback (1x speed)
    loader = LogDataLoader("dataset/test.txt", speed_multiplier=1.0, loop=False)
    start_time = time.time()
    for i, log in enumerate(loader.stream(max_logs=10)):
        elapsed = time.time() - start_time
        print(f"[{elapsed:.2f}s] Log {i+1}: {log['timestamp']} | {log['request_src']} -> {log['dest_path']} [{log['status']}]")
    


    # Fast playback (10x speed)
    loader2 = LogDataLoader("dataset/test.txt", speed_multiplier=10.0, loop=False)
    # Get 3 batches of 5 logs each
    start_time = time.time()
    for batch_num, batch in enumerate(loader2.batch_stream(batch_size=5, max_batches=3)):
        elapsed = time.time() - start_time
        print(f"\n[{elapsed:.2f}s] Batch {batch_num+1} ({len(batch)} logs):")
        for log in batch:
            print(f"  - {log['timestamp']} | {log['method']} {log['dest_path']}")
    


    # Speed change during playback
    loader3 = LogDataLoader("dataset/test.txt", speed_multiplier=5.0, loop=True)
    count = 0
    start_time = time.time()
    for log in loader3:
        elapsed = time.time() - start_time
        print(f"[{elapsed:.2f}s] {log['timestamp']} | {log['request_src']}")
        count += 1

        # Change speed after 5 logs
        if count == 5:
            print(">>> Speeding up to 20x")
            loader3.set_speed(20.0)
        
        if count >= 10:
            break