import random
import time
import sys
import os
import pandas as pd
from datetime import datetime

from traffic_monitor.modules.log_simulator import LogDataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocessing.data_parser import load_raw_data, preprocess_data, load_raw_data_online

class PredictDataLoader:
    def __init__(self, file_path, speed_multiplier=1.0, shuffle=False, loop=False):
        self.file_path = file_path
        self.speed_multiplier = speed_multiplier
        self.shuffle = shuffle
        self.loop = loop
        
        # Load raw lines
        self._load_cached_data()
        self.current_index = 0
        
    def _load_cached_data(self):
        self.lines = pd.read_csv(self.file_path).to_dict('records')
        self.total_lines = len(self.lines)
    
    def _parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object."""
        try:
            return datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
        except:
            return datetime.now()
    
    def __len__(self):
        """Return the number of logs in the dataset."""
        return self.total_lines
    
    def __iter__(self):
        """Return the iterator object (self)."""
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.lines)
        return self
    
    def __next__(self):
        """Yield the next log entry with timestamp-based delay."""
        if self.current_index >= self.total_lines:
            if self.loop:
                self.current_index = 0
                if self.shuffle:
                    random.shuffle(self.lines)
            else:
                raise StopIteration
        
        log = self.lines[self.current_index]
        
        # Apply delay based on timestamp difference (scaled by speed multiplier)
        if self.current_index > 0:
            delay = 3 / self.speed_multiplier
            if delay > 0:
                time.sleep(delay)
        
        self.current_index += 1
        return log
    
    def _get_next_no_delay(self):
        """Get next log without applying delay (for batch streaming)."""
        if self.current_index >= self.total_lines:
            if self.loop:
                self.current_index = 0
                if self.shuffle:
                    random.shuffle(self.lines)
            else:
                raise StopIteration
        
        log = self.lines[self.current_index]
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
                        delay = 3 / self.speed_multiplier
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
            if self.current_index >= self.total_lines:
                if self.loop:
                    self.current_index = 0
                    if self.shuffle:
                        random.shuffle(self.lines)
                else:
                    break
            
            if self.current_index < self.total_lines:
                batch.append(self.lines[self.current_index])
                self.current_index += 1
        
        return batch

    
# Example usage
if __name__ == "__main__":
    simulator = PredictDataLoader(
        file_path='./cache/simulated_data.csv',
        speed_multiplier=2.0,
        shuffle=True,
        loop=True
    )
    
    # Stream logs one by one
    for i, log in enumerate(simulator.stream(max_logs=5)):
        print(log)
    
    # Stream logs in batches
    for i, batch in enumerate(simulator.batch_stream(batch_size=3, max_batches=2)):
        print(f"Batch {i+1}:")
        for log in batch:
            print(log)