import cProfile
import pstats
import io
from functools import wraps
from typing import Optional, Dict
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Profiler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.profiler = cProfile.Profile()
        self.is_active = False
        self.function_times: Dict[str, float] = {}
        self.samples = {}
        self.sample_count = 100
        self.log_counter = 0
        self.log_interval = 60  # Log every 60 frames
        self.performance_summary = {
            'Physics': {'time': 0.0, 'samples': 0},
            'Rendering': {'time': 0.0, 'samples': 0},
            'Collisions': {'time': 0.0, 'samples': 0}
        }
    
    def start(self):
        self.is_active = True
        self.profiler.enable()
    
    def stop(self):
        self.profiler.disable()
        self.is_active = False
        
        # Process and log results
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Show top 20 functions
        logger.info(f"\nProfile Results:\n{s.getvalue()}")
        
        self.profiler = cProfile.Profile()  # Reset profiler
    
    def log_function_time(self, func_name: str, execution_time: float):
        """Log execution time for a function and maintain running average"""
        if func_name not in self.samples:
            self.samples[func_name] = []
        
        samples = self.samples[func_name]
        samples.append(execution_time)
        
        # Keep only the last sample_count samples
        if len(samples) > self.sample_count:
            samples.pop(0)
        
        # Update running average
        self.function_times[func_name] = sum(samples) / len(samples)
        
        # Categorize and accumulate performance data
        if 'Physics' in func_name:
            self.performance_summary['Physics']['time'] += execution_time
            self.performance_summary['Physics']['samples'] += 1
        elif 'Renderer' in func_name:
            self.performance_summary['Rendering']['time'] += execution_time
            self.performance_summary['Rendering']['samples'] += 1
        elif 'Collision' in func_name:
            self.performance_summary['Collisions']['time'] += execution_time
            self.performance_summary['Collisions']['samples'] += 1

        # Log summary periodically
        self.log_counter += 1
        if self.log_counter >= self.log_interval:
            self._log_performance_summary()
            self.log_counter = 0
            self._reset_performance_summary()

    def _log_performance_summary(self):
        """Log a summary of performance metrics"""
        summary_lines = ["Performance Summary:"]
        
        for category, data in self.performance_summary.items():
            if data['samples'] > 0:
                avg_time = (data['time'] / data['samples']) * 1000  # Convert to ms
                summary_lines.append(f"{category}: {avg_time:.2f}ms avg")
        
        logger.info("\n".join(summary_lines))

    def _reset_performance_summary(self):
        """Reset performance summary counters"""
        for category in self.performance_summary:
            self.performance_summary[category] = {'time': 0.0, 'samples': 0}

def profile_function(threshold_ms: Optional[float] = None):
    """
    Decorator to profile individual functions
    :param threshold_ms: Only log if execution time exceeds this threshold (in milliseconds)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler()
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Only log if exceeds threshold (if specified)
            if threshold_ms is None or execution_time * 1000 > threshold_ms:
                profiler.log_function_time(func.__qualname__, execution_time)
            
            return result
        return wrapper
    return decorator 