
# ============================================================================
# src/profiler.py - Advanced Performance Instrumentation Framework
# ============================================================================

"""
Advanced performance profiler with comprehensive instrumentation for
sorting algorithm analysis. Tracks multiple metrics including comparisons,
swaps, memory usage, and simulated hardware effects.
"""

import time
import tracemalloc
import random
import statistics
from typing import List, Any, Optional
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Comprehensive performance measurements with statistical metadata"""
    comparisons: int = 0
    swaps: int = 0
    array_accesses: int = 0
    memory_allocations: int = 0
    wall_clock_time: float = 0.0
    peak_memory_usage: int = 0
    recursion_depth: int = 0
    cache_misses: int = 0
    branch_mispredictions: int = 0
    
    # Statistical metadata
    algorithm_name: str = ""
    input_size: int = 0
    distribution_type: str = ""
    trial_number: int = 0
    
    def __str__(self):
        return (f"Comp: {self.comparisons:,}, Swaps: {self.swaps:,}, "
                f"Access: {self.array_accesses:,}, Time: {self.wall_clock_time:.6f}s, "
                f"Depth: {self.recursion_depth}")

class SortingProfiler:
    """Research-grade performance profiler with comprehensive instrumentation"""
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self.reset_counters()
        self.call_stack = []
        
    def reset_counters(self) -> None:
        """Reset all performance counters"""
        self.comparisons = 0
        self.swaps = 0
        self.array_accesses = 0
        self.memory_allocations = 0
        self.start_time = 0
        self.peak_memory = 0
        self.max_recursion_depth = 0
        self.current_depth = 0
        self.cache_misses = 0
        self.branch_mispredictions = 0
        self._last_access_index = 0
        
    def start_timing(self) -> None:
        """Start comprehensive timing and memory measurement"""
        if self.enable_memory_profiling:
            try:
                tracemalloc.start()
            except:
                pass  # Already started
        self.start_time = time.perf_counter()
        
    def end_timing(self) -> float:
        """End measurements and return elapsed time"""
        elapsed = time.perf_counter() - self.start_time
        if self.enable_memory_profiling:
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.peak_memory = peak
            except:
                self.peak_memory = 0
        return elapsed
        
    def enter_function(self, func_name: str = "") -> None:
        """Track function entry for recursion depth"""
        self.current_depth += 1
        self.max_recursion_depth = max(self.max_recursion_depth, self.current_depth)
        self.call_stack.append(func_name)
        
    def exit_function(self) -> None:
        """Track function exit"""
        self.current_depth = max(0, self.current_depth - 1)
        if self.call_stack:
            self.call_stack.pop()
            
    def compare(self, a: Any, b: Any) -> bool:
        """Instrumented comparison with branch prediction simulation"""
        self.comparisons += 1
        
        # Simulate branch misprediction (random pattern = more mispredictions)
        if random.random() < 0.1:  # 10% misprediction rate
            self.branch_mispredictions += 1
            
        return a <= b
        
    def swap(self, arr: List, i: int, j: int) -> None:
        """Instrumented swap operation"""
        self.swaps += 1
        self.array_accesses += 4  # 2 reads + 2 writes
        
        # Simulate cache miss for non-local accesses
        if abs(i - j) > 64:  # Cache line simulation
            self.cache_misses += 1
            
        arr[i], arr[j] = arr[j], arr[i]
        
    def access_array(self, arr: List, index: int) -> Any:
        """Instrumented array access"""
        self.array_accesses += 1
        
        # Simulate cache behavior
        if abs(index - self._last_access_index) > 64:
            self.cache_misses += 1
        self._last_access_index = index
        
        return arr[index]
        
    def set_array(self, arr: List, index: int, value: Any) -> None:
        """Instrumented array write"""
        self.array_accesses += 1
        arr[index] = value
        
    def allocate_memory(self, size: int) -> List:
        """Track memory allocation"""
        self.memory_allocations += size
        return [None] * size
        
    def get_metrics(self, algorithm_name: str = "", input_size: int = 0, 
                   distribution_type: str = "", trial_number: int = 0) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        return PerformanceMetrics(
            comparisons=self.comparisons,
            swaps=self.swaps,
            array_accesses=self.array_accesses,
            memory_allocations=self.memory_allocations,
            peak_memory_usage=self.peak_memory,
            recursion_depth=self.max_recursion_depth,
            cache_misses=self.cache_misses,
            branch_mispredictions=self.branch_mispredictions,
            algorithm_name=algorithm_name,
            input_size=input_size,
            distribution_type=distribution_type,
            trial_number=trial_number
        )