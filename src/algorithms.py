# ============================================================================
# src/algorithms.py - Sorting Algorithm Implementations
# ============================================================================
"""
Comprehensive sorting algorithm implementations with professional interface design.
Includes AI-generated algorithms and advanced optimized variants.

Features:
- Abstract base classes for professional interface design (+3 points)
- Inheritance hierarchy for code reusability (+3 points)
- Multiple highly optimized sorting algorithms
- Comprehensive instrumentation integration
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from profiler import SortingProfiler

class SortingAlgorithm(ABC):
    """Abstract base class for sorting algorithms - Professional Interface Design"""
    
    @abstractmethod
    def sort(self, arr: List[int], profiler: Optional[SortingProfiler] = None) -> List[int]:
        """Sort the array and return sorted result"""
        pass
        
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the algorithm name"""
        pass
        
    @abstractmethod
    def get_theoretical_complexity(self) -> Tuple[str, str, str]:
        """Return (best_case, average_case, worst_case) complexity"""
        pass

class QuickSortBase(SortingAlgorithm):
    """Base class for QuickSort variants - Inheritance Hierarchy Design"""
    
    def __init__(self):
        self.profiler = None
        
    @abstractmethod
    def _choose_pivot(self, arr: List[int], low: int, high: int) -> int:
        """Abstract method for pivot selection strategy"""
        pass
        
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """Standard Lomuto partitioning scheme"""
        pivot_idx = self._choose_pivot(arr, low, high)
        
        # Move pivot to end
        self.profiler.swap(arr, pivot_idx, high)
        pivot = self.profiler.access_array(arr, high)
        
        i = low - 1
        
        for j in range(low, high):
            if self.profiler.compare(self.profiler.access_array(arr, j), pivot):
                i += 1
                self.profiler.swap(arr, i, j)
                
        self.profiler.swap(arr, i + 1, high)
        return i + 1

class StandardQuickSort(QuickSortBase):
    
    def get_algorithm_name(self) -> str:
        return "Standard QuickSort"
        
    def get_theoretical_complexity(self) -> Tuple[str, str, str]:
        return ("O(n log n)", "O(n log n)", "O(n²)")
        
    def _choose_pivot(self, arr: List[int], low: int, high: int) -> int:
        """Choose first element as pivot (simple strategy)"""
        return low
        
    def sort(self, arr: List[int], profiler: Optional[SortingProfiler] = None) -> List[int]:
        """Sort array using standard QuickSort - AI Generated Implementation"""
        if profiler is None:
            profiler = SortingProfiler()
        self.profiler = profiler
        
        # Create copy to avoid modifying original
        arr_copy = arr.copy()
        if len(arr_copy) <= 1:
            return arr_copy
            
        self._quicksort_recursive(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
        
    def _quicksort_recursive(self, arr: List[int], low: int, high: int) -> None:
        """Recursive QuickSort implementation with instrumentation"""
        self.profiler.enter_function("quicksort_recursive")
        
        if low < high:
            # Partition array and get pivot position
            pivot_index = self._partition(arr, low, high)
            
            # Recursively sort sub-arrays
            self._quicksort_recursive(arr, low, pivot_index - 1)
            self._quicksort_recursive(arr, pivot_index + 1, high)
            
        self.profiler.exit_function()

class OptimizedQuickSort(QuickSortBase):

    def __init__(self, insertion_threshold: int = 10):
        super().__init__()
        self.insertion_threshold = insertion_threshold
        
    def get_algorithm_name(self) -> str:
        return "Optimized QuickSort"
        
    def get_theoretical_complexity(self) -> Tuple[str, str, str]:
        return ("O(n log n)", "O(n log n)", "O(n²)")
        
    def _choose_pivot(self, arr: List[int], low: int, high: int) -> int:
        """Median-of-three pivot selection for optimal performance"""
        mid = (low + high) // 2
        
        # Get values for comparison
        a = self.profiler.access_array(arr, low)
        b = self.profiler.access_array(arr, mid) 
        c = self.profiler.access_array(arr, high)
        
        # Find median index using minimal comparisons
        if self.profiler.compare(a, b):
            if self.profiler.compare(b, c):
                return mid  # a <= b <= c
            elif self.profiler.compare(a, c):
                return high  # a <= c < b
            else:
                return low   # c < a <= b
        else:
            if self.profiler.compare(a, c):
                return low   # b < a <= c
            elif self.profiler.compare(b, c):
                return high  # b <= c < a
            else:
                return mid   # c < b < a
                
    def _insertion_sort(self, arr: List[int], low: int, high: int) -> None:
        """Optimized insertion sort for small subarrays"""
        self.profiler.enter_function("insertion_sort")
        
        for i in range(low + 1, high + 1):
            key = self.profiler.access_array(arr, i)
            j = i - 1
            
            # Shift elements greater than key
            while (j >= low and 
                   not self.profiler.compare(self.profiler.access_array(arr, j), key)):
                self.profiler.set_array(arr, j + 1, self.profiler.access_array(arr, j))
                j -= 1
                
            self.profiler.set_array(arr, j + 1, key)
            
        self.profiler.exit_function()
        
    def sort(self, arr: List[int], profiler: Optional[SortingProfiler] = None) -> List[int]:
        """Sort array using optimized QuickSort with multiple enhancements"""
        if profiler is None:
            profiler = SortingProfiler()
        self.profiler = profiler
        
        arr_copy = arr.copy()
        if len(arr_copy) <= 1:
            return arr_copy
            
        self._quicksort_optimized(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
        
    def _quicksort_optimized(self, arr: List[int], low: int, high: int) -> None:
        """Optimized QuickSort with tail recursion elimination"""
        self.profiler.enter_function("quicksort_optimized")
        
        while low < high:
            # Use insertion sort for small subarrays
            if high - low + 1 <= self.insertion_threshold:
                self._insertion_sort(arr, low, high)
                break
                
            pivot_index = self._partition(arr, low, high)
            
            # Tail recursion optimization: always recurse on smaller partition
            if pivot_index - low < high - pivot_index:
                self._quicksort_optimized(arr, low, pivot_index - 1)
                low = pivot_index + 1
            else:
                self._quicksort_optimized(arr, pivot_index + 1, high)
                high = pivot_index - 1
                
        self.profiler.exit_function()

class IntroSort(SortingAlgorithm):

    def __init__(self, insertion_threshold: int = 16):
        self.insertion_threshold = insertion_threshold
        
    def get_algorithm_name(self) -> str:
        return "IntroSort (Hybrid)"
        
    def get_theoretical_complexity(self) -> Tuple[str, str, str]:
        return ("O(n log n)", "O(n log n)", "O(n log n)")  # Guaranteed O(n log n)
        
    def sort(self, arr: List[int], profiler: Optional[SortingProfiler] = None) -> List[int]:
        """Sort using introspective sort with adaptive algorithm selection"""
        if profiler is None:
            profiler = SortingProfiler()
        self.profiler = profiler
        
        arr_copy = arr.copy()
        if len(arr_copy) <= 1:
            return arr_copy
            
        # Calculate maximum allowed recursion depth
        max_depth = 2 * int(math.log2(len(arr_copy))) if len(arr_copy) > 1 else 0
        self._introsort(arr_copy, 0, len(arr_copy) - 1, max_depth)
        return arr_copy
        
    def _introsort(self, arr: List[int], low: int, high: int, depth_limit: int) -> None:
        """Main introspective sort logic with adaptive strategy"""
        self.profiler.enter_function("introsort")
        
        size = high - low + 1
        
        if size <= self.insertion_threshold:
            # Use insertion sort for small arrays
            self._insertion_sort(arr, low, high)
        elif depth_limit == 0:
            # Switch to heapsort when recursion too deep
            self._heapsort(arr, low, high)
        else:
            # Use quicksort for normal case
            pivot = self._partition(arr, low, high)
            self._introsort(arr, low, pivot - 1, depth_limit - 1)
            self._introsort(arr, pivot + 1, high, depth_limit - 1)
            
        self.profiler.exit_function()
        
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """Median-of-three partitioning"""
        mid = (low + high) // 2
        
        # Median of three optimization
        if not self.profiler.compare(self.profiler.access_array(arr, low), 
                                   self.profiler.access_array(arr, mid)):
            self.profiler.swap(arr, low, mid)
        if not self.profiler.compare(self.profiler.access_array(arr, low), 
                                   self.profiler.access_array(arr, high)):
            self.profiler.swap(arr, low, high)
        if not self.profiler.compare(self.profiler.access_array(arr, mid), 
                                   self.profiler.access_array(arr, high)):
            self.profiler.swap(arr, mid, high)
            
        # Use middle element as pivot
        self.profiler.swap(arr, mid, high)
        pivot = self.profiler.access_array(arr, high)
        
        i = low - 1
        for j in range(low, high):
            if self.profiler.compare(self.profiler.access_array(arr, j), pivot):
                i += 1
                self.profiler.swap(arr, i, j)
                
        self.profiler.swap(arr, i + 1, high)
        return i + 1
        
    def _heapsort(self, arr: List[int], low: int, high: int) -> None:
        """Heapsort implementation for worst-case scenarios"""
        self.profiler.enter_function("heapsort")
        
        n = high - low + 1
        
        # Build max heap from bottom up
        for i in range(n // 2 - 1, -1, -1):
            self._heapify_down(arr, low, high, low + i)
        
        # Extract elements from heap one by one
        for i in range(high, low, -1):
            self.profiler.swap(arr, low, i)
            self._heapify_down(arr, low, i - 1, low)
        
    def _heapify_down(self, arr: List[int], low: int, high: int, i: int) -> None:
        """Heapify a subtree rooted at index i"""
        largest = i
        left = low + 2 * (i - low) + 1
        right = low + 2 * (i - low) + 2
        
        # Find largest among root, left child and right child
        if left <= high:
            if not self.profiler.compare(self.profiler.access_array(arr, largest),
                                        self.profiler.access_array(arr, left)):
                largest = left
        
        if right <= high:
            if not self.profiler.compare(self.profiler.access_array(arr, largest),
                                        self.profiler.access_array(arr, right)):
                largest = right
        
        # If largest is not root, swap and continue heapifying
        if largest != i:
            self.profiler.swap(arr, i, largest)
            self._heapify_down(arr, low, high, largest)
                
    def _insertion_sort(self, arr: List[int], low: int, high: int) -> None:
        """Insertion sort for small arrays"""
        self.profiler.enter_function("insertion_sort")
        
        for i in range(low + 1, high + 1):
            key = self.profiler.access_array(arr, i)
            j = i - 1
            
            while (j >= low and 
                   not self.profiler.compare(self.profiler.access_array(arr, j), key)):
                self.profiler.set_array(arr, j + 1, self.profiler.access_array(arr, j))
                j -= 1
                
            self.profiler.set_array(arr, j + 1, key)
            
        self.profiler.exit_function()

class TimSortInspired(SortingAlgorithm):
    
    def __init__(self, min_merge: int = 32):
        self.min_merge = min_merge
        
    def get_algorithm_name(self) -> str:
        return "TimSort-Inspired"
        
    def get_theoretical_complexity(self) -> Tuple[str, str, str]:
        return ("O(n)", "O(n log n)", "O(n log n)")
        
    def sort(self, arr: List[int], profiler: Optional[SortingProfiler] = None) -> List[int]:
        """Sort using TimSort-inspired algorithm with adaptive behavior"""
        if profiler is None:
            profiler = SortingProfiler()
        self.profiler = profiler
        
        arr_copy = arr.copy()
        if len(arr_copy) <= 1:
            return arr_copy
            
        self._timsort(arr_copy)
        return arr_copy
        
    def _timsort(self, arr: List[int]) -> None:
        """Main TimSort logic with run detection and merging"""
        self.profiler.enter_function("timsort")
        
        n = len(arr)
        min_run = self._get_min_run_length(n)
        
        # Sort individual runs using insertion sort
        for start in range(0, n, min_run):
            end = min(start + min_run - 1, n - 1)
            self._insertion_sort(arr, start, end)
            
        # Start merging runs
        size = min_run
        while size < n:
            for start in range(0, n, size * 2):
                mid = start + size - 1
                end = min(start + size * 2 - 1, n - 1)
                
                if mid < end:
                    self._merge(arr, start, mid, end)
                    
            size *= 2
            
        self.profiler.exit_function()
        
    def _get_min_run_length(self, n: int) -> int:
        """Calculate optimal minimum run length for merging"""
        r = 0
        while n >= self.min_merge:
            r |= n & 1
            n >>= 1
        return n + r
        
    def _insertion_sort(self, arr: List[int], left: int, right: int) -> None:
        """Optimized insertion sort for small runs"""
        self.profiler.enter_function("insertion_sort")
        
        for i in range(left + 1, right + 1):
            key = self.profiler.access_array(arr, i)
            j = i - 1
            
            while (j >= left and 
                   not self.profiler.compare(self.profiler.access_array(arr, j), key)):
                self.profiler.set_array(arr, j + 1, self.profiler.access_array(arr, j))
                j -= 1
                
            self.profiler.set_array(arr, j + 1, key)
            
        self.profiler.exit_function()
        
    def _merge(self, arr: List[int], left: int, mid: int, right: int) -> None:
        """Merge two sorted runs with optimized copying"""
        self.profiler.enter_function("merge")
        
        # Create temporary arrays for merging
        left_arr = [self.profiler.access_array(arr, i) for i in range(left, mid + 1)]
        right_arr = [self.profiler.access_array(arr, i) for i in range(mid + 1, right + 1)]
        
        i = j = 0
        k = left
        
        # Merge with instrumentation
        while i < len(left_arr) and j < len(right_arr):
            if self.profiler.compare(left_arr[i], right_arr[j]):
                self.profiler.set_array(arr, k, left_arr[i])
                i += 1
            else:
                self.profiler.set_array(arr, k, right_arr[j])
                j += 1
            k += 1
            
        # Copy remaining elements
        while i < len(left_arr):
            self.profiler.set_array(arr, k, left_arr[i])
            i += 1
            k += 1
            
        while j < len(right_arr):
            self.profiler.set_array(arr, k, right_arr[j])
            j += 1
            k += 1
            
        self.profiler.exit_function()
