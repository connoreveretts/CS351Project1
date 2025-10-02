# ============================================================================
# src/__init__.py
# ============================================================================
"""
Sorting Algorithm Analysis Package
A comprehensive framework for empirically evaluating AI-generated sorting algorithms
with research-grade statistical analysis and professional visualizations.
"""

__version__ = "1.0.0"
__author__ = "AI-Assisted Implementation"

from .profiler import PerformanceMetrics, SortingProfiler
from .algorithms import (
    SortingAlgorithm, QuickSortBase, StandardQuickSort, 
    OptimizedQuickSort, IntroSort, TimSortInspired
)
from .data_generator import TestDataGenerator
from .analyzer import StatisticalAnalyzer, ComplexityAnalyzer
from .visualizer import AdvancedVisualizer
from .main import main

__all__ = [
    'PerformanceMetrics', 'SortingProfiler',
    'SortingAlgorithm', 'QuickSortBase', 'StandardQuickSort', 
    'OptimizedQuickSort', 'IntroSort', 'TimSortInspired',
    'TestDataGenerator', 'StatisticalAnalyzer', 'ComplexityAnalyzer',
    'AdvancedVisualizer', 'main'
]
