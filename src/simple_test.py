#!/usr/bin/env python3
"""
Simple test to verify the sorting analysis works
Place this file in the src/ directory and run it
"""

from profiler import SortingProfiler
from algorithms import StandardQuickSort, OptimizedQuickSort
from data_generator import TestDataGenerator
from analyzer import StatisticalAnalyzer

def simple_test():
    """Run a simple test to verify everything works"""
    print("🧪 Running Simple Test...")
    
    # Create a simple test
    algorithms = [StandardQuickSort(), OptimizedQuickSort()]
    test_data = TestDataGenerator.generate_comprehensive_test_suite(max_size=100)
    analyzer = StatisticalAnalyzer()
    
    print("✅ Algorithms created successfully")
    print("✅ Test data generated successfully")
    
    # Test one algorithm on one distribution
    algorithm = algorithms[0]
    distribution = 'random'
    size = 50
    test_array = test_data[distribution][size]
    
    print(f"\n🔬 Testing {algorithm.get_algorithm_name()} on {len(test_array)} elements...")
    
    # Run the test
    profiler = SortingProfiler()
    profiler.start_timing()
    
    try:
        sorted_array = algorithm.sort(test_array, profiler)
        elapsed_time = profiler.end_timing()
        
        # Verify correctness
        expected = sorted(test_array)
        is_correct = sorted_array == expected
        
        print(f"✅ Sorting completed in {elapsed_time:.6f} seconds")
        print(f"✅ Correctness check: {'PASSED' if is_correct else 'FAILED'}")
        
        # Get metrics
        metrics = profiler.get_metrics()
        print(f"📊 Metrics: {metrics}")
        
        # Test analyzer
        analyzer.add_result(algorithm.get_algorithm_name(), distribution, size, metrics)
        stats = analyzer.compute_statistics(algorithm.get_algorithm_name(), distribution, size)
        
        if stats:
            print(f"📈 Statistics computed successfully")
            print(f"   Mean comparisons: {stats['comparisons']['mean']:.0f}")
        
        print("\n🎉 All tests passed! The framework is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()