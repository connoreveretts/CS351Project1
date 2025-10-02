# ============================================================================
# src/data_generator.py - Advanced Test Data Generation
# ============================================================================
"""
Advanced test data generation for comprehensive algorithm evaluation.
Supports multiple distribution types and sophisticated data patterns
for thorough algorithm analysis.
"""

import random
from typing import List, Dict, Any

class TestDataGenerator:
    """Generate sophisticated test datasets for comprehensive analysis"""
    
    @staticmethod
    def generate_dataset(size: int, distribution: str, seed: int = None, **kwargs) -> List[int]:
        """Generate dataset with specified distribution and parameters
        
        Args:
            size: Number of elements in dataset
            distribution: Type of data distribution
            seed: Random seed for reproducibility
            **kwargs: Additional parameters for specific distributions
            
        Returns:
            List of integers representing the dataset
        """
        if seed is not None:
            random.seed(seed)
            
        if distribution == "random":
            return [random.randint(1, size * 10) for _ in range(size)]
            
        elif distribution == "sorted":
            return list(range(1, size + 1))
            
        elif distribution == "reverse_sorted":
            return list(range(size, 0, -1))
            
        elif distribution == "nearly_sorted":
            # Control the "nearly" factor with disorder_factor parameter
            disorder_factor = kwargs.get('disorder_factor', 0.05)
            arr = list(range(1, size + 1))
            swaps = max(1, int(size * disorder_factor))
            for _ in range(swaps):
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
            
        elif distribution == "many_duplicates":
            # Control number of unique values with unique_ratio
            unique_ratio = kwargs.get('unique_ratio', 0.2)
            unique_vals = max(1, int(size * unique_ratio))
            return [random.randint(1, unique_vals) for _ in range(size)]
            
        elif distribution == "few_unique":
            # Very few unique values (stress test for some algorithms)
            unique_count = kwargs.get('unique_count', 3)
            return [random.choice(range(1, unique_count + 1)) for _ in range(size)]
            
        elif distribution == "pipe_organ":
            # Sorted ascending to middle, then descending
            mid = size // 2
            return list(range(1, mid + 1)) + list(range(mid, 0, -1))
            
        elif distribution == "sawtooth":
            # Repeating sawtooth pattern
            pattern_size = kwargs.get('pattern_size', 10)
            pattern = list(range(1, pattern_size + 1))
            return (pattern * (size // pattern_size + 1))[:size]
            
        elif distribution == "zipf":
            # Zipf distribution (power law) - realistic for many real-world datasets
            alpha = kwargs.get('alpha', 1.5)
            values = []
            for _ in range(size):
                rank = random.randint(1, size // 10)
                value = int(rank ** (-1/alpha))
                values.append(max(1, value))
            return values
            
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def generate_comprehensive_test_suite(max_size: int = 10000) -> Dict[str, Dict[int, List[int]]]:
        """Generate comprehensive test suite with multiple sizes and distributions
        
        Args:
            max_size: Maximum array size to generate
            
        Returns:
            Dictionary mapping distribution names to size-array mappings
        """
        # Standard test sizes for comprehensive analysis
        sizes = [10, 50, 100, 500, 1000, 2000, 5000]
        if max_size >= 10000:
            sizes.append(10000)
        
        sizes = [s for s in sizes if s <= max_size]
        
        # Distribution configurations with parameters
        distributions = {
            "random": {},
            "sorted": {},
            "reverse_sorted": {},
            "nearly_sorted": {"disorder_factor": 0.05},
            "many_duplicates": {"unique_ratio": 0.2},
            "few_unique": {"unique_count": 3},
            "pipe_organ": {},
            "sawtooth": {"pattern_size": 10},
            "zipf": {"alpha": 1.5}
        }
        
        test_suite = {}
        for dist_name, params in distributions.items():
            test_suite[dist_name] = {}
            for size in sizes:
                test_suite[dist_name][size] = TestDataGenerator.generate_dataset(
                    size, dist_name, seed=42, **params
                )
        
        return test_suite
    
    @staticmethod
    def generate_adversarial_cases() -> Dict[str, List[int]]:
        """Generate adversarial test cases that expose algorithm weaknesses"""
        cases = {}
        
        # Killer sequence for quicksort with first-element pivot
        cases["quicksort_killer"] = list(range(100, 0, -1))
        
        # All identical elements
        cases["all_identical"] = [42] * 100
        
        # Two distinct values alternating
        cases["alternating"] = [1, 2] * 50
        
        # Large range with single outlier
        base = list(range(1, 100))
        base.append(999999)
        cases["outlier"] = base
        
        return cases