# src/analyzer.py - FIXED VERSION
"""
Advanced statistical analysis framework for algorithm performance evaluation.
Provides research-grade statistical analysis with confidence intervals,
hypothesis testing, and multiple complexity model fitting.
"""

import math
import statistics
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy import stats
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, some features will be limited")

from profiler import PerformanceMetrics

class StatisticalAnalyzer:
    """Advanced statistical analysis with confidence intervals and hypothesis testing"""
    
    def __init__(self):
        self.results = defaultdict(list)  # FIXED: Simple defaultdict(list)
        
    def add_result(self, algorithm: str, distribution: str, size: int, 
                  metrics: PerformanceMetrics) -> None:
        """Add performance result for statistical analysis"""
        key = (algorithm, distribution, size)
        self.results[key].append(metrics)
        
    def compute_statistics(self, algorithm: str, distribution: str, size: int) -> Dict[str, Any]:
        """Compute comprehensive statistics for given parameters"""
        key = (algorithm, distribution, size)
        if key not in self.results:
            return {}
            
        metrics_list = self.results[key]
        if not metrics_list:
            return {}
            
        # Extract metric values
        comparisons = [m.comparisons for m in metrics_list]
        times = [m.wall_clock_time for m in metrics_list]
        swaps = [m.swaps for m in metrics_list]
        memory = [m.peak_memory_usage for m in metrics_list]
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            if not values or len(values) == 0:
                return {}
            
            mean_val = statistics.mean(values)
            
            stats_dict = {
                'mean': mean_val,
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values)
            }
            
            if len(values) > 1:
                stats_dict['std_dev'] = statistics.stdev(values)
                stats_dict['variance'] = statistics.variance(values)
                stats_dict['confidence_interval_95'] = self._confidence_interval(values, 0.95)
                stats_dict['coefficient_of_variation'] = statistics.stdev(values) / mean_val if mean_val != 0 else 0
            else:
                stats_dict['std_dev'] = 0
                stats_dict['variance'] = 0
                stats_dict['confidence_interval_95'] = (mean_val, mean_val)
                stats_dict['coefficient_of_variation'] = 0
                
            return stats_dict
            
        return {
            'comparisons': compute_stats(comparisons),
            'times': compute_stats(times),
            'swaps': compute_stats(swaps),
            'memory': compute_stats(memory),
            'sample_size': len(metrics_list)
        }
        
    def _confidence_interval(self, values: List[float], confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval using t-distribution"""
        if len(values) < 2:
            mean_val = values[0] if values else 0
            return (mean_val, mean_val)
            
        mean_val = statistics.mean(values)
        std_err = statistics.stdev(values) / math.sqrt(len(values))
        
        # Use t-distribution for small samples, normal for large
        if len(values) < 30:
            # Approximate t-value for common confidence levels
            t_values = {0.90: 1.833, 0.95: 2.262, 0.99: 3.250}
            t_val = t_values.get(confidence, 2.262)
        else:
            # Normal approximation for large samples
            z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            t_val = z_values.get(confidence, 1.96)
            
        margin = t_val * std_err
        return (mean_val - margin, mean_val + margin)
        
    def complexity_analysis(self, algorithm: str, distribution: str) -> Dict[str, Any]:
        """Advanced complexity analysis with multiple model fitting"""
        # Collect data points
        data_points = []
        for (alg, dist, size), metrics_list in self.results.items():
            if alg == algorithm and dist == distribution:
                comparisons = [m.comparisons for m in metrics_list]
                if comparisons:
                    mean_comparisons = statistics.mean(comparisons)
                    data_points.append((size, mean_comparisons))
                    
        if len(data_points) < 3:
            return {'error': 'Insufficient data points for analysis'}
            
        # Sort by size
        data_points.sort()
        sizes = np.array([p[0] for p in data_points])
        comparisons = np.array([p[1] for p in data_points])
        
        models = {}
        
        # Power law model: T(n) = a * n^b
        try:
            log_sizes = np.log10(sizes)
            log_comparisons = np.log10(comparisons + 1)  # +1 to avoid log(0)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_comparisons)
            
            models['power_law'] = {
                'formula': f'T(n) = {10**intercept:.2e} * n^{slope:.3f}',
                'exponent': slope,
                'coefficient': 10**intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'standard_error': std_err,
                'classification': self._classify_complexity(slope)
            }
        except Exception as e:
            models['power_law'] = {'error': str(e)}
            
        # Polynomial models (degrees 1-3) - only if sklearn available
        if HAS_SKLEARN:
            for degree in [1, 2, 3]:
                try:
                    coeffs = np.polyfit(sizes, comparisons, degree)
                    predicted = np.polyval(coeffs, sizes)
                    r_squared = r2_score(comparisons, predicted)
                    
                    models[f'polynomial_degree_{degree}'] = {
                        'coefficients': coeffs.tolist(),
                        'r_squared': r_squared,
                        'formula': self._format_polynomial(coeffs),
                        'aic': self._calculate_aic(comparisons, predicted, degree + 1)
                    }
                except Exception as e:
                    models[f'polynomial_degree_{degree}'] = {'error': str(e)}
        
        # Linearithmic model: T(n) = a * n * log(n) + b - only if sklearn available
        if HAS_SKLEARN:
            try:
                X = np.column_stack([sizes * np.log2(sizes), np.ones(len(sizes))])
                reg = LinearRegression().fit(X, comparisons)
                predicted = reg.predict(X)
                r_squared = r2_score(comparisons, predicted)
                
                models['linearithmic'] = {
                    'formula': f'T(n) = {reg.coef_[0]:.2e} * n * log(n) + {reg.coef_[1]:.2e}',
                    'coefficient_nlogn': reg.coef_[0],
                    'constant': reg.coef_[1],
                    'r_squared': r_squared
                }
            except Exception as e:
                models['linearithmic'] = {'error': str(e)}
            
        # Find best fitting model
        best_model = self._find_best_model(models)
        
        return {
            'models': models,
            'best_fit': best_model,
            'data_points': data_points,
            'goodness_of_fit_summary': self._summarize_model_quality(models)
        }
        
    def _classify_complexity(self, exponent: float) -> str:
        """Classify algorithmic complexity based on exponent"""
        if exponent < 0.5:
            return "O(1) - Constant"
        elif exponent < 1.2:
            return "O(n) - Linear"
        elif exponent < 1.8:
            return "O(n log n) - Linearithmic"
        elif exponent < 2.2:
            return "O(n²) - Quadratic"
        elif exponent < 3.2:
            return "O(n³) - Cubic"
        else:
            return f"O(n^{exponent:.1f}) - Polynomial"
            
    def _find_best_model(self, models: Dict[str, Any]) -> str:
        """Find the model with best fit based on R-squared"""
        best_model = ""
        best_r_squared = -1
        
        for model_name, model_data in models.items():
            if isinstance(model_data, dict) and 'r_squared' in model_data:
                if model_data['r_squared'] > best_r_squared:
                    best_r_squared = model_data['r_squared']
                    best_model = model_name
                    
        return best_model
        
    def _format_polynomial(self, coeffs: np.ndarray) -> str:
        """Format polynomial equation in readable form"""
        terms = []
        degree = len(coeffs) - 1
        
        for i, coeff in enumerate(coeffs):
            power = degree - i
            if abs(coeff) < 1e-10:
                continue
                
            coeff_str = f"{coeff:.2e}" if abs(coeff) >= 1000 or abs(coeff) < 0.01 else f"{coeff:.3f}"
            
            if power == 0:
                terms.append(coeff_str)
            elif power == 1:
                terms.append(f"{coeff_str}*n")
            else:
                terms.append(f"{coeff_str}*n^{power}")
                
        return " + ".join(terms) if terms else "0"
        
    def _calculate_aic(self, actual: np.ndarray, predicted: np.ndarray, num_params: int) -> float:
        """Calculate Akaike Information Criterion for model comparison"""
        n = len(actual)
        sse = np.sum((actual - predicted) ** 2)
        aic = n * np.log(sse / n) + 2 * num_params
        return aic
        
    def _summarize_model_quality(self, models: Dict[str, Any]) -> Dict[str, str]:
        """Summarize model quality for easy interpretation"""
        summary = {}
        
        for model_name, model_data in models.items():
            if isinstance(model_data, dict) and 'r_squared' in model_data:
                r_sq = model_data['r_squared']
                if r_sq >= 0.95:
                    quality = "Excellent fit"
                elif r_sq >= 0.85:
                    quality = "Good fit"
                elif r_sq >= 0.70:
                    quality = "Moderate fit"
                else:
                    quality = "Poor fit"
                    
                summary[model_name] = f"{quality} (R² = {r_sq:.3f})"
            else:
                summary[model_name] = "Analysis failed"
                
        return summary

class ComplexityAnalyzer:
    """Legacy compatibility class - use StatisticalAnalyzer for new code"""
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        
    def add_result(self, algorithm_name: str, distribution: str, input_size: int, 
                  metrics: PerformanceMetrics) -> None:
        """Add result - delegates to StatisticalAnalyzer"""
        self.statistical_analyzer.add_result(algorithm_name, distribution, input_size, metrics)
        
    def estimate_time_complexity(self, algorithm_name: str, distribution: str) -> Dict[str, float]:
        """Estimate complexity - simplified interface"""
        analysis = self.statistical_analyzer.complexity_analysis(algorithm_name, distribution)
        
        if 'models' in analysis and 'power_law' in analysis['models']:
            power_law = analysis['models']['power_law']
            if 'error' not in power_law:
                return {
                    'exponent': power_law['exponent'],
                    'coefficient': power_law['coefficient'],
                    'r_squared': power_law['r_squared']
                }
        
        return {}
    