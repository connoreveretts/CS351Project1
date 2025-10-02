# ============================================================================
# src/visualizer.py - Advanced Visualization Framework
# ============================================================================
"""
Professional-grade visualization framework for algorithm performance analysis.
Creates publication-quality plots with comprehensive statistical overlays.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from typing import Dict, List, Any, Optional

from analyzer import StatisticalAnalyzer
from profiler import PerformanceMetrics

class AdvancedVisualizer:
    """Professional-grade visualization for algorithm performance analysis"""
    
    def __init__(self, style: str = 'publication'):
        """Initialize visualizer with publication-ready styling"""
        if style == 'publication':
            plt.style.use(['seaborn-v0_8-darkgrid', 'seaborn-v0_8-colorblind'])
        
        # Set up color palette
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })
        
    def create_comprehensive_dashboard(self, results: Dict[str, Any], 
                                     statistical_analyzer: StatisticalAnalyzer) -> None:
        """Create comprehensive analysis dashboard with multiple visualizations"""
        
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. Algorithm performance comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_algorithm_comparison(ax1, results, 'random')
        
        # 2. Distribution sensitivity analysis  
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_distribution_sensitivity(ax2, results, self._get_first_algorithm(results))
        
        # 3. Complexity curve fitting
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_complexity_curves(ax3, statistical_analyzer, self._get_first_algorithm(results), 'random')
        
        # 4. Statistical comparison with error bars
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_statistical_comparison(ax4, statistical_analyzer, 'random', 1000)
        
        # 5. Memory usage analysis
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_memory_analysis(ax5, results)
        
        # 6. Performance scaling analysis
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_scaling_analysis(ax6, results, 'random')
        
        # 7. Best vs worst case analysis
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_best_worst_case(ax7, results, self._get_first_algorithm(results))
        
        # 8. Extrapolation predictions
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_extrapolation_analysis(ax8, statistical_analyzer)
        
        plt.suptitle('Comprehensive Sorting Algorithm Performance Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
    def _get_first_algorithm(self, results: Dict[str, Any]) -> str:
        """Helper to get first algorithm name from results"""
        return next(iter(results.keys())) if results else "Unknown"
        
    def _plot_algorithm_comparison(self, ax, results: Dict[str, Any], distribution: str) -> None:
        """Plot algorithm performance comparison for specific distribution"""
        ax.set_title(f'Algorithm Performance Comparison - {distribution.replace("_", " ").title()} Distribution', 
                    fontweight='bold', pad=20)
        
        for i, (algorithm, algo_results) in enumerate(results.items()):
            if distribution not in algo_results:
                continue
                
            sizes = []
            comparisons = []
            error_bars = []
            
            for size, size_results in algo_results[distribution].items():
                if 'statistics' in size_results and size_results['statistics']:
                    stats = size_results['statistics']
                    if 'comparisons' in stats and stats['comparisons']:
                        sizes.append(size)
                        comparisons.append(stats['comparisons']['mean'])
                        
                        # Add error bars if available
                        if 'std_dev' in stats['comparisons']:
                            error_bars.append(stats['comparisons']['std_dev'])
                        else:
                            error_bars.append(0)
                        
            if sizes and comparisons:
                ax.errorbar(sizes, comparisons, yerr=error_bars, 
                           label=algorithm, color=self.colors[i], 
                           linewidth=2, markersize=6, capsize=5, capthick=2)
                
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Input Size (n)', fontweight='bold')
        ax.set_ylabel('Average Comparisons', fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add theoretical complexity lines
        if sizes:
            x_theory = np.array(sizes)
            
            # O(n log n) reference
            nlogn_theory = x_theory * np.log2(x_theory) * (comparisons[0] / (sizes[0] * np.log2(sizes[0])))
            ax.plot(x_theory, nlogn_theory, '--', color='gray', alpha=0.7, 
                   label='O(n log n) reference')
            
            # O(n²) reference  
            n2_theory = x_theory ** 2 * (comparisons[0] / sizes[0] ** 2)
            ax.plot(x_theory, n2_theory, ':', color='gray', alpha=0.7,
                   label='O(n²) reference')
                   
        ax.legend(frameon=True, shadow=True)
        
    def _plot_distribution_sensitivity(self, ax, results: Dict[str, Any], algorithm: str) -> None:
        """Plot algorithm sensitivity to different data distributions"""
        ax.set_title(f'{algorithm}\nDistribution Sensitivity', fontweight='bold')
        
        if algorithm not in results:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
            
        algo_results = results[algorithm]
        
        for i, (distribution, dist_results) in enumerate(algo_results.items()):
            sizes = []
            comparisons = []
            
            for size, size_results in dist_results.items():
                if 'statistics' in size_results and size_results['statistics']:
                    stats = size_results['statistics']
                    if 'comparisons' in stats and stats['comparisons']:
                        sizes.append(size)
                        comparisons.append(stats['comparisons']['mean'])
                        
            if sizes and comparisons:
                label = distribution.replace('_', ' ').title()
                ax.loglog(sizes, comparisons, 'o-', label=label, 
                         color=self.colors[i], linewidth=2, markersize=4)
                
        ax.set_xlabel('Input Size (n)', fontweight='bold')
        ax.set_ylabel('Average Comparisons', fontweight='bold')
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)
        
    def _plot_complexity_curves(self, ax, statistical_analyzer: StatisticalAnalyzer, 
                               algorithm: str, distribution: str) -> None:
        """Plot complexity curve fitting results with model equations"""
        ax.set_title(f'{algorithm}\nComplexity Analysis', fontweight='bold')
        
        analysis = statistical_analyzer.complexity_analysis(algorithm, distribution)
        
        if 'error' in analysis or 'data_points' not in analysis:
            ax.text(0.5, 0.5, 'Insufficient data\nfor analysis', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
            
        data_points = analysis['data_points']
        sizes, comparisons = zip(*data_points)
        
        # Plot actual data points
        ax.loglog(sizes, comparisons, 'o', label='Measured Data', 
                 color='black', markersize=8, zorder=3)
        
        # Plot fitted models
        if 'models' in analysis:
            models = analysis['models']
            x_smooth = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
            
            # Power law fit
            if 'power_law' in models and 'error' not in models['power_law']:
                model = models['power_law']
                y_smooth = model['coefficient'] * (x_smooth ** model['exponent'])
                
                label = (f"Power Law: {model['formula']}\n"
                        f"R² = {model['r_squared']:.4f}\n"
                        f"Classification: {model['classification']}")
                
                ax.loglog(x_smooth, y_smooth, '--', label=label, 
                         linewidth=3, alpha=0.8, color='red')
                         
            # Linearithmic fit if available
            if 'linearithmic' in models and 'error' not in models['linearithmic']:
                model = models['linearithmic']
                y_smooth = model['coefficient_nlogn'] * x_smooth * np.log2(x_smooth) + model['constant']
                
                ax.loglog(x_smooth, y_smooth, '-.', 
                         label=f"n log n fit (R² = {model['r_squared']:.3f})", 
                         linewidth=2, alpha=0.8, color='blue')
                
        ax.set_xlabel('Input Size (n)', fontweight='bold')
        ax.set_ylabel('Comparisons', fontweight='bold')
        ax.legend(fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3)
        
    def _plot_statistical_comparison(self, ax, statistical_analyzer: StatisticalAnalyzer, 
                                   distribution: str, size: int) -> None:
        """Plot statistical comparison with confidence intervals"""
        ax.set_title(f'Statistical Comparison\n(Size {size}, {distribution.title()})', fontweight='bold')
        
        # Find all algorithms for this distribution and size
        algorithms = set()
        for (alg, dist, sz), _ in statistical_analyzer.results.items():
            if dist == distribution and sz == size:
                algorithms.add(alg)
                
        algorithms = sorted(algorithms)
        
        if not algorithms:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        means = []
        errors = []
        labels = []
        
        for algorithm in algorithms:
            stats = statistical_analyzer.compute_statistics(algorithm, distribution, size)
            if stats and 'comparisons' in stats and stats['comparisons']:
                comp_stats = stats['comparisons']
                mean = comp_stats['mean']
                
                # Calculate error bar from confidence interval
                if 'confidence_interval_95' in comp_stats:
                    ci_low, ci_high = comp_stats['confidence_interval_95']
                    error = ci_high - mean
                else:
                    error = comp_stats.get('std_dev', 0)
                
                means.append(mean)
                errors.append(error)
                labels.append(algorithm.replace(' ', '\n'))
                
        if means:
            bars = ax.bar(range(len(means)), means, yerr=errors, 
                         capsize=8, alpha=0.7, color=self.colors[:len(means)],
                         edgecolor='black', linewidth=1)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Comparisons', fontweight='bold')
            
            # Add value labels on bars
            for bar, mean, error in zip(bars, means, errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + error + max(errors)*0.05,
                       f'{mean:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                       
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_memory_analysis(self, ax, results: Dict[str, Any]) -> None:
        """Plot memory usage analysis across algorithms"""
        ax.set_title('Peak Memory Usage Analysis', fontweight='bold')
        
        for i, (algorithm, algo_results) in enumerate(results.items()):
            if 'random' not in algo_results:
                continue
                
            sizes = []
            memory_usage = []
            
            for size, size_results in algo_results['random'].items():
                if 'trials' in size_results and size_results['trials']:
                    trials = size_results['trials']
                    avg_memory = sum(t.peak_memory_usage for t in trials) / len(trials)
                    sizes.append(size)
                    memory_usage.append(max(1, avg_memory / 1024))  # Convert to KB, min 1
                    
            if sizes and memory_usage:
                ax.plot(sizes, memory_usage, 'o-', label=algorithm, 
                       color=self.colors[i], linewidth=2, markersize=5)
                
        ax.set_xlabel('Input Size (n)', fontweight='bold')
        ax.set_ylabel('Peak Memory Usage (KB)', fontweight='bold')
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)
        
        # Add theoretical O(1) space line if data exists
        if sizes:
            ax.axhline(y=memory_usage[0] if memory_usage else 1, color='gray', 
                      linestyle='--', alpha=0.5, label='O(1) space reference')
            ax.legend(fontsize=9, frameon=True)
        
    def _plot_scaling_analysis(self, ax, results: Dict[str, Any], distribution: str) -> None:
        """Plot scaling factor analysis (how performance changes with doubling input size)"""
        ax.set_title(f'Performance Scaling Analysis\n{distribution.replace("_", " ").title()} Distribution', 
                    fontweight='bold')
        
        for i, (algorithm, algo_results) in enumerate(results.items()):
            if distribution not in algo_results:
                continue
                
            sizes = []
            scaling_factors = []
            
            sorted_sizes = sorted(algo_results[distribution].keys())
            
            for j in range(1, len(sorted_sizes)):
                prev_size = sorted_sizes[j-1]
                curr_size = sorted_sizes[j]
                
                prev_results = algo_results[distribution][prev_size]
                curr_results = algo_results[distribution][curr_size]
                
                if ('statistics' in prev_results and 'statistics' in curr_results and
                    prev_results['statistics'] and curr_results['statistics']):
                    
                    prev_comps = prev_results['statistics']['comparisons']['mean']
                    curr_comps = curr_results['statistics']['comparisons']['mean']
                    
                    size_ratio = curr_size / prev_size
                    perf_ratio = curr_comps / prev_comps if prev_comps > 0 else 1
                    
                    # Calculate scaling factor
                    scaling_factor = perf_ratio / size_ratio
                    
                    sizes.append(curr_size)
                    scaling_factors.append(scaling_factor)
                    
            if sizes and scaling_factors:
                ax.semilogx(sizes, scaling_factors, 'o-', label=algorithm, 
                           color=self.colors[i], linewidth=2, markersize=5)
                
        # Add theoretical scaling lines
        if sizes:
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, 
                      label='O(n) scaling')
            ax.axhline(y=np.log2(2), color='blue', linestyle='--', alpha=0.7, 
                      label='O(n log n) scaling')
            ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, 
                      label='O(n²) scaling')
                
        ax.set_xlabel('Input Size (n)', fontweight='bold')
        ax.set_ylabel('Scaling Factor', fontweight='bold')
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)
        
    def _plot_best_worst_case(self, ax, results: Dict[str, Any], algorithm: str) -> None:
        """Plot best vs worst case performance for an algorithm"""
        ax.set_title(f'{algorithm}\nBest vs Worst Case Performance', fontweight='bold')
        
        if algorithm not in results:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
            
        algo_results = results[algorithm]
        
        # Define best and worst case distributions
        best_case_dist = 'sorted'  # Usually best for most algorithms
        worst_case_dist = 'reverse_sorted'  # Usually worst
        
        for case_name, dist, color in [('Best Case', best_case_dist, 'green'), 
                                      ('Worst Case', worst_case_dist, 'red')]:
            if dist not in algo_results:
                continue
                
            sizes = []
            comparisons = []
            
            for size, size_results in algo_results[dist].items():
                if 'statistics' in size_results and size_results['statistics']:
                    stats = size_results['statistics']
                    if 'comparisons' in stats and stats['comparisons']:
                        sizes.append(size)
                        comparisons.append(stats['comparisons']['mean'])
                        
            if sizes and comparisons:
                ax.loglog(sizes, comparisons, 'o-', label=f'{case_name} ({dist})', 
                         color=color, linewidth=2, markersize=5)
                
        ax.set_xlabel('Input Size (n)', fontweight='bold')
        ax.set_ylabel('Average Comparisons', fontweight='bold')
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)
        
    def _plot_extrapolation_analysis(self, ax, statistical_analyzer: StatisticalAnalyzer) -> None:
        """Plot extrapolation predictions for large datasets with uncertainty bands"""
        ax.set_title('Performance Extrapolation for Large Datasets', fontweight='bold', pad=20)
        
        # Choose first algorithm with sufficient data
        algorithm = None
        distribution = 'random'
        
        for (alg, dist, size), metrics_list in statistical_analyzer.results.items():
            if dist == distribution and len(metrics_list) > 0:
                algorithm = alg
                break
                
        if not algorithm:
            ax.text(0.5, 0.5, 'Insufficient data\nfor extrapolation', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
            
        analysis = statistical_analyzer.complexity_analysis(algorithm, distribution)
        
        if 'error' in analysis or 'data_points' not in analysis:
            ax.text(0.5, 0.5, 'Analysis failed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
            
        # Current data points
        data_points = analysis['data_points']
        sizes, comparisons = zip(*data_points)
        
        # Extrapolation range
        max_current = max(sizes)
        extrapolation_sizes = np.logspace(np.log10(max_current), 
                                        np.log10(max_current * 1000), 100)
        
        # Plot current data
        ax.loglog(sizes, comparisons, 'o', label='Measured Data', 
                 color='black', markersize=10, zorder=3)
        
        # Plot extrapolations with uncertainty
        if 'models' in analysis and 'power_law' in analysis['models']:
            model = analysis['models']['power_law']
            if 'error' not in model:
                extrapolated = model['coefficient'] * (extrapolation_sizes ** model['exponent'])
                
                # Main extrapolation line
                ax.loglog(extrapolation_sizes, extrapolated, '--', 
                         label=f"Extrapolation: {model['formula']}", 
                         linewidth=3, alpha=0.9, color='red')
                         
                # Uncertainty bands (±50% as rough estimate)
                uncertainty_factor = 2.0
                lower_bound = extrapolated / uncertainty_factor
                upper_bound = extrapolated * uncertainty_factor
                
                ax.fill_between(extrapolation_sizes, lower_bound, upper_bound,
                               alpha=0.2, color='red', label='Uncertainty Range (±100%)')
                               
        # Add practical size markers
        practical_sizes = [10**5, 10**6, 10**7, 10**8]
        for ps in practical_sizes:
            if ps <= max(extrapolation_sizes) and ps >= max_current:
                ax.axvline(ps, color='gray', linestyle=':', alpha=0.7)
                ax.text(ps, ax.get_ylim()[1] * 0.1, f'{ps:,.0e}', 
                       rotation=90, ha='right', va='bottom', fontsize=9)
                       
        ax.set_xlabel('Input Size (n)', fontweight='bold')
        ax.set_ylabel('Predicted Comparisons', fontweight='bold')
        ax.legend(fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3)
        
        # Add performance estimates
        textstr = f'Algorithm: {algorithm}\nModel R²: {model.get("r_squared", 0):.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    def save_all_plots(self, results: Dict[str, Any], statistical_analyzer: StatisticalAnalyzer, 
                      output_dir: str = 'results/plots') -> None:
        """Save individual plots for inclusion in reports"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Individual algorithm comparison plots
        for distribution in ['random', 'sorted', 'reverse_sorted']:
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_algorithm_comparison(ax, results, distribution)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/algorithm_comparison_{distribution}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Individual complexity analysis plots
        for algorithm in results.keys():
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_complexity_curves(ax, statistical_analyzer, algorithm, 'random')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/complexity_analysis_{algorithm.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"Individual plots saved to {output_dir}/")