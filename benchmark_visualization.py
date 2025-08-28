#!/usr/bin/env python3
"""
P3M Benchmark Visualization Script

This script reads benchmark data from CSV files and generates comprehensive
visualization plots to analyze the performance and accuracy of the P3M method
compared to direct summation.

Requirements:
- Python 3.6+
- numpy
- pandas
- matplotlib
- seaborn (optional, for better styling)

Usage:
    python benchmark_visualization.py

The script will look for the following CSV files:
- performance.csv: Performance benchmarks across different particle counts
- precision.csv: Precision analysis for different alpha values
- scaling.csv: Scaling analysis for computational complexity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up matplotlib parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Try to import seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not available - using matplotlib defaults")

class P3MBenchmarkVisualizer:
    """Main class for P3M benchmark visualization"""

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.data_frames = {}

        # Color scheme for consistent plotting
        self.colors = {
            'Direct': '#e74c3c',      # Red
            'P3M': '#3498db',         # Blue
            'P3M_16': '#1abc9c',      # Teal
            'P3M_32': '#3498db',      # Blue
            'P3M_64': '#9b59b6',      # Purple
        }

        # Marker styles
        self.markers = {
            'Direct': 'o',
            'P3M': 's',
            'P3M_16': '^',
            'P3M_32': 's',
            'P3M_64': 'D',
        }

    def load_data(self) -> bool:
        """Load all benchmark data files"""
        required_files = ['performance.csv', 'precision.csv', 'scaling.csv']

        for filename in required_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    self.data_frames[filename.replace('.csv', '')] = df
                    print(f"✓ Loaded {filename} ({len(df)} records)")
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
                    return False
            else:
                print(f"✗ File not found: {filename}")
                return False

        return True

    def create_performance_plots(self) -> plt.Figure:
        """Create performance comparison plots"""
        if 'performance' not in self.data_frames:
            raise ValueError("Performance data not available")

        df = self.data_frames['performance']
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # Plot 1: Execution time vs particle count
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_execution_time(ax1, df)

        # Plot 2: Memory usage vs particle count
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_memory_usage(ax2, df)

        # Plot 3: Performance ratio (Direct/P3M)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_performance_ratio(ax3, df)

        # Plot 4: Energy conservation
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_energy_conservation(ax4, df)

        fig.suptitle('P3M Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_precision_plots(self) -> plt.Figure:
        """Create precision analysis plots"""
        if 'precision' not in self.data_frames:
            raise ValueError("Precision data not available")

        df = self.data_frames['precision']
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig)

        # Plot 1: Max error vs alpha
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_error_vs_alpha(ax1, df, 'maxError', 'Maximum Error')

        # Plot 2: RMS error vs alpha
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_error_vs_alpha(ax2, df, 'rmsError', 'RMS Error')

        # Plot 3: Execution time vs alpha
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_time_vs_alpha(ax3, df)

        fig.suptitle('P3M Precision Analysis vs Ewald Alpha Parameter', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_scaling_plots(self) -> plt.Figure:
        """Create scaling analysis plots"""
        if 'scaling' not in self.data_frames:
            raise ValueError("Scaling data not available")

        df = self.data_frames['scaling']
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig)

        # Plot 1: Time scaling (linear scale)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_scaling_linear(ax1, df)

        # Plot 2: Time scaling (log-log scale)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_scaling_loglog(ax2, df)

        # Plot 3: Complexity analysis
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_complexity_analysis(ax3, df)

        fig.suptitle('Computational Complexity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def _plot_execution_time(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot execution time vs particle count"""
        methods = df['method'].unique()

        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                color = self.colors.get(method, 'black')
                marker = self.markers.get(method, 'o')

                ax.scatter(method_data['numParticles'], method_data['executionTime'],
                          color=color, marker=marker, s=50, alpha=0.7, label=method)

                # Add trend line for P3M methods
                if method.startswith('P3M'):
                    self._add_trend_line(ax, method_data['numParticles'], method_data['executionTime'],
                                       color, method)

        ax.set_xlabel('Number of Particles')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Execution Time vs Particle Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

    def _plot_memory_usage(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot memory usage vs particle count"""
        methods = df['method'].unique()

        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                color = self.colors.get(method, 'black')
                marker = self.markers.get(method, 'o')

                ax.scatter(method_data['numParticles'], method_data['memoryUsage'] / 1024 / 1024,
                          color=color, marker=marker, s=50, alpha=0.7, label=method)

        ax.set_xlabel('Number of Particles')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage vs Particle Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    def _plot_performance_ratio(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot performance ratio between Direct and P3M"""
        # Group by particle count and calculate ratios
        ratios = []
        particle_counts = []

        for particles in df['numParticles'].unique():
            particle_data = df[df['numParticles'] == particles]

            direct_time = particle_data[particle_data['method'] == 'Direct']['executionTime']
            p3m_times = particle_data[particle_data['method'].str.startswith('P3M')]['executionTime']

            if len(direct_time) > 0 and len(p3m_times) > 0:
                direct_avg = direct_time.mean()
                p3m_avg = p3m_times.mean()
                ratio = direct_avg / p3m_avg

                ratios.append(ratio)
                particle_counts.append(particles)

        if ratios:
            ax.bar(range(len(ratios)), ratios, color=self.colors['P3M'], alpha=0.7)
            ax.set_xlabel('Particle Count')
            ax.set_ylabel('Speedup (Direct/P3M)')
            ax.set_title('P3M Speedup vs Direct Summation')
            ax.set_xticks(range(len(ratios)))
            ax.set_xticklabels([f'{int(x):,}' for x in particle_counts])
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, ratio in enumerate(ratios):
                ax.text(i, ratio + 0.1, '.1f', ha='center', va='bottom')

    def _plot_energy_conservation(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot energy conservation analysis"""
        methods = df['method'].unique()

        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                color = self.colors.get(method, 'black')
                marker = self.markers.get(method, 'o')

                total_energy = method_data['potentialEnergy'] + method_data['kineticEnergy']

                ax.scatter(method_data['numParticles'], total_energy.abs(),
                          color=color, marker=marker, s=50, alpha=0.7, label=method)

        ax.set_xlabel('Number of Particles')
        ax.set_ylabel('|Total Energy|')
        ax.set_title('Energy Conservation Check')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

    def _plot_error_vs_alpha(self, ax: plt.Axes, df: pd.DataFrame, error_col: str, title: str):
        """Plot error vs alpha parameter"""
        grid_sizes = df['gridSize'].unique()

        for grid_size in grid_sizes:
            grid_data = df[df['gridSize'] == grid_size]
            if len(grid_data) > 0:
                color = self.colors.get(f'P3M_{grid_size}', 'black')
                marker = self.markers.get(f'P3M_{grid_size}', 'o')

                ax.scatter(grid_data['alpha'], grid_data[error_col],
                          color=color, marker=marker, s=50, alpha=0.7,
                          label=f'Grid {grid_size}³')

                # Add trend line
                if len(grid_data) > 2:
                    z = np.polyfit(grid_data['alpha'], grid_data[error_col], 2)
                    p = np.poly1d(z)
                    x_trend = np.linspace(grid_data['alpha'].min(), grid_data['alpha'].max(), 100)
                    ax.plot(x_trend, p(x_trend), color=color, alpha=0.5, linestyle='--')

        ax.set_xlabel('Ewald Alpha Parameter')
        ax.set_ylabel(error_col.replace('Error', ' Error'))
        ax.set_title(f'{title} vs Alpha')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_time_vs_alpha(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot execution time vs alpha parameter"""
        grid_sizes = df['gridSize'].unique()

        for grid_size in grid_sizes:
            grid_data = df[df['gridSize'] == grid_size]
            if len(grid_data) > 0:
                color = self.colors.get(f'P3M_{grid_size}', 'black')
                marker = self.markers.get(f'P3M_{grid_size}', 'o')

                ax.scatter(grid_data['alpha'], grid_data['executionTime'],
                          color=color, marker=marker, s=50, alpha=0.7,
                          label=f'Grid {grid_size}³')

        ax.set_xlabel('Ewald Alpha Parameter')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Execution Time vs Alpha')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_scaling_linear(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot scaling analysis (linear scale)"""
        methods = df['method'].unique()

        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                color = self.colors.get(method, 'black')
                marker = self.markers.get(method, 'o')

                ax.scatter(method_data['numParticles'], method_data['executionTime'],
                          color=color, marker=marker, s=50, alpha=0.7, label=method)

        ax.set_xlabel('Number of Particles')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Scaling Analysis (Linear Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_scaling_loglog(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot scaling analysis (log-log scale)"""
        methods = df['method'].unique()

        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                color = self.colors.get(method, 'black')
                marker = self.markers.get(method, 'o')

                ax.scatter(method_data['numParticles'], method_data['executionTime'],
                          color=color, marker=marker, s=50, alpha=0.7, label=method)

                # Add trend line and complexity annotation
                if len(method_data) > 2:
                    self._add_scaling_trend(ax, method_data['numParticles'], method_data['executionTime'],
                                          color, method)

        ax.set_xlabel('Number of Particles')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Scaling Analysis (Log-Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

    def _plot_complexity_analysis(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot computational complexity analysis"""
        methods = df['method'].unique()
        complexities = []

        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) >= 3:
                # Fit power law: time = c * N^d
                x = np.log(method_data['numParticles'].values)
                y = np.log(method_data['executionTime'].values)

                # Remove any infinite or NaN values
                valid_idx = np.isfinite(x) & np.isfinite(y)
                x = x[valid_idx]
                y = y[valid_idx]

                if len(x) >= 3:
                    coeffs = np.polyfit(x, y, 1)
                    complexity = coeffs[0]  # slope = d in O(N^d)

                    complexities.append({
                        'method': method,
                        'complexity': complexity,
                        'color': self.colors.get(method, 'black')
                    })

        if complexities:
            methods_list = [c['method'] for c in complexities]
            complexity_values = [c['complexity'] for c in complexities]
            colors_list = [c['color'] for c in complexities]

            bars = ax.bar(methods_list, complexity_values, color=colors_list, alpha=0.7)

            ax.set_xlabel('Method')
            ax.set_ylabel('Complexity Exponent (d in O(N^d))')
            ax.set_title('Computational Complexity Analysis')
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, complexity_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       '.2f', ha='center', va='bottom')

            # Add reference lines
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='O(N)')
            ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='O(N²)')
            ax.axhline(y=np.log2(10), color='green', linestyle='--', alpha=0.5, label='O(N log N)')

            ax.legend()

    def _add_trend_line(self, ax: plt.Axes, x: pd.Series, y: pd.Series, color: str, label: str):
        """Add trend line to plot"""
        if len(x) < 2:
            return

        x_vals = x.values
        y_vals = y.values

        # Fit polynomial trend
        try:
            z = np.polyfit(x_vals, y_vals, 2)
            p = np.poly1d(z)

            x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_trend, p(x_trend), color=color, alpha=0.5, linestyle='--')
        except:
            pass  # Skip trend line if fitting fails

    def _add_scaling_trend(self, ax: plt.Axes, x: pd.Series, y: pd.Series, color: str, method: str):
        """Add scaling trend line with complexity annotation"""
        if len(x) < 3:
            return

        x_vals = x.values
        y_vals = y.values

        # Fit power law in log space
        try:
            log_x = np.log(x_vals)
            log_y = np.log(y_vals)

            # Remove any infinite or NaN values
            valid_idx = np.isfinite(log_x) & np.isfinite(log_y)
            log_x = log_x[valid_idx]
            log_y = log_y[valid_idx]

            if len(log_x) >= 2:
                coeffs = np.polyfit(log_x, log_y, 1)
                slope = coeffs[0]

                # Generate trend line
                x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_trend = np.exp(coeffs[1]) * np.power(x_trend, slope)

                ax.plot(x_trend, y_trend, color=color, alpha=0.5, linestyle='--')

                # Add complexity annotation
                x_pos = x_vals[int(len(x_vals) * 0.7)]
                y_pos = np.exp(coeffs[1]) * np.power(x_pos, slope)

                complexity_str = ""
                if method == "Direct":
                    complexity_str = "O(N²)"
                elif method.startswith("P3M"):
                    if slope < 1.5:
                        complexity_str = "O(N)"
                    elif slope < 1.8:
                        complexity_str = "O(N log N)"
                    else:
                        complexity_str = ".1f"

                ax.annotate(complexity_str, (x_pos, y_pos), xytext=(10, 10),
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        except Exception as e:
            print(f"Warning: Could not fit scaling trend for {method}: {e}")

    def save_plots(self, output_dir: str = "benchmark_plots"):
        """Generate and save all plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        try:
            # Performance plots
            fig1 = self.create_performance_plots()
            fig1.savefig(output_path / "performance_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)
            print(f"✓ Saved performance analysis to {output_path / 'performance_analysis.png'}")

            # Precision plots
            fig2 = self.create_precision_plots()
            fig2.savefig(output_path / "precision_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"✓ Saved precision analysis to {output_path / 'precision_analysis.png'}")

            # Scaling plots
            fig3 = self.create_scaling_plots()
            fig3.savefig(output_path / "scaling_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
            print(f"✓ Saved scaling analysis to {output_path / 'scaling_analysis.png'}")

        except Exception as e:
            print(f"✗ Error generating plots: {e}")
            return False

        return True

    def print_summary_statistics(self):
        """Print summary statistics from the benchmark data"""
        print("\n" + "="*60)
        print("P3M BENCHMARK SUMMARY STATISTICS")
        print("="*60)

        for name, df in self.data_frames.items():
            print(f"\n{name.upper()} DATASET:")
            print(f"  Records: {len(df)}")
            print(f"  Methods: {', '.join(df['method'].unique())}")
            print(f"  Particle counts: {sorted(df['numParticles'].unique())}")

            if 'executionTime' in df.columns:
                print("  Execution time statistics:")
                for method in df['method'].unique():
                    method_data = df[df['method'] == method]['executionTime']
                    if len(method_data) > 0:
                        print(".2f"
                              ".2f"
                              ".2f")

            if 'maxError' in df.columns and 'rmsError' in df.columns:
                valid_errors = df[(df['maxError'] >= 0) & (df['rmsError'] >= 0)]
                if len(valid_errors) > 0:
                    print("  Error statistics:")
                    print(".6f")
                    print(".6f")

        print("\n" + "="*60)


def main():
    """Main function"""
    print("P3M Benchmark Visualization Tool")
    print("=" * 40)

    # Initialize visualizer
    visualizer = P3MBenchmarkVisualizer()

    # Load data
    if not visualizer.load_data():
        print("❌ Failed to load benchmark data files")
        print("Make sure the following files exist:")
        print("  - performance.csv")
        print("  - precision.csv")
        print("  - scaling.csv")
        sys.exit(1)

    # Print summary statistics
    visualizer.print_summary_statistics()

    # Generate and save plots
    print("\nGenerating visualization plots...")
    if visualizer.save_plots():
        print("✅ All plots generated successfully!")
        print("📁 Plots saved to: benchmark_plots/")
        print("   - performance_analysis.png")
        print("   - precision_analysis.png")
        print("   - scaling_analysis.png")
    else:
        print("❌ Failed to generate plots")
        sys.exit(1)

    print("\n🎉 Benchmark analysis complete!")
    print("Use the generated plots to analyze P3M performance and accuracy.")


if __name__ == "__main__":
    main()