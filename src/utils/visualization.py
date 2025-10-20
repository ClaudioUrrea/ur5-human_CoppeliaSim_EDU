"""
Visualization utilities for LBCF-MPC
Plotting functions for experimental results and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def plot_safety_trajectory(
    distances: np.ndarray,
    violations: np.ndarray,
    cbf_values: np.ndarray,
    dt: float = 0.02,
    threshold: float = 0.15,
    save_path: Optional[Path] = None
):
    """
    Plot safety metrics over time.
    
    Args:
        distances: Human-robot distances over time [n_steps]
        violations: Boolean array of violations [n_steps]
        cbf_values: CBF values over time [n_steps]
        dt: Time step (s)
        threshold: Safety distance threshold (m)
        save_path: Optional path to save figure
    """
    time = np.arange(len(distances)) * dt
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Distances
    axes[0].plot(time, distances, 'b-', linewidth=2, label='Distance', alpha=0.8)
    axes[0].axhline(threshold, color='r', linestyle='--', linewidth=2, 
                    label=f'Threshold ({threshold}m)')
    axes[0].fill_between(time, 0, threshold, alpha=0.2, color='red', 
                         label='Unsafe region')
    
    # Mark violations
    if np.any(violations):
        viol_times = time[violations]
        viol_dists = distances[violations]
        axes[0].scatter(viol_times, viol_dists, color='red', s=50, 
                       marker='x', label='Violations', zorder=5)
    
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Distance (m)', fontsize=12)
    axes[0].set_title('Human-Robot Distance Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Plot 2: CBF values
    axes[1].plot(time, cbf_values, 'g-', linewidth=2, label='CBF Value h(x)', alpha=0.8)
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2, 
                    label='Safety Boundary')
    axes[1].fill_between(time, -1, 0, alpha=0.2, color='red', 
                         label='Unsafe (h<0)')
    
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('h(x)', fontsize=12)
    axes[1].set_title('Control Barrier Function Value', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    return fig


def plot_method_comparison(
    results: Dict[str, Dict],
    metric: str = 'min_distance',
    save_path: Optional[Path] = None
):
    """
    Compare methods using box plots.
    
    Args:
        results: Dictionary mapping method names to result dictionaries
        metric: Metric to compare ('min_distance', 'violation_rate', etc.)
        save_path: Optional path to save figure
    """
    # Prepare data
    data = []
    methods = []
    
    for method, runs in results.items():
        values = [run[metric] for run in runs]
        data.extend(values)
        methods.extend([method] * len(values))
    
    df = pd.DataFrame({'Method': methods, metric: data})
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.boxplot(data=df, x='Method', y=metric, ax=ax)
    sns.swarmplot(data=df, x='Method', y=metric, color='black', 
                  alpha=0.3, size=3, ax=ax)
    
    ax.set_title(f'Method Comparison: {metric.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    return fig


def plot_pareto_front(
    throughputs: np.ndarray,
    energies: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """
    Plot Pareto front for multi-objective optimization.
    
    Args:
        throughputs: Throughput values [n_methods]
        energies: Energy consumption values [n_methods]
        labels: Method labels
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    if labels:
        for i, label in enumerate(labels):
            ax.scatter(throughputs[i], energies[i], s=200, alpha=0.7, label=label)
            ax.annotate(label, (throughputs[i], energies[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
    else:
        ax.scatter(throughputs, energies, s=200, alpha=0.7)
    
    ax.set_xlabel('Throughput (pieces/hour)', fontsize=12)
    ax.set_ylabel('Energy Consumption (kJ/hour)', fontsize=12)
    ax.set_title('Pareto Front: Throughput vs Energy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if labels:
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    return fig


def plot_conservatism_analysis(
    scenarios: List[str],
    manual_margins: List[float],
    learned_margins: List[float],
    save_path: Optional[Path] = None
):
    """
    Plot conservatism analysis across scenarios.
    
    Args:
        scenarios: Scenario names
        manual_margins: Safety margins for manual CBF
        learned_margins: Safety margins for learned CBF
        save_path: Optional path to save figure
    """
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, manual_margins, width, label='Manual CBF',
                   alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x + width/2, learned_margins, width, label='Learned CBF (LBCF-MPC)',
                   alpha=0.8, color='#4ECDC4')
    
    # Add reduction percentages
    for i in range(len(scenarios)):
        reduction = (manual_margins[i] - learned_margins[i]) / manual_margins[i] * 100
        ax.text(i, max(manual_margins[i], learned_margins[i]) + 0.02,
               f'-{reduction:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Safety Margin (m)', fontsize=12)
    ax.set_title('Conservatism Reduction: Manual vs Learned CBF', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    return fig


def plot_computational_performance(
    horizons: List[int],
    solve_times_mean: List[float],
    solve_times_std: List[float],
    save_path: Optional[Path] = None
):
    """
    Plot computational performance vs horizon length.
    
    Args:
        horizons: Horizon lengths
        solve_times_mean: Mean solve times (ms)
        solve_times_std: Std dev of solve times (ms)
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(horizons, solve_times_mean, yerr=solve_times_std,
               marker='o', markersize=10, linewidth=2, capsize=5,
               label='LBCF-MPC', alpha=0.8)
    
    # Add 20ms real-time threshold
    ax.axhline(20, color='red', linestyle='--', linewidth=2,
              label='Real-time threshold (50Hz)', alpha=0.7)
    
    ax.set_xlabel('Horizon Length H', fontsize=12)
    ax.set_ylabel('Solution Time (ms)', fontsize=12)
    ax.set_title('Computational Performance vs Horizon', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    return fig


def create_results_dashboard(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Create comprehensive dashboard of results.
    
    Args:
        results: Dictionary with all experimental results
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Safety performance
    ax1 = fig.add_subplot(gs[0, 0])
    # ... (add plots)
    
    # 2. Efficiency metrics
    ax2 = fig.add_subplot(gs[0, 1])
    # ... (add plots)
    
    # 3. Multi-objective performance
    ax3 = fig.add_subplot(gs[0, 2])
    # ... (add plots)
    
    # 4. Conservatism reduction
    ax4 = fig.add_subplot(gs[1, :2])
    # ... (add plots)
    
    # 5. Computational time
    ax5 = fig.add_subplot(gs[1, 2])
    # ... (add plots)
    
    # 6. Statistical summary
    ax6 = fig.add_subplot(gs[2, :])
    # ... (add table)
    
    fig.suptitle('LBCF-MPC Experimental Results Dashboard',
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved dashboard to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization functions...")
    
    # Test data
    n_steps = 1000
    distances = 0.3 + 0.1 * np.sin(np.linspace(0, 10, n_steps)) + np.random.randn(n_steps) * 0.02
    violations = distances < 0.15
    cbf_values = distances - 0.15 + np.random.randn(n_steps) * 0.01
    
    # Test plots
    fig1 = plot_safety_trajectory(distances, violations, cbf_values)
    plt.show()
    
    print("✓ Visualization tests passed")