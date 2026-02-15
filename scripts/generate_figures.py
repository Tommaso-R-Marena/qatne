#!/usr/bin/env python3
"""
Generate publication-quality figures from QATNE results

Usage:
    python scripts/generate_figures.py --input results/benchmark.json --output figures/
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_results(filepath):
    """Load benchmark results from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_convergence(results, output_dir):
    """
    Plot energy convergence for all molecules
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (molecule, data) in enumerate(results.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Plot all trials
        for trial in data['trials']:
            energy_history = trial['energy_history']
            ax.plot(energy_history, alpha=0.3, color='blue')
        
        # Plot mean
        all_histories = [t['energy_history'] for t in data['trials']]
        min_len = min(len(h) for h in all_histories)
        histories_array = np.array([h[:min_len] for h in all_histories])
        mean_history = np.mean(histories_array, axis=0)
        
        ax.plot(mean_history, color='red', linewidth=2, label='Mean')
        ax.axhline(
            data['trials'][0]['exact_energy'],
            color='green',
            linestyle='--',
            linewidth=2,
            label='Exact'
        )
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Ha)')
        ax.set_title(f'{molecule} Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: convergence_curves.png")


def plot_error_comparison(results, output_dir):
    """
    Compare errors across molecules
    """
    molecules = []
    mean_errors = []
    std_errors = []
    
    for molecule, data in results.items():
        molecules.append(molecule)
        mean_errors.append(data['statistics']['mean_error'])
        std_errors.append(data['statistics']['std_error'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(molecules))
    ax.bar(x_pos, mean_errors, yerr=std_errors, capsize=5,
           color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(molecules)
    ax.set_ylabel('Absolute Error (Ha)')
    ax.set_title('QATNE Accuracy Across Molecules')
    ax.axhline(1.6e-3, color='r', linestyle='--', linewidth=2, label='Chemical accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: error_comparison.png")


def plot_timing_analysis(results, output_dir):
    """
    Analyze computational timing
    """
    molecules = []
    qubits = []
    times = []
    
    for molecule, data in results.items():
        molecules.append(molecule)
        qubits.append(data['trials'][0]['num_qubits'])
        times.append(data['statistics']['mean_time'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time by molecule
    ax = axes[0]
    ax.bar(molecules, times, color='coral', edgecolor='black', alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Time by Molecule')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Scaling with qubits
    ax = axes[1]
    ax.plot(qubits, times, 'bo-', markersize=10, linewidth=2)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Scaling with System Size')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: timing_analysis.png")


def plot_statistical_distributions(results, output_dir):
    """
    Plot error distributions for each molecule
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (molecule, data) in enumerate(results.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        errors = [trial['error'] for trial in data['trials']]
        
        ax.hist(errors, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(data['statistics']['mean_error'], color='red',
                   linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Absolute Error (Ha)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{molecule} Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: error_distributions.png")


def plot_gradient_evolution(results, output_dir):
    """
    Plot gradient norm evolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (molecule, data) in enumerate(results.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Plot gradient norms for all trials
        for trial in data['trials']:
            gradient_norms = trial['gradient_norms']
            ax.semilogy(gradient_norms, alpha=0.3, color='orange')
        
        # Mean gradient norm
        all_grads = [t['gradient_norms'] for t in data['trials']]
        min_len = min(len(g) for g in all_grads)
        grads_array = np.array([g[:min_len] for g in all_grads])
        mean_grads = np.mean(grads_array, axis=0)
        
        ax.semilogy(mean_grads, color='red', linewidth=2, label='Mean')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'{molecule} Gradient Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: gradient_evolution.png")


def generate_summary_table(results, output_dir):
    """
    Generate LaTeX summary table
    """
    table_lines = []
    table_lines.append(r"\begin{table}[h]")
    table_lines.append(r"\centering")
    table_lines.append(r"\begin{tabular}{lccccc}")
    table_lines.append(r"\hline")
    table_lines.append(r"Molecule & Qubits & $E_{\text{exact}}$ (Ha) & $E_{\text{QATNE}}$ (Ha) & Error (Ha) & Time (s) \\")
    table_lines.append(r"\hline")
    
    for molecule, data in results.items():
        stats = data['statistics']
        trial = data['trials'][0]
        
        line = f"{molecule} & {trial['num_qubits']} & "
        line += f"{trial['exact_energy']:.6f} & "
        line += f"{stats['mean_energy']:.6f} & "
        line += f"{stats['mean_error']:.2e} & "
        line += f"{stats['mean_time']:.1f} \\\\"
        
        table_lines.append(line)
    
    table_lines.append(r"\hline")
    table_lines.append(r"\end{tabular}")
    table_lines.append(r"\caption{QATNE benchmark results on molecular systems.}")
    table_lines.append(r"\label{tab:qatne_benchmarks}")
    table_lines.append(r"\end{table}")
    
    latex_file = output_dir / 'summary_table.tex'
    with open(latex_file, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"Saved: summary_table.tex")


def main():
    parser = argparse.ArgumentParser(
        description='Generate figures from QATNE benchmark results'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file with benchmark results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures',
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    data = load_results(args.input)
    results = data['results']
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating figures...")
    
    # Generate all figures
    plot_convergence(results, output_dir)
    plot_error_comparison(results, output_dir)
    plot_timing_analysis(results, output_dir)
    plot_statistical_distributions(results, output_dir)
    plot_gradient_evolution(results, output_dir)
    generate_summary_table(results, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
