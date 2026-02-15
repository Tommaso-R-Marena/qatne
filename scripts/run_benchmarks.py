#!/usr/bin/env python3
"""
Run comprehensive QATNE benchmarks

Usage:
    python scripts/run_benchmarks.py --molecules H2 LiH --trials 10 --output results/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from qatne.algorithms import QATNESolver
    from qatne.benchmarks.molecular_systems import (
        h2_molecule, lih_molecule, beh2_molecule, h2o_molecule
    )
except ImportError:
    print("Error: qatne package not found. Install with: pip install -e .")
    sys.exit(1)


MOLECULE_REGISTRY = {
    'H2': h2_molecule,
    'LiH': lih_molecule,
    'BeH2': beh2_molecule,
    'H2O': h2o_molecule
}


def run_single_benchmark(
    molecule_name,
    hamiltonian,
    num_qubits,
    exact_energy,
    config
):
    """
    Run QATNE on single molecular system
    """
    print(f"\nRunning {molecule_name}...")
    print(f"  Qubits: {num_qubits}")
    print(f"  Exact energy: {exact_energy:.8f} Ha")
    
    # Create solver
    solver = QATNESolver(
        hamiltonian=hamiltonian,
        num_qubits=num_qubits,
        max_bond_dim=config['max_bond_dim'],
        convergence_threshold=config['convergence_threshold'],
        shots=config['shots']
    )
    
    # Run optimization
    import time
    start_time = time.time()
    
    ground_energy, optimal_params = solver.solve(
        max_iterations=config['max_iterations']
    )
    
    elapsed_time = time.time() - start_time
    
    # Compute metrics
    error = abs(ground_energy - exact_energy)
    rel_error = error / abs(exact_energy)
    
    result = {
        'molecule': molecule_name,
        'num_qubits': num_qubits,
        'exact_energy': float(exact_energy),
        'qatne_energy': float(ground_energy),
        'error': float(error),
        'relative_error': float(rel_error),
        'time_seconds': float(elapsed_time),
        'iterations': len(solver.energy_history),
        'final_bond_dim': solver.tensor_network.bond_dim,
        'energy_history': [float(e) for e in solver.energy_history],
        'gradient_norms': [float(g) for g in solver.gradient_norms]
    }
    
    print(f"  QATNE energy: {ground_energy:.8f} Ha")
    print(f"  Error: {error:.2e} Ha")
    print(f"  Time: {elapsed_time:.2f} s")
    
    return result


def run_multiple_trials(
    molecule_name,
    generator_func,
    num_trials,
    config
):
    """
    Run multiple trials for statistical analysis
    """
    print(f"\n{'='*60}")
    print(f"Running {num_trials} trials on {molecule_name}")
    print(f"{'='*60}")
    
    # Generate molecule
    hamiltonian, num_qubits, exact_energy = generator_func()
    
    results = []
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Different random seed each trial
        np.random.seed(trial + 1000)
        
        result = run_single_benchmark(
            molecule_name,
            hamiltonian,
            num_qubits,
            exact_energy,
            config
        )
        result['trial_id'] = trial
        results.append(result)
    
    # Compute statistics
    energies = np.array([r['qatne_energy'] for r in results])
    errors = np.array([r['error'] for r in results])
    times = np.array([r['time_seconds'] for r in results])
    
    statistics = {
        'molecule': molecule_name,
        'num_trials': num_trials,
        'mean_energy': float(np.mean(energies)),
        'std_energy': float(np.std(energies)),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times))
    }
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics for {molecule_name}")
    print(f"{'='*60}")
    print(f"Mean energy: {statistics['mean_energy']:.8f} ± {statistics['std_energy']:.2e} Ha")
    print(f"Mean error:  {statistics['mean_error']:.2e} ± {statistics['std_error']:.2e} Ha")
    print(f"Mean time:   {statistics['mean_time']:.2f} ± {statistics['std_time']:.2f} s")
    
    return {
        'trials': results,
        'statistics': statistics
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run QATNE benchmarks on molecular systems'
    )
    parser.add_argument(
        '--molecules',
        nargs='+',
        default=['H2'],
        choices=list(MOLECULE_REGISTRY.keys()),
        help='Molecules to benchmark'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=5,
        help='Number of trials per molecule'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=200,
        help='Maximum optimization iterations'
    )
    parser.add_argument(
        '--max-bond-dim',
        type=int,
        default=16,
        help='Maximum tensor network bond dimension'
    )
    parser.add_argument(
        '--shots',
        type=int,
        default=8192,
        help='Number of measurement shots'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'max_iterations': args.max_iterations,
        'max_bond_dim': args.max_bond_dim,
        'convergence_threshold': 1e-6,
        'shots': args.shots
    }
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    timestamp = datetime.now().isoformat()
    
    for molecule_name in args.molecules:
        if molecule_name not in MOLECULE_REGISTRY:
            print(f"Warning: Unknown molecule {molecule_name}, skipping")
            continue
        
        generator_func = MOLECULE_REGISTRY[molecule_name]
        
        results = run_multiple_trials(
            molecule_name,
            generator_func,
            args.trials,
            config
        )
        
        all_results[molecule_name] = results
    
    # Save results
    output_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_data = {
        'timestamp': timestamp,
        'config': config,
        'results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
