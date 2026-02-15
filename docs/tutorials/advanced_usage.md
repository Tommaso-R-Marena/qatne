# Advanced Usage Guide

This guide covers advanced features of QATNE for experienced users.

## Custom Optimization Strategies

### Adaptive Learning Rates

```python
from qatne.algorithms import QATNESolver
import numpy as np

class AdaptiveLearningRateSolver(QATNESolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 0.1
        self.decay_rate = 0.95
        
    def compute_adaptive_lr(self, iteration, gradient_norm):
        """
        Compute adaptive learning rate based on:
        - Iteration number
        - Gradient magnitude
        - Energy change
        """
        base_lr = self.initial_lr * (self.decay_rate ** (iteration / 50))
        
        # Increase LR if gradient is small (might be stuck)
        if gradient_norm < 1e-4:
            base_lr *= 2.0
        
        # Decrease LR if energy oscillates
        if len(self.energy_history) > 2:
            recent_change = abs(
                self.energy_history[-1] - self.energy_history[-2]
            )
            if recent_change > 0.01:
                base_lr *= 0.5
        
        return np.clip(base_lr, 1e-5, 0.5)
```

### Custom Convergence Criteria

```python
def custom_convergence_check(
    solver,
    window_size=10,
    relative_threshold=1e-6
):
    """
    Check convergence using moving average of energy
    """
    if len(solver.energy_history) < window_size:
        return False
    
    recent_energies = solver.energy_history[-window_size:]
    energy_std = np.std(recent_energies)
    energy_mean = np.mean(recent_energies)
    
    relative_std = energy_std / abs(energy_mean)
    
    return relative_std < relative_threshold
```

## Advanced Tensor Network Manipulation

### Manual Bond Dimension Control

```python
from qatne.core.tensor_network import TensorNetwork

# Create tensor network with custom structure
tn = TensorNetwork(num_sites=8, bond_dim=4)

# Manually increase bond dimension at specific sites
for site in [2, 3, 5]:
    tn.increase_bond_dim(site)
    tn.increase_bond_dim(site)  # Double increase

print(f"Bond dimensions: {tn.bond_dims}")

# Custom entanglement pattern
custom_pairs = [
    [(0, 2), (1, 3), (4, 6), (5, 7)],  # Layer 0: long-range
    [(0, 1), (2, 3), (4, 5), (6, 7)],  # Layer 1: nearest
]
tn.entanglement_pairs = custom_pairs
```

### Entanglement-Guided Adaptation

```python
def adapt_based_on_entanglement(
    solver,
    state_vector,
    threshold=1.0
):
    """
    Adapt tensor network based on entanglement entropy
    """
    num_qubits = solver.num_qubits
    
    for partition in range(1, num_qubits):
        entropy = solver.tensor_network.compute_entanglement_entropy(
            state_vector, partition
        )
        
        # If entropy is high, increase bond dimension
        if entropy > threshold:
            solver.tensor_network.increase_bond_dim(partition - 1)
            print(f"Increased bond dim at site {partition-1} "
                  f"(entropy = {entropy:.4f})")
```

## Parallel Execution

### Multiple Trial Parallelization

```python
from multiprocessing import Pool
import functools

def run_single_trial(trial_id, hamiltonian, num_qubits, config):
    """
    Run single QATNE trial with specific seed
    """
    np.random.seed(trial_id + 1000)
    
    solver = QATNESolver(
        hamiltonian=hamiltonian,
        num_qubits=num_qubits,
        **config
    )
    
    energy, params = solver.solve(
        max_iterations=config['max_iterations']
    )
    
    return {
        'trial_id': trial_id,
        'energy': energy,
        'params': params,
        'history': solver.energy_history
    }

# Run trials in parallel
config = {
    'max_bond_dim': 16,
    'convergence_threshold': 1e-6,
    'shots': 8192,
    'max_iterations': 200
}

run_trial = functools.partial(
    run_single_trial,
    hamiltonian=H_matrix,
    num_qubits=num_qubits,
    config=config
)

with Pool(processes=4) as pool:
    results = pool.map(run_trial, range(20))

# Analyze results
energies = [r['energy'] for r in results]
print(f"Mean: {np.mean(energies):.8f}")
print(f"Std:  {np.std(energies):.2e}")
```

## Hardware Execution

### IBM Quantum Hardware

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qatne.algorithms import QATNESolver

# Setup IBM Quantum
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibm_brisbane")

print(f"Using backend: {backend.name}")
print(f"Number of qubits: {backend.num_qubits}")

# Create solver with hardware backend
solver = QATNESolver(
    hamiltonian=H_matrix,
    num_qubits=4,
    backend=backend,  # Use real hardware
    shots=4096  # Reduced for hardware
)

# Enable error mitigation
solver.enable_error_mitigation = True
solver.mitigation_method = 'measurement'

# Run on hardware
energy, params = solver.solve(max_iterations=100)
```

### Error Mitigation Strategies

```python
from qatne.algorithms.error_mitigation import (
    zero_noise_extrapolation,
    measurement_error_mitigation
)

# Zero-noise extrapolation
noise_factors = [1.0, 1.5, 2.0, 2.5]
energies_at_noise = []

for factor in noise_factors:
    solver.noise_amplification_factor = factor
    energy, _ = solver.solve(max_iterations=50)
    energies_at_noise.append(energy)

# Extrapolate to zero noise
zero_noise_energy = zero_noise_extrapolation(
    noise_factors, energies_at_noise
)
print(f"Zero-noise energy: {zero_noise_energy:.8f} Ha")
```

## Advanced Visualization

### Custom Energy Landscape

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_energy_landscape_2d(
    solver,
    optimal_params,
    param_indices=[0, 1],
    resolution=50
):
    """
    Plot 2D slice of energy landscape
    """
    idx1, idx2 = param_indices
    
    # Create parameter grid
    p1_range = np.linspace(
        optimal_params[idx1] - np.pi,
        optimal_params[idx1] + np.pi,
        resolution
    )
    p2_range = np.linspace(
        optimal_params[idx2] - np.pi,
        optimal_params[idx2] + np.pi,
        resolution
    )
    
    P1, P2 = np.meshgrid(p1_range, p2_range)
    energies = np.zeros_like(P1)
    
    # Compute energies
    for i in range(resolution):
        for j in range(resolution):
            params_test = optimal_params.copy()
            params_test[idx1] = P1[i, j]
            params_test[idx2] = P2[i, j]
            energies[i, j] = solver._compute_energy(params_test)
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(P1, P2, energies, cmap='viridis', alpha=0.8)
    ax1.scatter(
        [optimal_params[idx1]],
        [optimal_params[idx2]],
        [solver.energy_history[-1]],
        color='red', s=100, label='Optimum'
    )
    ax1.set_xlabel(f'θ_{idx1}')
    ax1.set_ylabel(f'θ_{idx2}')
    ax1.set_zlabel('Energy (Ha)')
    ax1.set_title('Energy Landscape')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(P1, P2, energies, levels=20, cmap='viridis')
    ax2.scatter(
        optimal_params[idx1],
        optimal_params[idx2],
        color='red', s=100, marker='*'
    )
    ax2.set_xlabel(f'θ_{idx1}')
    ax2.set_ylabel(f'θ_{idx2}')
    ax2.set_title('Energy Contours')
    plt.colorbar(contour, ax=ax2, label='Energy (Ha)')
    
    plt.tight_layout()
    return fig
```

### Tensor Network Structure Visualization

```python
import networkx as nx

def visualize_tensor_network(tensor_network):
    """
    Visualize tensor network as graph
    """
    G = nx.Graph()
    
    # Add nodes
    for i in range(tensor_network.num_sites):
        G.add_node(i, label=f'q{i}')
    
    # Add edges with bond dimensions
    for (i, j), bond_dim in tensor_network.bond_dims.items():
        G.add_edge(i, j, weight=bond_dim)
    
    # Draw
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color='lightblue',
        node_size=800, ax=ax
    )
    
    # Draw edges with width proportional to bond dimension
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    widths = [w / max(weights) * 5 for w in weights]
    
    nx.draw_networkx_edges(
        G, pos, width=widths, alpha=0.6, ax=ax
    )
    
    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Edge labels (bond dimensions)
    edge_labels = {
        (i, j): f'χ={bond_dim}'
        for (i, j), bond_dim in tensor_network.bond_dims.items()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
    
    ax.set_title('Tensor Network Structure', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    return fig
```

## Performance Optimization

### Circuit Compilation

```python
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGatesDecomposition,
    CXCancellation,
    CommutativeCancellation
)

def optimize_circuit(circuit, backend):
    """
    Optimize quantum circuit for specific backend
    """
    # Create pass manager
    pm = PassManager([
        Optimize1qGatesDecomposition(basis=['rx', 'ry', 'rz']),
        CXCancellation(),
        CommutativeCancellation()
    ])
    
    # Optimize
    optimized_circuit = pm.run(circuit)
    
    print(f"Original depth: {circuit.depth()}")
    print(f"Optimized depth: {optimized_circuit.depth()}")
    print(f"Reduction: {(1 - optimized_circuit.depth() / circuit.depth()) * 100:.1f}%")
    
    return optimized_circuit
```

### Gradient Computation Optimization

```python
def compute_gradient_batched(
    solver,
    params,
    batch_size=4
):
    """
    Compute gradient in batches for parallel execution
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def compute_param_gradient(i):
        shift = np.pi / 2
        params_plus = params.copy()
        params_plus[i] += shift
        params_minus = params.copy()
        params_minus[i] -= shift
        
        return (
            solver._compute_energy(params_plus) -
            solver._compute_energy(params_minus)
        ) / 2.0
    
    gradient = np.zeros_like(params)
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        gradients = list(executor.map(
            compute_param_gradient,
            range(len(params))
        ))
    
    return np.array(gradients)
```

## Best Practices

### 1. Start Small
- Begin with small molecules (H2, LiH)
- Use low bond dimensions initially
- Gradually increase complexity

### 2. Monitor Convergence
- Track multiple metrics (energy, gradient, parameters)
- Use early stopping to save computational resources
- Save checkpoints regularly

### 3. Statistical Validation
- Always run multiple trials
- Report confidence intervals
- Test against known benchmarks

### 4. Hardware Considerations
- Use simulators for algorithm development
- Reserve hardware time for final validation
- Apply appropriate error mitigation

### 5. Documentation
- Document all hyperparameters
- Save experimental configurations
- Version control your analysis notebooks

## Troubleshooting

### Convergence Issues

**Problem:** Optimization gets stuck in local minimum

**Solutions:**
- Reduce learning rate
- Increase tensor network bond dimension
- Try different random seeds
- Use multiple random initializations

### Memory Issues

**Problem:** Out of memory errors

**Solutions:**
- Reduce number of shots
- Decrease maximum bond dimension
- Use sparse matrix representations
- Process results in batches

### Slow Execution

**Problem:** Optimization takes too long

**Solutions:**
- Reduce number of shots for initial iterations
- Use gradient checkpointing
- Parallelize independent operations
- Profile code to identify bottlenecks

## Next Steps

- Explore [Custom Hamiltonians](custom_hamiltonians.md)
- Review [API Reference](../api/api_reference.md)
- Check [Mathematical Proofs](../theory/proofs.md)
