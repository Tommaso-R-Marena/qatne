# Quick Start Guide

Get started with QATNE in 5 minutes!

## Installation

### Option 1: Google Colab (Easiest)

Click this button to open the main notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/02_algorithm_implementation.ipynb)

No installation needed! Everything runs in your browser.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/qatne.git
cd qatne

# Create environment
conda env create -f environment.yml
conda activate qatne

# Install package
pip install -e .
```

## Your First Calculation

### 1. Import QATNE

```python
from qatne.algorithms import QATNESolver
import numpy as np
```

### 2. Create a Hamiltonian

Let's use a simple 2-qubit system:

```python
# Simple Hamiltonian: H = Z_0 + Z_1
H = np.array([
    [2,  0,  0,  0],
    [0,  0,  0,  0],
    [0,  0,  0,  0],
    [0,  0,  0, -2]
])

num_qubits = 2
```

Exact ground state energy: -2.0 Ha

### 3. Initialize Solver

```python
solver = QATNESolver(
    hamiltonian=H,
    num_qubits=num_qubits,
    max_bond_dim=8,
    convergence_threshold=1e-4,
    shots=4096
)
```

**Parameters:**
- `hamiltonian`: Matrix representation of your Hamiltonian
- `num_qubits`: Number of qubits in the system
- `max_bond_dim`: Maximum tensor network bond dimension
- `convergence_threshold`: Energy convergence criterion
- `shots`: Number of measurement shots per evaluation

### 4. Run Optimization

```python
# Set random seed for reproducibility
np.random.seed(42)

# Solve for ground state
ground_energy, optimal_params = solver.solve(max_iterations=100)

print(f"Ground state energy: {ground_energy:.6f} Ha")
print(f"Number of iterations: {len(solver.energy_history)}")
print(f"Final gradient norm: {solver.gradient_norms[-1]:.6e}")
```

**Expected Output:**
```
Starting QATNE optimization with 12 parameters...
Initial tensor network bond dimension: 4
Iter    0 | Energy:   -0.85432100 | ||∇E||:   0.432100 | Bond dim: 4
Iter   10 | Energy:   -1.45678900 | ||∇E||:   0.123456 | Bond dim: 4
...
Converged after 45 iterations!
Ground state energy: -1.987654 Ha
Number of iterations: 45
Final gradient norm: 9.87e-05
```

### 5. Analyze Results

```python
import matplotlib.pyplot as plt

# Plot energy convergence
plt.figure(figsize=(10, 6))
plt.plot(solver.energy_history, label='QATNE')
plt.axhline(-2.0, color='r', linestyle='--', label='Exact')
plt.xlabel('Iteration')
plt.ylabel('Energy (Ha)')
plt.title('Energy Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Molecular Example: H₂

Now let's try a real molecule:

```python
from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf

# Define H2 molecule
geometry = [('H', (0, 0, 0)), ('H', (0, 0, 0.735))]

molecule = MolecularData(
    geometry=geometry,
    basis='sto-3g',
    multiplicity=1,
    charge=0
)

# Run classical calculation for reference
molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

# Convert to qubit Hamiltonian
hamiltonian_ferm = molecule.get_molecular_hamiltonian()
hamiltonian_jw = jordan_wigner(hamiltonian_ferm)
H_matrix = hamiltonian_jw.to_matrix()

num_qubits = int(np.log2(H_matrix.shape[0]))
exact_energy = molecule.fci_energy

print(f"System: H2 molecule")
print(f"Number of qubits: {num_qubits}")
print(f"FCI energy: {exact_energy:.10f} Ha")
```

**Initialize and solve:**

```python
solver = QATNESolver(
    hamiltonian=H_matrix,
    num_qubits=num_qubits,
    max_bond_dim=16,
    convergence_threshold=1e-6,
    shots=8192
)

ground_energy, optimal_params = solver.solve(max_iterations=200)

error = abs(ground_energy - exact_energy)
relative_error = error / abs(exact_energy) * 100

print(f"\nResults:")
print(f"QATNE energy:  {ground_energy:.10f} Ha")
print(f"FCI energy:    {exact_energy:.10f} Ha")
print(f"Error:         {error:.2e} Ha")
print(f"Relative error: {relative_error:.4f}%")
```

## Understanding the Output

### Energy History

```python
print(f"Initial energy: {solver.energy_history[0]:.6f} Ha")
print(f"Final energy:   {solver.energy_history[-1]:.6f} Ha")
print(f"Improvement:    {solver.energy_history[0] - solver.energy_history[-1]:.6f} Ha")
```

### Gradient Norms

```python
plt.semilogy(solver.gradient_norms)
plt.xlabel('Iteration')
plt.ylabel('||∇E||')
plt.title('Gradient Norm Evolution')
plt.grid(True, alpha=0.3)
plt.show()
```

Small gradient norm → converged to critical point (hopefully minimum!)

### Tensor Network Structure

```python
print(f"Initial bond dimension: 4")
print(f"Final bond dimension:   {solver.tensor_network.bond_dim}")
print(f"Number of layers:       {solver.tensor_network.num_layers}")
```

Adaptive bond dimension increases in regions with high entanglement.

## Advanced Usage

### Custom Initial Parameters

```python
num_params = solver._estimate_num_parameters()
initial_params = np.random.randn(num_params) * 0.01  # Small random initialization

ground_energy, optimal_params = solver.solve(
    initial_params=initial_params,
    max_iterations=200
)
```

### Get State Vector

```python
statevector = solver.get_statevector(optimal_params)
print(f"State vector shape: {statevector.shape}")
print(f"Normalization: {np.linalg.norm(statevector):.10f}")
```

### Compute Fidelity

```python
# Get exact ground state
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
ground_state_exact = eigenvectors[:, 0]

# Compute fidelity with QATNE state
fidelity = solver.compute_fidelity(optimal_params, ground_state_exact)
print(f"Fidelity with exact ground state: {fidelity:.6f}")
```

## Tips for Best Results

### 1. Shot Count

- **Low shots** (1024): Fast but noisy, good for testing
- **Medium shots** (4096-8192): Balanced, recommended for most cases
- **High shots** (16384+): Slow but accurate, for final results

### 2. Bond Dimension

- Start with `max_bond_dim=8` for small systems (≤4 qubits)
- Use `max_bond_dim=16-32` for medium systems (5-8 qubits)
- Increase further for larger or highly entangled systems

### 3. Convergence Threshold

- `1e-3`: Chemical accuracy (~1 kcal/mol)
- `1e-4`: Good accuracy for most applications
- `1e-6`: High precision (requires more iterations)

### 4. Initialization

- Small random: `np.random.randn(n) * 0.1`
- Zero initialization: Can work but may converge slower
- Warm start: Use parameters from similar system

## Troubleshooting

### Issue: Optimization Not Converging

**Solutions:**
- Increase `max_iterations`
- Reduce learning rate (not exposed, but can modify code)
- Increase `shots` for more accurate gradients
- Try different random seed

### Issue: Error Too Large

**Solutions:**
- Increase `max_bond_dim`
- Increase `shots` for better statistics
- Run for more iterations
- Check Hamiltonian is correct

### Issue: Too Slow

**Solutions:**
- Reduce `shots` (trade accuracy for speed)
- Reduce `max_bond_dim`
- Use fewer `max_iterations`
- Run fast tests first, then scale up

## Next Steps

1. **Explore Notebooks**: Check out `notebooks/` for detailed examples
2. **Read Theory**: See `docs/theory/` for mathematical foundations
3. **Run Benchmarks**: Try `notebooks/04_molecular_benchmarks.ipynb`
4. **Deploy to Hardware**: See `docs/tutorials/hardware_deployment.md`
5. **Contribute**: Read `CONTRIBUTING.md` to improve QATNE

## Need Help?

- **Documentation**: https://github.com/Tommaso-R-Marena/qatne/tree/main/docs
- **Issues**: https://github.com/Tommaso-R-Marena/qatne/issues
- **Discussions**: https://github.com/Tommaso-R-Marena/qatne/discussions

Happy quantum computing! 🚀
