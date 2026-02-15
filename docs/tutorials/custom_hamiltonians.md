# Custom Hamiltonians Guide

This guide shows how to create and use custom Hamiltonians with QATNE.

## Overview

QATNE can work with any Hermitian operator represented as a matrix. This guide covers:

1. Creating custom molecular Hamiltonians
2. Using predefined molecular systems
3. Implementing problem-specific Hamiltonians
4. Hamiltonian transformations and mappings

## Molecular Hamiltonians

### Using OpenFermion

```python
from openfermion import MolecularData, jordan_wigner, bravyi_kitaev
from openfermionpyscf import run_pyscf
import numpy as np

def create_molecular_hamiltonian(
    atoms,
    basis='sto-3g',
    multiplicity=1,
    charge=0,
    mapping='jordan_wigner'
):
    """
    Create molecular Hamiltonian from atomic geometry
    
    Args:
        atoms: List of (element, coordinates) tuples
        basis: Basis set (sto-3g, 6-31g, etc.)
        multiplicity: Spin multiplicity
        charge: Total charge
        mapping: Fermion-to-qubit mapping (jordan_wigner or bravyi_kitaev)
    
    Returns:
        hamiltonian_matrix: Matrix representation
        num_qubits: Number of qubits needed
        fci_energy: Exact FCI energy
    """
    # Create molecule
    molecule = MolecularData(
        geometry=atoms,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge
    )
    
    # Run classical calculation
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    
    # Get fermionic Hamiltonian
    hamiltonian_ferm = molecule.get_molecular_hamiltonian()
    
    # Convert to qubit operator
    if mapping == 'jordan_wigner':
        hamiltonian_qubit = jordan_wigner(hamiltonian_ferm)
    elif mapping == 'bravyi_kitaev':
        hamiltonian_qubit = bravyi_kitaev(hamiltonian_ferm)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")
    
    # Convert to matrix
    hamiltonian_matrix = hamiltonian_qubit.to_matrix()
    num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
    
    return hamiltonian_matrix, num_qubits, molecule.fci_energy

# Example: H2O molecule
water_geometry = [
    ('O', (0.0, 0.0, 0.0)),
    ('H', (0.757, 0.586, 0.0)),
    ('H', (-0.757, 0.586, 0.0))
]

H_water, n_qubits, E_exact = create_molecular_hamiltonian(
    water_geometry,
    basis='sto-3g'
)

print(f"Water molecule:")
print(f"  Number of qubits: {n_qubits}")
print(f"  FCI energy: {E_exact:.8f} Ha")
print(f"  Hamiltonian shape: {H_water.shape}")
```

### Predefined Molecular Systems

```python
from qatne.benchmarks.molecular_systems import (
    h2_molecule,
    lih_molecule,
    h2o_molecule,
    beh2_molecule
)

# H2 at various bond lengths
for bond_length in [0.5, 0.735, 1.0, 1.5]:
    H, n_q, E_fci = h2_molecule(bond_length=bond_length)
    print(f"H2 at {bond_length} Å: {E_fci:.8f} Ha")

# LiH molecule
H_lih, n_q_lih, E_fci_lih = lih_molecule(bond_length=1.6)
print(f"\nLiH: {n_q_lih} qubits, E = {E_fci_lih:.8f} Ha")

# Water molecule
H_water, n_q_water, E_fci_water = h2o_molecule()
print(f"H2O: {n_q_water} qubits, E = {E_fci_water:.8f} Ha")
```

## Custom Problem Hamiltonians

### Max-Cut Problem

```python
import networkx as nx

def maxcut_hamiltonian(graph):
    """
    Create Hamiltonian for Max-Cut problem
    
    H = -0.5 * Σ_{(i,j) ∈ E} (1 - Z_i Z_j)
    """
    num_nodes = len(graph.nodes())
    num_qubits = num_nodes
    
    # Initialize Hamiltonian
    H = np.zeros((2**num_qubits, 2**num_qubits))
    
    # Pauli Z operator
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    
    # Add terms for each edge
    for i, j in graph.edges():
        # Construct Z_i ⊗ Z_j operator
        operator = np.array([[1.0]])
        for qubit in range(num_qubits):
            if qubit == i or qubit == j:
                operator = np.kron(operator, Z)
            else:
                operator = np.kron(operator, I)
        
        # Add to Hamiltonian
        H -= 0.5 * (np.eye(2**num_qubits) - operator)
    
    return H, num_qubits

# Example: Complete graph K4
G = nx.complete_graph(4)
H_maxcut, n_q = maxcut_hamiltonian(G)

print(f"Max-Cut on K4:")
print(f"  Number of qubits: {n_q}")
print(f"  Hamiltonian eigenvalues: {np.linalg.eigvalsh(H_maxcut)[:5]}")
```

### Ising Model

```python
def ising_hamiltonian(num_spins, J_coupling, h_field, boundary='open'):
    """
    Create 1D Ising model Hamiltonian
    
    H = -J Σ_i Z_i Z_{i+1} - h Σ_i Z_i
    
    Args:
        num_spins: Number of spins
        J_coupling: Coupling strength
        h_field: External field
        boundary: 'open' or 'periodic'
    """
    num_qubits = num_spins
    H = np.zeros((2**num_qubits, 2**num_qubits))
    
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    
    # Coupling terms: -J Z_i Z_{i+1}
    pairs = list(range(num_spins - 1))
    if boundary == 'periodic':
        pairs.append((num_spins - 1, 0))
    
    for i in pairs:
        operator = np.array([[1.0]])
        for q in range(num_qubits):
            if q == i or q == (i + 1) % num_spins:
                operator = np.kron(operator, Z)
            else:
                operator = np.kron(operator, I)
        H -= J_coupling * operator
    
    # Field terms: -h Z_i
    for i in range(num_spins):
        operator = np.array([[1.0]])
        for q in range(num_qubits):
            if q == i:
                operator = np.kron(operator, Z)
            else:
                operator = np.kron(operator, I)
        H -= h_field * operator
    
    return H, num_qubits

# Example: 6-spin chain
H_ising, n_q = ising_hamiltonian(
    num_spins=6,
    J_coupling=1.0,
    h_field=0.5,
    boundary='open'
)

print(f"Ising model:")
print(f"  Number of qubits: {n_q}")
print(f"  Ground state energy: {np.min(np.linalg.eigvalsh(H_ising)):.6f}")
```

### Travelling Salesman Problem (TSP)

```python
def tsp_hamiltonian(distance_matrix, penalty_weight=10.0):
    """
    Create QUBO Hamiltonian for TSP
    
    Args:
        distance_matrix: N×N matrix of distances between cities
        penalty_weight: Weight for constraint penalties
    """
    n_cities = len(distance_matrix)
    n_qubits = n_cities ** 2  # Binary variable x_{i,t}
    
    # Initialize
    H = np.zeros((2**n_qubits, 2**n_qubits))
    
    # Helper function to get qubit index
    def qubit_index(city, time):
        return city * n_cities + time
    
    # Objective: minimize total distance
    # (implementation details omitted for brevity)
    
    # Constraint 1: Each city visited exactly once
    # Constraint 2: Each time step has exactly one city
    # (penalty terms added to H)
    
    return H, n_qubits
```

## Hamiltonian Transformations

### Active Space Reduction

```python
def reduce_active_space(
    molecule,
    n_active_electrons,
    n_active_orbitals
):
    """
    Reduce molecular Hamiltonian to active space
    
    This reduces the number of qubits by freezing core orbitals
    and removing virtual orbitals.
    """
    from openfermion import FermionOperator
    
    # Get full Hamiltonian
    hamiltonian_full = molecule.get_molecular_hamiltonian()
    
    # Define active space
    occupied_indices = list(range(n_active_electrons // 2))
    active_indices = list(range(
        n_active_electrons // 2,
        n_active_electrons // 2 + n_active_orbitals
    ))
    
    # Project to active space
    # (implementation uses OpenFermion's projection utilities)
    
    return hamiltonian_active

# Example: Reduce H2O to 4 electrons in 4 orbitals
molecule = MolecularData(
    geometry=water_geometry,
    basis='sto-3g',
    multiplicity=1
)
molecule = run_pyscf(molecule)

H_active = reduce_active_space(
    molecule,
    n_active_electrons=4,
    n_active_orbitals=4
)
```

### Symmetry Reduction

```python
def apply_symmetry_reduction(hamiltonian, symmetry_sector):
    """
    Reduce Hamiltonian by exploiting symmetries
    
    Example: Particle number conservation, spin symmetry
    """
    # Find basis states in symmetry sector
    n_qubits = int(np.log2(hamiltonian.shape[0]))
    
    # Example: Fix particle number
    target_particles = symmetry_sector['n_particles']
    
    valid_states = []
    for i in range(2**n_qubits):
        bitstring = format(i, f'0{n_qubits}b')
        if bitstring.count('1') == target_particles:
            valid_states.append(i)
    
    # Project Hamiltonian to reduced subspace
    H_reduced = hamiltonian[np.ix_(valid_states, valid_states)]
    
    return H_reduced, valid_states

# Example
H_reduced, basis = apply_symmetry_reduction(
    H_water,
    symmetry_sector={'n_particles': 5}
)

print(f"Original size: {H_water.shape}")
print(f"Reduced size: {H_reduced.shape}")
print(f"Reduction factor: {H_water.shape[0] / H_reduced.shape[0]:.1f}x")
```

## Hamiltonian Analysis Tools

### Spectrum Analysis

```python
def analyze_hamiltonian_spectrum(hamiltonian, n_states=10):
    """
    Analyze eigenvalue spectrum of Hamiltonian
    """
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    
    print(f"Spectrum Analysis:")
    print(f"  Ground state energy: {eigenvalues[0]:.8f}")
    print(f"  First excited state: {eigenvalues[1]:.8f}")
    print(f"  Gap: {eigenvalues[1] - eigenvalues[0]:.8f}")
    print(f"  Spectral range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
    
    # Plot spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Low-lying states
    axes[0].bar(range(n_states), eigenvalues[:n_states])
    axes[0].set_xlabel('State index')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Low-lying States')
    axes[0].grid(True, alpha=0.3)
    
    # Full spectrum histogram
    axes[1].hist(eigenvalues, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(eigenvalues[0], color='r', linestyle='--', label='Ground state')
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Energy Spectrum Histogram')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

fig = analyze_hamiltonian_spectrum(H_water)
```

### Pauli Decomposition

```python
def pauli_decomposition(hamiltonian):
    """
    Decompose Hamiltonian into Pauli strings
    
    H = Σ_i c_i P_i
    where P_i are Pauli strings
    """
    from openfermion import QubitOperator
    from openfermion.linalg import get_sparse_operator
    
    n_qubits = int(np.log2(hamiltonian.shape[0]))
    
    # Enumerate all Pauli strings
    pauli_terms = []
    pauli_labels = ['I', 'X', 'Y', 'Z']
    
    # (Implementation details for full decomposition)
    
    # Return as QubitOperator
    hamiltonian_pauli = QubitOperator()
    
    return hamiltonian_pauli
```

## Using Custom Hamiltonians with QATNE

### Basic Usage

```python
from qatne.algorithms import QATNESolver

# Create custom Hamiltonian (any of the above methods)
H_custom, n_qubits = ising_hamiltonian(
    num_spins=6,
    J_coupling=1.0,
    h_field=0.5
)

# Solve with QATNE
solver = QATNESolver(
    hamiltonian=H_custom,
    num_qubits=n_qubits,
    max_bond_dim=16,
    shots=8192
)

energy, params = solver.solve(max_iterations=200)

# Compare to exact
exact_energy = np.min(np.linalg.eigvalsh(H_custom))
print(f"QATNE energy: {energy:.8f}")
print(f"Exact energy: {exact_energy:.8f}")
print(f"Error: {abs(energy - exact_energy):.2e}")
```

### Batch Processing Multiple Hamiltonians

```python
def benchmark_hamiltonian_family(hamiltonian_generator, param_range):
    """
    Benchmark QATNE on family of related Hamiltonians
    """
    results = []
    
    for param in param_range:
        H, n_q = hamiltonian_generator(param)
        
        solver = QATNESolver(
            hamiltonian=H,
            num_qubits=n_q,
            max_bond_dim=16
        )
        
        energy, _ = solver.solve(max_iterations=100)
        exact = np.min(np.linalg.eigvalsh(H))
        
        results.append({
            'parameter': param,
            'qatne_energy': energy,
            'exact_energy': exact,
            'error': abs(energy - exact)
        })
    
    return results

# Example: H2 dissociation curve
bond_lengths = np.linspace(0.4, 3.0, 20)
results = benchmark_hamiltonian_family(
    lambda d: h2_molecule(bond_length=d)[:2],
    bond_lengths
)

# Plot dissociation curve
energies_qatne = [r['qatne_energy'] for r in results]
energies_exact = [r['exact_energy'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(bond_lengths, energies_exact, 'b-', label='Exact', linewidth=2)
plt.plot(bond_lengths, energies_qatne, 'r--', label='QATNE', linewidth=2)
plt.xlabel('Bond length (Å)')
plt.ylabel('Energy (Ha)')
plt.title('H₂ Dissociation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices

1. **Validate Hamiltonian Properties**
   - Check Hermiticity: `np.allclose(H, H.conj().T)`
   - Verify sparsity for large systems
   - Confirm expected symmetries

2. **Start with Known Systems**
   - Test on molecules with known FCI energies
   - Compare to classical methods first
   - Verify convergence behavior

3. **Scale Appropriately**
   - Start with small active spaces
   - Use symmetry reduction when possible
   - Consider problem-specific encodings

4. **Document Hamiltonians**
   - Save construction parameters
   - Record transformations applied
   - Store classical benchmark results

## Next Steps

- Review [Advanced Usage](advanced_usage.md) for optimization strategies
- Check [API Reference](../api/api_reference.md) for detailed documentation
- Explore [Benchmark Systems](../../qatne/benchmarks/molecular_systems.py)
