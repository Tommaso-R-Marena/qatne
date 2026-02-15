# API Reference

Complete API documentation for QATNE.

## Core Modules

### `qatne.algorithms.QATNESolver`

Main solver class for quantum adaptive tensor network eigensolver.

#### Class Definition

```python
class QATNESolver:
    def __init__(
        self,
        hamiltonian: np.ndarray,
        num_qubits: int,
        max_bond_dim: int = 32,
        convergence_threshold: float = 1e-6,
        shots: int = 8192
    )
```

**Parameters:**

- **hamiltonian** (*np.ndarray*): Molecular Hamiltonian in matrix form, shape (2^n, 2^n)
- **num_qubits** (*int*): Number of qubits required for simulation
- **max_bond_dim** (*int*, optional): Maximum tensor network bond dimension. Default: 32
- **convergence_threshold** (*float*, optional): Energy convergence criterion in Hartree. Default: 1e-6
- **shots** (*int*, optional): Number of measurement shots per circuit evaluation. Default: 8192

**Attributes:**

- **energy_history** (*List[float]*): History of energy values during optimization
- **parameter_history** (*List[np.ndarray]*): History of parameter vectors
- **gradient_norms** (*List[float]*): History of gradient magnitudes
- **tensor_network** (*TensorNetwork*): Adaptive tensor network structure

#### Methods

##### `solve()`

```python
def solve(
    self,
    initial_params: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    optimizer: str = 'adam'
) -> Tuple[float, np.ndarray]
```

Solve for ground state energy using QATNE algorithm.

**Parameters:**

- **initial_params** (*np.ndarray*, optional): Initial circuit parameters. If None, random initialization is used
- **max_iterations** (*int*, optional): Maximum number of optimization iterations. Default: 1000
- **optimizer** (*str*, optional): Optimization method ('adam', 'lbfgs', 'sgd'). Default: 'adam'

**Returns:**

- **ground_energy** (*float*): Estimated ground state energy in Hartree
- **optimal_params** (*np.ndarray*): Optimal circuit parameters

**Example:**

```python
solver = QATNESolver(hamiltonian=H, num_qubits=4)
energy, params = solver.solve(max_iterations=200)
print(f"Ground energy: {energy:.10f} Ha")
```

---

##### `get_statevector()`

```python
def get_statevector(
    self,
    params: np.ndarray
) -> np.ndarray
```

Get quantum state vector for given parameters.

**Parameters:**

- **params** (*np.ndarray*): Circuit parameters, shape (num_params,)

**Returns:**

- **statevector** (*np.ndarray*): Complex state vector, shape (2^num_qubits,)

**Example:**

```python
state = solver.get_statevector(optimal_params)
print(f"State norm: {np.linalg.norm(state):.10f}")
```

---

##### `compute_fidelity()`

```python
def compute_fidelity(
    self,
    params: np.ndarray,
    target_state: np.ndarray
) -> float
```

Compute fidelity between parameterized state and target state.

**Parameters:**

- **params** (*np.ndarray*): Circuit parameters
- **target_state** (*np.ndarray*): Target state vector, normalized

**Returns:**

- **fidelity** (*float*): Fidelity in range [0, 1]

**Example:**

```python
# Compare with exact ground state
eigenvalues, eigenvectors = np.linalg.eigh(H)
ground_state = eigenvectors[:, 0]
fidelity = solver.compute_fidelity(optimal_params, ground_state)
print(f"Fidelity: {fidelity:.6f}")
```

---

### `qatne.core.TensorNetwork`

Adaptive Matrix Product State tensor network.

#### Class Definition

```python
class TensorNetwork:
    def __init__(
        self,
        num_sites: int,
        bond_dim: int = 4,
        max_bond_dim: int = 32
    )
```

**Parameters:**

- **num_sites** (*int*): Number of tensor network sites (equals num_qubits)
- **bond_dim** (*int*, optional): Initial bond dimension. Default: 4
- **max_bond_dim** (*int*, optional): Maximum allowed bond dimension. Default: 32

**Attributes:**

- **num_layers** (*int*): Number of entangling layers in circuit
- **bond_dims** (*Dict*): Bond dimensions for each connection

#### Methods

##### `get_entanglement_pairs()`

```python
def get_entanglement_pairs(
    self,
    layer: int
) -> List[Tuple[int, int]]
```

Get qubit pairs for entangling gates in specified layer.

**Parameters:**

- **layer** (*int*): Layer index

**Returns:**

- **pairs** (*List[Tuple[int, int]]*): List of qubit index pairs

---

##### `increase_bond_dim()`

```python
def increase_bond_dim(
    self,
    site: int
) -> None
```

Increase bond dimension at specified site.

**Parameters:**

- **site** (*int*): Site index where bond dimension should increase

---

##### `compute_entanglement_entropy()`

```python
def compute_entanglement_entropy(
    self,
    state_vector: np.ndarray,
    partition: int
) -> float
```

Compute von Neumann entanglement entropy across bipartition.

**Parameters:**

- **state_vector** (*np.ndarray*): Quantum state vector
- **partition** (*int*): Partition size (number of qubits in subsystem A)

**Returns:**

- **entropy** (*float*): Entanglement entropy in bits

**Example:**

```python
tn = TensorNetwork(num_sites=4)
state = solver.get_statevector(params)
entropy = tn.compute_entanglement_entropy(state, partition=2)
print(f"Entropy: {entropy:.6f} bits")
```

---

## Utility Functions

### `qatne.benchmarks.create_h2_hamiltonian()`

```python
def create_h2_hamiltonian(
    bond_length: float = 0.735
) -> Tuple[np.ndarray, int, float]
```

Create H₂ molecule Hamiltonian for benchmarking.

**Parameters:**

- **bond_length** (*float*, optional): H-H distance in Angstroms. Default: 0.735 (equilibrium)

**Returns:**

- **hamiltonian_matrix** (*np.ndarray*): Qubit Hamiltonian matrix
- **num_qubits** (*int*): Number of qubits required
- **fci_energy** (*float*): Exact FCI ground state energy

**Example:**

```python
from qatne.benchmarks import create_h2_hamiltonian

H, n_qubits, exact_e = create_h2_hamiltonian(bond_length=0.735)
print(f"H2 Hamiltonian: {n_qubits} qubits, exact energy: {exact_e:.10f} Ha")
```

---

## Constants

### Physical Constants

```python
from qatne.utils import constants

HARTREE_TO_EV = 27.211386245988  # Hartree to eV conversion
HARTREE_TO_KCAL = 627.5094740631  # Hartree to kcal/mol
BOHR_TO_ANGSTROM = 0.529177210903  # Bohr to Angstrom
```

### Default Parameters

```python
DEFAULT_MAX_BOND_DIM = 32
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_SHOTS = 8192
DEFAULT_MAX_ITERATIONS = 1000
```

---

## Exceptions

### `ConvergenceError`

Raised when optimization fails to converge.

```python
from qatne.algorithms import ConvergenceError

try:
    energy, params = solver.solve(max_iterations=10)
except ConvergenceError as e:
    print(f"Failed to converge: {e}")
```

### `InvalidHamiltonianError`

Raised when Hamiltonian has invalid format.

```python
from qatne.core import InvalidHamiltonianError

try:
    solver = QATNESolver(hamiltonian=bad_H, num_qubits=4)
except InvalidHamiltonianError as e:
    print(f"Invalid Hamiltonian: {e}")
```

---

## Type Definitions

### Custom Types

```python
from typing import List, Tuple, Optional, Union
import numpy as np

# Parameter vector
ParameterVector = np.ndarray  # shape: (num_params,)

# State vector
StateVector = np.ndarray  # shape: (2^num_qubits,), dtype: complex

# Hamiltonian matrix
HamiltonianMatrix = np.ndarray  # shape: (2^n, 2^n), dtype: complex

# Energy value
Energy = float  # in Hartree

# Optimization history
OptimizationHistory = List[Energy]
```

---

## Configuration

### Logging

```python
import logging
from qatne import set_log_level

# Set logging level
set_log_level(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR

# Custom logger
logger = logging.getLogger('qatne')
logger.info("Starting optimization...")
```

### Random Seed

```python
import numpy as np
from qatne.utils import set_global_seed

# Set global random seed for reproducibility
set_global_seed(42)

# Or manually
np.random.seed(42)
```

---

For more examples, see the [tutorials](../tutorials/quickstart.md) and [notebooks](../../notebooks/).
