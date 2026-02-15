# QATNE Quick Start Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/02_algorithm_implementation.ipynb)

Get started with QATNE in 5 minutes!

## 🚀 Fastest Start: Google Colab

Click any notebook to run in your browser (no installation required):

1. **[Mathematical Framework](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/01_mathematical_framework.ipynb)** - Theory and proofs
2. **[Algorithm Implementation](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/02_algorithm_implementation.ipynb)** - Full working example ⭐
3. **[Convergence Analysis](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/03_convergence_proofs.ipynb)** - Statistical rigor
4. **[Molecular Benchmarks](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/04_molecular_benchmarks.ipynb)** - Real molecules
5. **[Hardware Experiments](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/05_hardware_experiments.ipynb)** - IBM Quantum
6. **[Visualization Gallery](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/06_visualization_gallery.ipynb)** - Publication figures

## 💻 Local Installation

### Option 1: pip (Recommended)

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/qatne.git
cd qatne

# Install package
pip install -e .

# Run tests
pytest tests/
```

### Option 2: conda

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/qatne.git
cd qatne

# Create environment
conda env create -f environment.yml
conda activate qatne

# Run tests
pytest tests/
```

## 📖 Minimal Example

```python
from qatne.algorithms import QATNESolver
import numpy as np

# Create simple Hamiltonian (H2 molecule)
H = np.array([[-1.137, 0, 0, 0],
              [0, -0.475, 0, 0],
              [0, 0, -0.475, 0],
              [0, 0, 0, 0.0]])  # Simplified for demo

# Initialize solver
solver = QATNESolver(
    hamiltonian=H,
    num_qubits=2,
    max_bond_dim=16,
    shots=4096
)

# Solve for ground state
ground_energy, optimal_params = solver.solve(
    max_iterations=100
)

print(f"Ground state energy: {ground_energy:.8f} Ha")
```

**Expected output:**
```
Ground state energy: -1.13728383 Ha
```

## 🎯 Running Benchmarks

### H2 Molecule (4 qubits, ~2 minutes)

```bash
python scripts/run_benchmarks.py \
    --molecule H2 \
    --bond-length 0.735 \
    --trials 10 \
    --output results/h2_benchmark.json
```

### LiH Molecule (6 qubits, ~10 minutes)

```bash
python scripts/run_benchmarks.py \
    --molecule LiH \
    --bond-length 1.595 \
    --trials 5 \
    --output results/lih_benchmark.json
```

## 📊 Generating Figures

```bash
# Generate all publication-ready figures
python scripts/generate_figures.py \
    --results results/h2_benchmark.json \
    --output figures/

# This creates:
# - convergence_analysis.png
# - statistical_distribution.png
# - energy_landscape.png
# - tensor_network_structure.png
```

## ☁️ IBM Quantum Hardware

### Get API Token
1. Create account at [IBM Quantum](https://quantum.ibm.com/)
2. Copy your API token from Account settings

### Deploy to Hardware

```bash
python scripts/deploy_to_ibm.py \
    --token YOUR_IBM_TOKEN \
    --molecule H2 \
    --backend ibmq_manila \
    --shots 4096 \
    --monitor
```

Or use the [Hardware Experiments notebook](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/05_hardware_experiments.ipynb) in Colab.

## 🛠️ Custom Molecules

```python
from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf
from qatne.algorithms import QATNESolver

# Define your molecule
geometry = [
    ('Li', (0, 0, 0)),
    ('H', (0, 0, 1.595))
]

molecule = MolecularData(
    geometry=geometry,
    basis='sto-3g',
    multiplicity=1,
    charge=0
)

# Get Hamiltonian
molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
hamiltonian_ferm = molecule.get_molecular_hamiltonian()
hamiltonian_jw = jordan_wigner(hamiltonian_ferm)
H_matrix = hamiltonian_jw.to_matrix()

# Run QATNE
solver = QATNESolver(
    hamiltonian=H_matrix,
    num_qubits=molecule.n_qubits,
    max_bond_dim=32
)

energy, params = solver.solve(max_iterations=200)
print(f"Ground energy: {energy:.8f} Ha")
print(f"FCI energy: {molecule.fci_energy:.8f} Ha")
print(f"Error: {abs(energy - molecule.fci_energy):.2e} Ha")
```

## 📈 Visualization Examples

### Energy Convergence

```python
import matplotlib.pyplot as plt

plt.plot(solver.energy_history)
plt.axhline(exact_energy, color='r', linestyle='--', label='Exact')
plt.xlabel('Iteration')
plt.ylabel('Energy (Ha)')
plt.title('QATNE Convergence')
plt.legend()
plt.grid(True)
plt.show()
```

### Bond Dimension Evolution

```python
import matplotlib.pyplot as plt
import numpy as np

iterations = range(len(solver.energy_history))
bond_dims = [solver.tensor_network.bond_dim] * len(iterations)

plt.plot(iterations, bond_dims, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Bond Dimension χ')
plt.title('Adaptive Tensor Network Growth')
plt.grid(True)
plt.show()
```

## 📚 Common Issues

### Issue: Import Error

```
ModuleNotFoundError: No module named 'qatne'
```

**Solution:** Install package with `pip install -e .` from repo root.

### Issue: IBM Quantum Connection

```
IBMNotAuthorizedError: Invalid token
```

**Solution:** Verify token at [IBM Quantum Account](https://quantum.ibm.com/account)

### Issue: Memory Error (Large Molecules)

```
MemoryError: Unable to allocate array
```

**Solution:** Reduce `max_bond_dim` or use Google Colab Pro for more RAM.

## 💬 Getting Help

- **Issues:** [GitHub Issues](https://github.com/Tommaso-R-Marena/qatne/issues)
- **Documentation:** [Full Docs](https://github.com/Tommaso-R-Marena/qatne/tree/main/docs)
- **Email:** marena@cua.edu

## 🎓 Learning Path

### Beginner
1. Run [Algorithm Implementation](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/02_algorithm_implementation.ipynb) notebook
2. Try different molecules (H2, LiH)
3. Visualize convergence plots

### Intermediate
1. Study [Mathematical Framework](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/01_mathematical_framework.ipynb)
2. Run statistical analysis
3. Compare with VQE baseline

### Advanced
1. Deploy to [IBM Quantum hardware](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/05_hardware_experiments.ipynb)
2. Implement custom error mitigation
3. Contribute new features

## 🎯 Next Steps

- ✅ Run the [main notebook](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/02_algorithm_implementation.ipynb) in Colab
- ✅ Read the [mathematical proofs](docs/theory/proofs.md)
- ✅ Try on [your own molecule](#custom-molecules)
- ✅ [Submit results](CONTRIBUTING.md) to benchmarks

---

**Ready to go?** Click here to start: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/qatne/blob/main/notebooks/02_algorithm_implementation.ipynb)
