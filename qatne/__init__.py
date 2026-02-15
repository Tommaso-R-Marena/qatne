"""
QATNE: Quantum Adaptive Tensor Network Eigensolver

A hybrid quantum-classical algorithm for molecular ground state estimation
with adaptive tensor network representations.
"""

__version__ = "1.0.0"
__author__ = "Tommaso R. Marena"

from qatne.algorithms.qatne_solver import QATNESolver
from qatne.core.tensor_network import TensorNetwork
from qatne.core.hamiltonian import MolecularHamiltonian

__all__ = [
    "QATNESolver",
    "TensorNetwork",
    "MolecularHamiltonian",
]
