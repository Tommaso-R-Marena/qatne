"""
Hamiltonian construction and manipulation utilities.
"""

import numpy as np
from typing import List, Tuple


class MolecularHamiltonian:
    """
    Molecular Hamiltonian in qubit representation.
    
    Provides utilities for Hamiltonian decomposition and analysis.
    """
    
    def __init__(self, matrix: np.ndarray):
        """
        Initialize Hamiltonian
        
        Args:
            matrix: Hamiltonian matrix in computational basis
        """
        self.matrix = matrix
        self.num_qubits = int(np.log2(matrix.shape[0]))
        
    def get_ground_energy(self) -> float:
        """
        Compute exact ground state energy via diagonalization
        
        Returns:
            Ground state energy
        """
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return eigenvalues[0]
    
    def get_ground_state(self) -> np.ndarray:
        """
        Compute exact ground state vector via diagonalization
        
        Returns:
            Ground state vector
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        return eigenvectors[:, 0]
    
    def get_spectrum(self) -> np.ndarray:
        """
        Compute full energy spectrum
        
        Returns:
            Array of eigenvalues (sorted)
        """
        return np.linalg.eigvalsh(self.matrix)
    
    def get_spectral_gap(self) -> float:
        """
        Compute gap between ground and first excited state
        
        Returns:
            Energy gap
        """
        spectrum = self.get_spectrum()
        return spectrum[1] - spectrum[0]
    
    def compute_expectation(self, state: np.ndarray) -> float:
        """
        Compute expectation value for given state
        
        Args:
            state: Quantum state vector
        
        Returns:
            <state|H|state>
        """
        return np.real(state.conj().T @ self.matrix @ state)
    
    def get_norm(self) -> float:
        """
        Compute operator norm of Hamiltonian
        
        Returns:
            ||H||_op
        """
        return np.linalg.norm(self.matrix, ord=2)
