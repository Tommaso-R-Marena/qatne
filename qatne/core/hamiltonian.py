"""Hamiltonian construction and manipulation utilities."""

from __future__ import annotations

import numpy as np

from qatne.core.exceptions import QATNEError


class MolecularHamiltonian:
    """Molecular Hamiltonian in qubit representation."""

    def __init__(self, matrix: np.ndarray):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise QATNEError("Hamiltonian must be a square matrix")
        dim = matrix.shape[0]
        if dim & (dim - 1) != 0:
            raise QATNEError("Hamiltonian dimension must be a power of 2")

        self.matrix = matrix
        self.num_qubits = int(np.log2(dim))

    def get_ground_energy(self) -> float:
        """Compute exact ground-state energy via diagonalization."""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return float(eigenvalues[0])

    def get_ground_state(self) -> np.ndarray:
        """Compute exact ground-state vector via diagonalization."""
        _, eigenvectors = np.linalg.eigh(self.matrix)
        return eigenvectors[:, 0]

    def get_spectrum(self) -> np.ndarray:
        """Compute full sorted energy spectrum."""
        return np.linalg.eigvalsh(self.matrix)

    def get_spectral_gap(self) -> float:
        """Compute gap between ground and first excited state."""
        spectrum = self.get_spectrum()
        if spectrum.size < 2:
            raise QATNEError("Spectral gap is undefined for 1D spectrum")
        return float(spectrum[1] - spectrum[0])

    def compute_expectation(self, state: np.ndarray) -> float:
        """Compute expectation value for a state vector."""
        return float(np.real(state.conj().T @ self.matrix @ state))

    def get_norm(self) -> float:
        """Compute operator norm of Hamiltonian."""
        return float(np.linalg.norm(self.matrix, ord=2))
