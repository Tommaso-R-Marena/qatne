"""Hamiltonian construction and manipulation utilities."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qatne.core.exceptions import QATNEError


class MolecularHamiltonian:
    """Molecular Hamiltonian in qubit representation.

    Supports initialization from dense matrices, SparsePauliOp, or Pauli strings.
    """

    def __init__(
        self,
        data: np.ndarray | SparsePauliOp | str | list[tuple[str, complex]],
    ):
        self._matrix: np.ndarray | None = None

        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise QATNEError("Hamiltonian matrix must be square")
            dim = data.shape[0]
            if dim & (dim - 1) != 0:
                raise QATNEError("Hamiltonian dimension must be a power of 2")
            self._matrix = data
            self.op = SparsePauliOp.from_operator(data)
            self.num_qubits = int(np.log2(dim))
        elif isinstance(data, SparsePauliOp):
            self.op = data
            self.num_qubits = data.num_qubits
        elif isinstance(data, (str, list)):
            try:
                self.op = SparsePauliOp(data)
                self.num_qubits = self.op.num_qubits
            except Exception as e:
                raise QATNEError(f"Failed to initialize SparsePauliOp from data: {e}")
        else:
            raise QATNEError(f"Unsupported Hamiltonian data type: {type(data)}")

        # Handle zero operator or very small coefficients that Qiskit simplifies to empty
        # Qiskit EstimatorV2 raises ValueError: Empty observable was detected if it's purely zero.
        # We ensure it has a term that won't be simplified away.
        simplified_op = self.op.simplify()
        if len(simplified_op.coeffs) == 0 or np.allclose(simplified_op.coeffs, 0):
            # Using a term with a coefficient that is large enough to not be simplified,
            # but small enough to not affect typical chemical accuracy (1e-3 Ha).
            self.op = SparsePauliOp("I" * self.num_qubits, coeffs=[1e-6])
            # Clear matrix if it was set
            self._matrix = None
        else:
            self.op = simplified_op

    @property
    def matrix(self) -> np.ndarray:
        """Qubit operator as a dense matrix (lazily evaluated)."""
        if self._matrix is None:
            self._matrix = self.op.to_matrix()
        return self._matrix

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
