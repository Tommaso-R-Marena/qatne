"""Quantum circuit construction utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qatne.core.exceptions import QuantumCircuitError


class BaseAnsatz(ABC):
    """Abstract interface for ansatz builders."""

    @abstractmethod
    def build_circuit(
        self,
        params: np.ndarray,
        num_layers: int,
        entanglement_pairs_by_layer: list[list[tuple[int, int]]],
    ) -> QuantumCircuit:
        """Build and return a parameterized quantum circuit."""


class AdaptiveAnsatz(BaseAnsatz):
    """Adaptive quantum circuit ansatz.

    Dynamically constructs circuits based on tensor-network connectivity.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    """

    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise QuantumCircuitError("num_qubits must be >= 1")
        self.num_qubits = num_qubits

    def build_circuit(
        self,
        params: np.ndarray,
        num_layers: int,
        entanglement_pairs_by_layer: list[list[tuple[int, int]]],
    ) -> QuantumCircuit:
        """Build quantum circuit with specified entanglement structure.

        Parameters
        ----------
        params : np.ndarray
            Circuit parameters.
        num_layers : int
            Number of entangling layers.
        entanglement_pairs_by_layer : list[list[tuple[int, int]]]
            List of qubit pairs to entangle for each layer.

        Returns
        -------
        QuantumCircuit
            The constructed quantum circuit.
        """
        qr = QuantumRegister(self.num_qubits, "q")
        circuit = QuantumCircuit(qr)

        param_idx = 0

        # Initial RY and RZ gates on each qubit
        for i in range(self.num_qubits):
            if param_idx < len(params):
                circuit.ry(float(params[param_idx]), qr[i])
                param_idx += 1
            if param_idx < len(params):
                circuit.rz(float(params[param_idx]), qr[i])
                param_idx += 1

        for layer in range(num_layers):
            pairs = (
                entanglement_pairs_by_layer[layer]
                if layer < len(entanglement_pairs_by_layer)
                else []
            )
            for i, j in pairs:
                if not (0 <= i < self.num_qubits and 0 <= j < self.num_qubits):
                    raise QuantumCircuitError(f"invalid entanglement pair ({i}, {j})")

                circuit.cx(qr[i], qr[j])
                if param_idx < len(params):
                    circuit.ry(float(params[param_idx]), qr[j])
                    param_idx += 1
                circuit.cx(qr[i], qr[j])

            # Rotation layer
            for i in range(self.num_qubits):
                if param_idx < len(params):
                    circuit.ry(float(params[param_idx]), qr[i])
                    param_idx += 1
                if param_idx < len(params):
                    circuit.rz(float(params[param_idx]), qr[i])
                    param_idx += 1

        return circuit
