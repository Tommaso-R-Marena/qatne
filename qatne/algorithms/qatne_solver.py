"""Quantum Adaptive Tensor Network Eigensolver implementation."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, StatevectorSimulator
from tqdm.auto import trange

from qatne.core.adaptive_optimizer import AdamOptimizer, GradientDescentOptimizer
from qatne.core.exceptions import QATNEError
from qatne.core.hamiltonian import MolecularHamiltonian
from qatne.core.quantum_circuits import AdaptiveAnsatz
from qatne.core.tensor_network import TensorNetwork
from qatne.utils.metrics import compute_fidelity

LOGGER = logging.getLogger(__name__)


class QATNESolver:
    """Hybrid quantum-classical solver with adaptive tensor-network structure."""

    def __init__(
        self,
        hamiltonian: np.ndarray | MolecularHamiltonian,
        num_qubits: int | None = None,
        max_bond_dim: int = 32,
        convergence_threshold: float = 1e-6,
        shots: int = 8192,
        show_progress: bool = True,
    ) -> None:
        if isinstance(hamiltonian, np.ndarray):
            self.hamiltonian_obj = MolecularHamiltonian(hamiltonian)
        else:
            self.hamiltonian_obj = hamiltonian

        self.num_qubits = num_qubits or self.hamiltonian_obj.num_qubits
        self.max_bond_dim = max_bond_dim
        self.convergence_threshold = convergence_threshold
        self.shots = shots
        self.show_progress = show_progress

        self.tensor_network = TensorNetwork(num_sites=self.num_qubits, bond_dim=4, max_bond_dim=self.max_bond_dim)
        self.ansatz = AdaptiveAnsatz(num_qubits=self.num_qubits)
        self.backend = AerSimulator()

        self.energy_history: list[float] = []
        self.parameter_history: list[np.ndarray] = []
        self.gradient_norms: list[float] = []
        self.bond_dim_history: list[dict[tuple[int, int], int]] = []

    def _build_adaptive_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        entanglement_pairs_by_layer = [
            self.tensor_network.get_entanglement_pairs(layer)
            for layer in range(self.tensor_network.num_layers)
        ]
        return self.ansatz.build_circuit(params, self.tensor_network.num_layers, entanglement_pairs_by_layer)

    def _compute_energy(self, params: np.ndarray) -> float:
        circuit = self._build_adaptive_ansatz(params)
        circuit.measure_all()

        result = self.backend.run(circuit, shots=self.shots).result()
        counts = result.get_counts()

        energy = 0.0
        for bitstring, count in counts.items():
            prob = count / self.shots
            state_vector = self._bitstring_to_state(bitstring)
            energy += prob * self.hamiltonian_obj.compute_expectation(state_vector)

        return float(energy)

    def _compute_gradient(self, params: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(params)
        shift = np.pi / 2

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += shift
            params_minus = params.copy()
            params_minus[i] -= shift
            gradient[i] = (self._compute_energy(params_plus) - self._compute_energy(params_minus)) / 2.0

        return gradient

    def _adapt_tensor_network(self, gradient: np.ndarray) -> None:
        gradient_per_qubit = np.zeros(self.num_qubits)
        params_per_qubit = max(1, len(gradient) // self.num_qubits)

        for i in range(self.num_qubits):
            start_idx = i * params_per_qubit
            end_idx = min(start_idx + params_per_qubit, len(gradient))
            gradient_per_qubit[i] = np.linalg.norm(gradient[start_idx:end_idx])

        threshold = np.percentile(gradient_per_qubit, 75)
        for i in range(self.num_qubits - 1):
            if gradient_per_qubit[i] > threshold:
                self.tensor_network.increase_bond_dim(i)

    def _bitstring_to_state(self, bitstring: str) -> np.ndarray:
        n = len(bitstring)
        state = np.zeros(2**n, dtype=complex)
        state[int(bitstring, 2)] = 1.0
        return state

    def solve(
        self,
        initial_params: np.ndarray | None = None,
        max_iterations: int = 1000,
        optimizer_type: Literal["adam", "gradient_descent"] = "adam",
    ) -> tuple[float, np.ndarray]:

        if optimizer_type == "adam":
            optimizer = AdamOptimizer(learning_rate=0.05)
        elif optimizer_type == "gradient_descent":
            optimizer = GradientDescentOptimizer(learning_rate=0.1)
        else:
             raise QATNEError(f"Unsupported optimizer type: {optimizer_type}")

        if initial_params is None:
            initial_params = np.random.randn(self._estimate_num_parameters()) * 0.1

        params = initial_params.copy()
        LOGGER.info("Starting QATNE optimization with %d parameters", len(params))
        LOGGER.info("Initial tensor network bond dimension: %d", self.tensor_network.bond_dim)

        iterator = trange(max_iterations, disable=not self.show_progress, desc="QATNE")
        for iteration in iterator:
            energy = self._compute_energy(params)
            gradient = self._compute_gradient(params)
            grad_norm = float(np.linalg.norm(gradient))

            self.energy_history.append(energy)
            self.parameter_history.append(params.copy())
            self.gradient_norms.append(grad_norm)
            self.bond_dim_history.append(self.tensor_network.bond_dims.copy())

            if self.show_progress:
                iterator.set_postfix(energy=f"{energy:.6f}", grad=f"{grad_norm:.6f}", bond=self.tensor_network.bond_dim)
            elif iteration % 10 == 0:
                LOGGER.info("Iter %d Energy %.8f Grad %.6f Bond %d", iteration, energy, grad_norm, self.tensor_network.bond_dim)

            if len(self.energy_history) > 1:
                energy_change = abs(self.energy_history[-1] - self.energy_history[-2])
                if energy_change < self.convergence_threshold:
                    LOGGER.info("Converged after %d iterations", iteration)
                    return self.energy_history[-1], self.parameter_history[-1]

            if iteration % 50 == 0 and iteration > 0:
                self._adapt_tensor_network(gradient)
                if len(params) != self._estimate_num_parameters():
                    params = self._resize_parameters(params)
                    optimizer.reset()

            params = optimizer.step(params, gradient, iteration)

        LOGGER.warning("QATNE did not meet convergence threshold in %d iterations", max_iterations)
        return self.energy_history[-1], self.parameter_history[-1]

    def _estimate_num_parameters(self) -> int:
        # Initial RY and RZ on each qubit
        num_params = 2 * self.num_qubits

        # Gates in entanglement layers
        for layer in range(self.tensor_network.num_layers):
            pairs = self.tensor_network.get_entanglement_pairs(layer)
            # Each pair has a CX and an RY
            num_params += len(pairs)
            # Post-entanglement rotations: RY/RZ on each qubit
            num_params += 2 * self.num_qubits

        return num_params

    def _resize_parameters(self, old_params: np.ndarray) -> np.ndarray:
        new_size = self._estimate_num_parameters()
        if new_size > len(old_params):
            return np.concatenate([old_params, np.random.randn(new_size - len(old_params)) * 0.01])
        # If the parameter list is smaller, we keep the old ones (likely it never happens)
        # but the test expects it to be at least as large.
        if new_size < len(old_params):
             return old_params[:new_size]
        return old_params

    def get_statevector(self, params: np.ndarray) -> np.ndarray:
        circuit = self._build_adaptive_ansatz(params)
        result = StatevectorSimulator().run(circuit).result()
        return np.array(result.get_statevector())

    def compute_fidelity(self, params: np.ndarray, target_state: np.ndarray) -> float:
        state = self.get_statevector(params)
        return compute_fidelity(state, target_state)

    def get_bond_dim_evolution(self) -> np.ndarray:
        """Process recorded bond dimension history into a 2D array."""
        num_iterations = len(self.bond_dim_history)
        if num_iterations == 0:
            return np.array([[]])

        num_bonds = self.num_qubits - 1
        history = np.zeros((num_bonds, num_iterations))

        for it, dims in enumerate(self.bond_dim_history):
            for (i, j), dim in dims.items():
                if i < num_bonds:
                    history[i, it] = dim
        return history
