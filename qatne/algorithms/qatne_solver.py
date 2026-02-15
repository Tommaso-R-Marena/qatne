"""Quantum Adaptive Tensor Network Eigensolver implementation."""

from typing import Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator, StatevectorSimulator


class QATNESolver:
    """Hybrid quantum-classical solver with adaptive tensor-network structure."""

    def __init__(
        self,
        hamiltonian: np.ndarray,
        num_qubits: int,
        max_bond_dim: int = 32,
        convergence_threshold: float = 1e-6,
        shots: int = 8192,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.num_qubits = num_qubits
        self.max_bond_dim = max_bond_dim
        self.convergence_threshold = convergence_threshold
        self.shots = shots

        self.tensor_network = self._initialize_tensor_network()
        self.backend = AerSimulator()

        self.energy_history = []
        self.parameter_history = []
        self.gradient_norms = []

    def _initialize_tensor_network(self):
        from qatne.core.tensor_network import TensorNetwork

        return TensorNetwork(num_sites=self.num_qubits, bond_dim=4, max_bond_dim=self.max_bond_dim)

    def _build_adaptive_ansatz(self, params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        circuit = QuantumCircuit(qr)

        param_idx = 0
        for i in range(self.num_qubits):
            if param_idx < len(params):
                circuit.ry(params[param_idx], qr[i])
                param_idx += 1
            if param_idx < len(params):
                circuit.rz(params[param_idx], qr[i])
                param_idx += 1

        for layer in range(self.tensor_network.num_layers):
            pairs = self.tensor_network.get_entanglement_pairs(layer)
            for i, j in pairs:
                circuit.cx(qr[i], qr[j])
                if param_idx < len(params):
                    circuit.ry(params[param_idx], qr[j])
                    param_idx += 1
                circuit.cx(qr[i], qr[j])

            for i in range(self.num_qubits):
                if param_idx < len(params) - 1:
                    circuit.ry(params[param_idx], qr[i])
                    param_idx += 1
                    circuit.rz(params[param_idx], qr[i])
                    param_idx += 1

        return circuit

    def _compute_energy(self, params: np.ndarray) -> float:
        circuit = self._build_adaptive_ansatz(params)
        circuit.measure_all()

        result = self.backend.run(circuit, shots=self.shots).result()
        counts = result.get_counts()

        energy = 0.0
        for bitstring, count in counts.items():
            prob = count / self.shots
            state_vector = self._bitstring_to_state(bitstring)
            energy += prob * np.real(state_vector.conj().T @ self.hamiltonian @ state_vector)

        return energy

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
        initial_params: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        optimizer: str = "adam",
    ) -> Tuple[float, np.ndarray]:
        if initial_params is None:
            initial_params = np.random.randn(self._estimate_num_parameters()) * 0.1

        params = initial_params.copy()
        print(f"Starting QATNE optimization with {len(params)} parameters...")
        print(f"Initial tensor network bond dimension: {self.tensor_network.bond_dim}")

        for iteration in range(max_iterations):
            energy = self._compute_energy(params)
            gradient = self._compute_gradient(params)
            grad_norm = np.linalg.norm(gradient)

            self.energy_history.append(energy)
            self.parameter_history.append(params.copy())
            self.gradient_norms.append(grad_norm)

            if iteration % 10 == 0:
                print(
                    f"Iter {iteration:4d} | Energy: {energy:12.8f} | "
                    f"||∇E||: {grad_norm:10.6f} | "
                    f"Bond dim: {self.tensor_network.bond_dim}"
                )

            if len(self.energy_history) > 1:
                energy_change = abs(self.energy_history[-1] - self.energy_history[-2])
                if energy_change < self.convergence_threshold:
                    print(f"\nConverged after {iteration} iterations!")
                    break

            if iteration % 50 == 0 and iteration > 0:
                self._adapt_tensor_network(gradient)
                if len(params) != self._estimate_num_parameters():
                    params = self._resize_parameters(params)

            learning_rate = 0.1 / np.sqrt(iteration + 1)
            params -= learning_rate * gradient

        return self.energy_history[-1], self.parameter_history[-1]

    def _estimate_num_parameters(self) -> int:
        num_params = 2 * self.num_qubits
        for layer in range(self.tensor_network.num_layers):
            pairs = self.tensor_network.get_entanglement_pairs(layer)
            num_params += len(pairs)
            num_params += 2 * self.num_qubits
        return num_params

    def _resize_parameters(self, old_params: np.ndarray) -> np.ndarray:
        new_size = self._estimate_num_parameters()
        if new_size > len(old_params):
            return np.concatenate([old_params, np.random.randn(new_size - len(old_params)) * 0.01])
        return old_params[:new_size]

    def get_statevector(self, params: np.ndarray) -> np.ndarray:
        circuit = self._build_adaptive_ansatz(params)
        result = StatevectorSimulator().run(circuit).result()
        return np.array(result.get_statevector())

    def compute_fidelity(self, params: np.ndarray, target_state: np.ndarray) -> float:
        state = self.get_statevector(params)
        return np.abs(np.vdot(state, target_state)) ** 2
