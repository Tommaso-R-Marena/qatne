"""Quantum Adaptive Tensor Network Eigensolver implementation."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
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
        hamiltonian: np.ndarray | MolecularHamiltonian | SparsePauliOp | str,
        num_qubits: int | None = None,
        max_bond_dim: int = 32,
        convergence_threshold: float = 1e-6,
        shots: int = 8192,
        show_progress: bool = True,
    ) -> None:
        if isinstance(hamiltonian, MolecularHamiltonian):
            self.hamiltonian_obj = hamiltonian
        else:
            self.hamiltonian_obj = MolecularHamiltonian(hamiltonian)

        if num_qubits is not None and num_qubits != self.hamiltonian_obj.num_qubits:
            raise QATNEError(
                f"Specified num_qubits ({num_qubits}) does not match "
                f"Hamiltonian num_qubits ({self.hamiltonian_obj.num_qubits})"
            )

        self.num_qubits = self.hamiltonian_obj.num_qubits
        self.max_bond_dim = max_bond_dim
        self.convergence_threshold = convergence_threshold
        self.shots = shots
        self.show_progress = show_progress

        self.tensor_network = TensorNetwork(
            num_sites=self.num_qubits, bond_dim=4, max_bond_dim=self.max_bond_dim
        )
        self.ansatz = AdaptiveAnsatz(num_qubits=self.num_qubits)
        self.estimator = EstimatorV2()

        self._cached_circuit: QuantumCircuit | None = None
        self._parameter_vector: ParameterVector | None = None

        self.energy_history: list[float] = []
        self.parameter_history: list[np.ndarray] = []
        self.gradient_norms: list[float] = []
        self.bond_dim_history: list[dict[tuple[int, int], int]] = []

    def _get_parameterized_circuit(self) -> tuple[QuantumCircuit, ParameterVector]:
        """Get or build the parameterized template circuit."""
        num_params = self._estimate_num_parameters()
        if (
            self._cached_circuit is None
            or self._parameter_vector is None
            or len(self._parameter_vector) != num_params
        ):
            self._parameter_vector = ParameterVector("θ", num_params)
            entanglement_pairs_by_layer = [
                self.tensor_network.get_entanglement_pairs(layer)
                for layer in range(self.tensor_network.num_layers)
            ]
            self._cached_circuit = self.ansatz.build_circuit(
                self._parameter_vector,
                self.tensor_network.num_layers,
                entanglement_pairs_by_layer,
            )
        return self._cached_circuit, self._parameter_vector

    def _compute_energy(self, params: np.ndarray) -> float:
        circuit, param_vector = self._get_parameterized_circuit()
        pub = (circuit, self.hamiltonian_obj.op, params)
        job = self.estimator.run([pub], precision=1.0 / np.sqrt(self.shots))
        result = job.result()
        evs = result[0].data.evs
        if evs.ndim > 0:
            return float(evs[0])
        return float(evs)

    def _compute_gradient(self, params: np.ndarray) -> np.ndarray:
        circuit, param_vector = self._get_parameterized_circuit()
        shift = np.pi / 2
        num_params = len(params)
        num_circuit_params = len(param_vector)

        # Truncate params if it's longer than circuit params (should not happen in practice)
        # or pad with zeros if shorter.
        active_params = np.zeros(num_circuit_params)
        n = min(num_params, num_circuit_params)
        active_params[:n] = params[:n]

        pubs = []
        for i in range(n):
            params_plus = active_params.copy()
            params_plus[i] += shift
            params_minus = active_params.copy()
            params_minus[i] -= shift
            pubs.append((circuit, self.hamiltonian_obj.op, params_plus))
            pubs.append((circuit, self.hamiltonian_obj.op, params_minus))

        job = self.estimator.run(pubs, precision=1.0 / np.sqrt(self.shots))
        results = job.result()

        gradient = np.zeros(num_params)
        for i in range(n):
            ev_plus_data = results[2 * i].data.evs
            ev_minus_data = results[2 * i + 1].data.evs
            ev_plus = (
                float(ev_plus_data[0]) if ev_plus_data.ndim > 0 else float(ev_plus_data)
            )
            ev_minus = (
                float(ev_minus_data[0])
                if ev_minus_data.ndim > 0
                else float(ev_minus_data)
            )
            gradient[i] = (ev_plus - ev_minus) / 2.0

        return gradient

    def _adapt_tensor_network(self, gradient: np.ndarray) -> None:
        # Clear cache when structure changes
        self._cached_circuit = None
        self._parameter_vector = None

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

    def solve(
        self,
        initial_params: np.ndarray | None = None,
        max_iterations: int = 1000,
        optimizer_type: Literal["adam", "gradient_descent"] = "adam",
        learning_rate: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """Solve for the ground state using adaptive optimization.

        Parameters
        ----------
        initial_params : np.ndarray, optional
            Initial parameter values.
        max_iterations : int, default=1000
            Maximum number of optimization iterations.
        optimizer_type : str, default="adam"
            Type of optimizer ('adam' or 'gradient_descent').
        learning_rate : float, optional
            Learning rate for the optimizer.

        Returns
        -------
        tuple[float, np.ndarray]
            Final energy and optimal parameters.
        """
        optimizer = self._setup_optimizer(optimizer_type, learning_rate)
        params = self._initialize_params(initial_params)

        iterator = trange(max_iterations, disable=not self.show_progress, desc="QATNE")
        for iteration in iterator:
            energy, gradient, grad_norm = self._optimization_step(params)
            self._record_history(energy, params, grad_norm)
            self._update_progress_bar(iterator, iteration, energy, grad_norm)

            if self._check_convergence():
                LOGGER.info("Converged after %d iterations", iteration)
                break

            if self._should_adapt(iteration):
                params = self._handle_adaptation(params, gradient, optimizer)

            params = optimizer.step(params, gradient, iteration)
        else:
            LOGGER.warning(
                "QATNE did not meet convergence threshold in %d iterations",
                max_iterations,
            )

        return self.energy_history[-1], self.parameter_history[-1]

    def _setup_optimizer(
        self, optimizer_type: str, learning_rate: float | None
    ) -> AdamOptimizer | GradientDescentOptimizer:
        if optimizer_type == "adam":
            return AdamOptimizer(learning_rate=learning_rate or 0.05)
        if optimizer_type == "gradient_descent":
            return GradientDescentOptimizer(learning_rate=learning_rate or 0.1)
        raise QATNEError(f"Unsupported optimizer type: {optimizer_type}")

    def _initialize_params(self, initial_params: np.ndarray | None) -> np.ndarray:
        if initial_params is None:
            initial_params = np.random.randn(self._estimate_num_parameters()) * 0.1

        params = initial_params.copy()
        LOGGER.info("Starting QATNE optimization with %d parameters", len(params))
        LOGGER.info(
            "Initial tensor network bond dimension: %d", self.tensor_network.bond_dim
        )
        return params

    def _optimization_step(self, params: np.ndarray) -> tuple[float, np.ndarray, float]:
        energy = self._compute_energy(params)
        gradient = self._compute_gradient(params)
        grad_norm = float(np.linalg.norm(gradient))
        return energy, gradient, grad_norm

    def _record_history(
        self, energy: float, params: np.ndarray, grad_norm: float
    ) -> None:
        self.energy_history.append(energy)
        self.parameter_history.append(params.copy())
        self.gradient_norms.append(grad_norm)
        self.bond_dim_history.append(self.tensor_network.bond_dims.copy())

    def _update_progress_bar(
        self, iterator: trange, iteration: int, energy: float, grad_norm: float
    ) -> None:
        if self.show_progress:
            iterator.set_postfix(
                energy=f"{energy:.6f}",
                grad=f"{grad_norm:.6f}",
                bond=self.tensor_network.bond_dim,
            )
        elif iteration % 10 == 0:
            LOGGER.info(
                "Iter %d Energy %.8f Grad %.6f Bond %d",
                iteration,
                energy,
                grad_norm,
                self.tensor_network.bond_dim,
            )

    def _check_convergence(self) -> bool:
        if len(self.energy_history) < 2:
            return False
        energy_change = abs(self.energy_history[-1] - self.energy_history[-2])
        return energy_change < self.convergence_threshold

    def _should_adapt(self, iteration: int) -> bool:
        return iteration % 50 == 0 and iteration > 0

    def _handle_adaptation(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
        optimizer: AdamOptimizer | GradientDescentOptimizer,
    ) -> np.ndarray:
        self._adapt_tensor_network(gradient)
        if len(params) != self._estimate_num_parameters():
            params = self._resize_parameters(params)
            optimizer.reset()
        return params

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
            return np.concatenate(
                [old_params, np.random.randn(new_size - len(old_params)) * 0.01]
            )
        if new_size < len(old_params):
            return old_params[:new_size]
        return old_params

    def get_statevector(self, params: np.ndarray) -> np.ndarray:
        template_circuit, param_vector = self._get_parameterized_circuit()
        circuit = template_circuit.assign_parameters({param_vector: params})
        # Using AerSimulator for statevector consistency
        sv_backend = AerSimulator(method="statevector")
        circuit.save_statevector()
        result = sv_backend.run(circuit).result()
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
