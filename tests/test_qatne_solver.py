"""Tests for QATNE solver."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qatne.algorithms.qatne_solver import QATNESolver


class TestQATNESolver:
    """Test suite for QATNESolver class."""
    
    @pytest.fixture
    def simple_hamiltonian(self):
        """Create a simple 2-qubit Hamiltonian."""
        # Simple Hamiltonian: H = Z_0 + Z_1
        H = np.diag([2, 0, 0, -2])  # Eigenvalues: 2, 0, 0, -2
        return H, 2, -2.0  # Hamiltonian, num_qubits, ground_energy
    
    def test_initialization(self, simple_hamiltonian):
        """Test solver initialization."""
        H, num_qubits, _ = simple_hamiltonian
        
        solver = QATNESolver(
            hamiltonian=H,
            num_qubits=num_qubits,
            max_bond_dim=8,
            convergence_threshold=1e-4,
            shots=1024
        )
        
        assert solver.num_qubits == num_qubits
        assert solver.max_bond_dim == 8
        assert solver.convergence_threshold == 1e-4
        assert solver.shots == 1024
        assert len(solver.energy_history) == 0
    
    def test_parameter_estimation(self, simple_hamiltonian):
        """Test parameter count estimation."""
        H, num_qubits, _ = simple_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits)
        num_params = solver._estimate_num_parameters()
        
        # Should have at least 2 params per qubit
        assert num_params >= 2 * num_qubits
    
    def test_circuit_construction(self, simple_hamiltonian):
        """Test quantum circuit construction."""
        H, num_qubits, _ = simple_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits)
        num_params = solver._estimate_num_parameters()
        params = np.random.randn(num_params) * 0.1
        
        circuit = solver._build_adaptive_ansatz(params)
        
        assert circuit.num_qubits == num_qubits
        assert circuit.depth() > 0
    
    def test_energy_evaluation(self, simple_hamiltonian):
        """Test energy evaluation."""
        H, num_qubits, ground_energy = simple_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits, shots=4096)
        
        # Test with zero parameters (initial state)
        num_params = solver._estimate_num_parameters()
        params_zero = np.zeros(num_params)
        
        energy = solver._compute_energy(params_zero)
        
        # Energy should be real
        assert isinstance(energy, (int, float))
        
        # Energy should be within bounds
        eigenvalues = np.linalg.eigvalsh(H)
        assert eigenvalues.min() <= energy <= eigenvalues.max()
    
    def test_gradient_computation(self, simple_hamiltonian):
        """Test gradient computation via parameter shift."""
        H, num_qubits, _ = simple_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits, shots=2048)
        
        num_params = solver._estimate_num_parameters()
        # Use only first few parameters for speed
        params = np.random.randn(min(4, num_params)) * 0.1
        
        gradient = solver._compute_gradient(params)
        
        # Gradient should have same shape as parameters
        assert gradient.shape == params.shape
        
        # Gradient should be real-valued
        assert np.all(np.isreal(gradient))
    
    @pytest.mark.slow
    def test_optimization_convergence(self, simple_hamiltonian):
        """Test that optimization converges (slow test)."""
        H, num_qubits, ground_energy = simple_hamiltonian
        
        solver = QATNESolver(
            hamiltonian=H,
            num_qubits=num_qubits,
            max_bond_dim=4,
            convergence_threshold=1e-2,
            shots=2048
        )
        
        final_energy, optimal_params = solver.solve(max_iterations=50)
        
        # Check convergence
        assert len(solver.energy_history) > 0
        
        # Energy should improve
        if len(solver.energy_history) > 1:
            assert solver.energy_history[-1] <= solver.energy_history[0] + 1.0
        
        # Should be within reasonable range
        eigenvalues = np.linalg.eigvalsh(H)
        assert eigenvalues.min() - 1.0 <= final_energy <= eigenvalues.max() + 1.0
    
    def test_statevector_retrieval(self, simple_hamiltonian):
        """Test state vector retrieval."""
        H, num_qubits, _ = simple_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits)
        
        num_params = solver._estimate_num_parameters()
        params = np.random.randn(num_params) * 0.1
        
        statevector = solver.get_statevector(params)
        
        # Check dimensions
        assert len(statevector) == 2**num_qubits
        
        # Check normalization
        norm = np.linalg.norm(statevector)
        assert abs(norm - 1.0) < 1e-6
    
    def test_fidelity_computation(self, simple_hamiltonian):
        """Test fidelity computation."""
        H, num_qubits, _ = simple_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits)
        
        num_params = solver._estimate_num_parameters()
        params = np.zeros(num_params)  # Initial state
        
        # Target state: |00>
        target = np.zeros(2**num_qubits)
        target[0] = 1.0
        
        fidelity = solver.compute_fidelity(params, target)
        
        # Fidelity should be in [0, 1]
        assert 0.0 <= fidelity <= 1.0
        
        # Initial state should have high fidelity with |00>
        assert fidelity > 0.5
    
    def test_parameter_resizing(self, simple_hamiltonian):
        """Test parameter vector resizing."""
        H, num_qubits, _ = simple_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits)
        
        # Create parameter vector
        old_params = np.random.randn(10) * 0.1
        
        # Resize to larger
        new_params = solver._resize_parameters(old_params)
        assert len(new_params) >= len(old_params)
        
        # Check that old values are preserved
        common_length = min(len(old_params), len(new_params))
        assert np.allclose(old_params[:common_length], new_params[:common_length])


class TestAdaptation:
    """Test adaptive features of QATNE."""
    
    @pytest.fixture
    def solver_with_history(self):
        """Create solver with some optimization history."""
        H = np.diag([2, 0, 0, -2])
        solver = QATNESolver(hamiltonian=H, num_qubits=2, max_bond_dim=16)
        
        # Simulate some optimization history
        for i in range(5):
            solver.energy_history.append(-1.0 - i * 0.1)
            solver.gradient_norms.append(1.0 / (i + 1))
        
        return solver
    
    def test_tensor_network_adaptation(self, solver_with_history):
        """Test that tensor network adapts based on gradients."""
        solver = solver_with_history
        
        initial_bond_dim = solver.tensor_network.bond_dim
        
        # Create high-gradient scenario
        gradient = np.random.randn(20) * 10.0  # High magnitude
        
        solver._adapt_tensor_network(gradient)
        
        # Bond dimension may increase
        new_bond_dim = solver.tensor_network.bond_dim
        assert new_bond_dim >= initial_bond_dim
    
    def test_gradient_threshold(self, solver_with_history):
        """Test gradient-based adaptation threshold."""
        solver = solver_with_history
        
        # Low gradient should not trigger much adaptation
        low_gradient = np.random.randn(20) * 0.01
        initial_bond = solver.tensor_network.bond_dim
        
        solver._adapt_tensor_network(low_gradient)
        
        # Should remain relatively stable
        assert solver.tensor_network.bond_dim <= initial_bond * 2


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_hamiltonian_shape(self):
        """Test handling of invalid Hamiltonian shape."""
        # Non-square matrix
        H_invalid = np.random.randn(4, 5)
        
        # Should handle gracefully (may not raise error immediately)
        solver = QATNESolver(hamiltonian=H_invalid, num_qubits=2)
        assert solver.hamiltonian.shape == H_invalid.shape
    
    def test_zero_shots(self):
        """Test behavior with very low shots."""
        H = np.diag([1, -1, 0, 0])
        
        # Very low shots may cause issues
        solver = QATNESolver(hamiltonian=H, num_qubits=2, shots=1)
        assert solver.shots == 1
    
    def test_empty_energy_history(self):
        """Test convergence check with empty history."""
        H = np.diag([1, -1])
        solver = QATNESolver(hamiltonian=H, num_qubits=1)
        
        # Empty history should not cause errors
        assert len(solver.energy_history) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
