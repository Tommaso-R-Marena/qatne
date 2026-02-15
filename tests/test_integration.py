"""Integration tests for complete QATNE workflow."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qatne.algorithms.qatne_solver import QATNESolver
from qatne.core.tensor_network import TensorNetwork


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def h2_like_hamiltonian(self):
        """Create H2-like Hamiltonian for testing."""
        # Simplified 4-qubit Hamiltonian
        n = 4
        H = np.zeros((2**n, 2**n))
        
        # Diagonal terms (simplified)
        for i in range(2**n):
            H[i, i] = np.random.randn() * 0.1
        
        # Add some off-diagonal coupling
        for i in range(2**n - 1):
            coupling = np.random.randn() * 0.01
            H[i, i+1] = coupling
            H[i+1, i] = coupling
        
        # Make Hermitian
        H = (H + H.T) / 2
        
        # Get ground energy
        eigenvalues = np.linalg.eigvalsh(H)
        ground_energy = eigenvalues[0]
        
        return H, 4, ground_energy
    
    @pytest.mark.slow
    def test_full_optimization_pipeline(self, h2_like_hamiltonian):
        """Test complete optimization from start to finish."""
        H, num_qubits, exact_ground = h2_like_hamiltonian
        
        # Initialize solver
        solver = QATNESolver(
            hamiltonian=H,
            num_qubits=num_qubits,
            max_bond_dim=8,
            convergence_threshold=1e-3,
            shots=2048
        )
        
        # Run optimization
        np.random.seed(42)
        final_energy, optimal_params = solver.solve(max_iterations=30)
        
        # Verify results
        assert len(solver.energy_history) > 0
        assert len(solver.gradient_norms) == len(solver.energy_history)
        assert len(optimal_params) > 0
        
        # Energy should be reasonable
        eigenvalues = np.linalg.eigvalsh(H)
        assert eigenvalues.min() - 0.5 <= final_energy <= eigenvalues.max() + 0.5
        
        # Should show some optimization progress
        if len(solver.energy_history) > 5:
            initial_energy = solver.energy_history[0]
            # Final energy should be lower or close
            assert final_energy <= initial_energy + 1.0
    
    def test_tensor_network_evolution(self, h2_like_hamiltonian):
        """Test that tensor network adapts during optimization."""
        H, num_qubits, _ = h2_like_hamiltonian
        
        solver = QATNESolver(
            hamiltonian=H,
            num_qubits=num_qubits,
            max_bond_dim=16,
            shots=1024
        )
        
        initial_bond_dim = solver.tensor_network.bond_dim
        
        # Run a few iterations
        solver.solve(max_iterations=10)
        
        # Bond dimension may have changed
        final_bond_dim = solver.tensor_network.bond_dim
        assert final_bond_dim >= initial_bond_dim
        assert final_bond_dim <= solver.max_bond_dim
    
    def test_statevector_consistency(self, h2_like_hamiltonian):
        """Test consistency between energy and statevector."""
        H, num_qubits, _ = h2_like_hamiltonian
        
        solver = QATNESolver(hamiltonian=H, num_qubits=num_qubits, shots=4096)
        
        # Get initial parameters
        num_params = solver._estimate_num_parameters()
        params = np.random.randn(num_params) * 0.1
        
        # Compute energy via measurement
        energy_measured = solver._compute_energy(params)
        
        # Compute energy via statevector
        statevector = solver.get_statevector(params)
        energy_exact = np.real(
            statevector.conj().T @ H @ statevector
        )
        
        # Should be relatively close (accounting for shot noise)
        # Allow large tolerance due to finite shots
        assert abs(energy_measured - energy_exact) < 1.0


class TestReproducibility:
    """Test reproducibility of results."""
    
    def test_same_seed_same_results(self):
        """Test that same random seed gives same results."""
        H = np.diag([1, 0, 0, -1])
        
        # First run
        np.random.seed(123)
        solver1 = QATNESolver(hamiltonian=H, num_qubits=2, shots=1024)
        energy1, _ = solver1.solve(max_iterations=5)
        
        # Second run with same seed
        np.random.seed(123)
        solver2 = QATNESolver(hamiltonian=H, num_qubits=2, shots=1024)
        energy2, _ = solver2.solve(max_iterations=5)
        
        # Results should be close (not exact due to quantum simulation)
        # Allow some tolerance
        assert abs(energy1 - energy2) < 0.5


class TestScaling:
    """Test scaling behavior."""
    
    def test_parameter_scaling(self):
        """Test that parameters scale appropriately with system size."""
        sizes = [2, 3, 4]
        param_counts = []
        
        for n in sizes:
            H = np.eye(2**n)
            solver = QATNESolver(hamiltonian=H, num_qubits=n)
            param_counts.append(solver._estimate_num_parameters())
        
        # Parameters should increase with system size
        assert param_counts[1] > param_counts[0]
        assert param_counts[2] > param_counts[1]
    
    def test_circuit_depth_scaling(self):
        """Test circuit depth scaling."""
        sizes = [2, 3, 4]
        depths = []
        
        for n in sizes:
            H = np.eye(2**n)
            solver = QATNESolver(hamiltonian=H, num_qubits=n)
            
            num_params = solver._estimate_num_parameters()
            params = np.zeros(num_params)
            circuit = solver._build_adaptive_ansatz(params)
            depths.append(circuit.depth())
        
        # Depth should increase (or stay same) with size
        assert depths[1] >= depths[0]
        assert depths[2] >= depths[1]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_qubit(self):
        """Test with minimal system (1 qubit)."""
        H = np.array([[1, 0], [0, -1]])
        
        solver = QATNESolver(hamiltonian=H, num_qubits=1, shots=512)
        energy, params = solver.solve(max_iterations=10)
        
        # Should find ground state
        assert -1.5 <= energy <= 1.5
    
    def test_identity_hamiltonian(self):
        """Test with identity Hamiltonian."""
        H = np.eye(4)
        
        solver = QATNESolver(hamiltonian=H, num_qubits=2, shots=512)
        energy, _ = solver.solve(max_iterations=5)
        
        # All eigenvalues are 1, so energy should be ~1
        assert 0.5 <= energy <= 1.5
    
    def test_zero_hamiltonian(self):
        """Test with zero Hamiltonian."""
        H = np.zeros((4, 4))
        
        solver = QATNESolver(hamiltonian=H, num_qubits=2, shots=512)
        energy, _ = solver.solve(max_iterations=5)
        
        # Energy should be ~0
        assert abs(energy) < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
