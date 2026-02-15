"""
Tests for QATNE solver.
"""

import pytest
import numpy as np
from qatne.algorithms.qatne_solver import QATNESolver
from qatne.benchmarks.molecular_systems import create_h2_hamiltonian


class TestQATNESolver:
    
    def test_initialization(self):
        """Test solver initialization"""
        H = np.random.randn(16, 16)
        H = (H + H.T) / 2  # Make Hermitian
        
        solver = QATNESolver(
            hamiltonian=H,
            num_qubits=4,
            max_bond_dim=16
        )
        
        assert solver.num_qubits == 4
        assert solver.max_bond_dim == 16
        assert len(solver.energy_history) == 0
    
    def test_parameter_estimation(self):
        """Test parameter count estimation"""
        H = np.random.randn(16, 16)
        H = (H + H.T) / 2
        
        solver = QATNESolver(hamiltonian=H, num_qubits=4)
        num_params = solver._estimate_num_parameters()
        
        # Should have reasonable number of parameters
        assert num_params > 0
        assert num_params < 1000  # Sanity check
    
    def test_bitstring_conversion(self):
        """Test bitstring to state conversion"""
        H = np.random.randn(16, 16)
        H = (H + H.T) / 2
        
        solver = QATNESolver(hamiltonian=H, num_qubits=4)
        
        state = solver._bitstring_to_state("0000")
        assert state.shape == (16,)
        assert np.abs(state[0]) == 1.0
        assert np.sum(np.abs(state)**2) == 1.0
    
    @pytest.mark.slow
    def test_h2_optimization(self):
        """Test optimization on H2 molecule"""
        H_matrix, num_qubits, exact_energy = create_h2_hamiltonian()
        
        solver = QATNESolver(
            hamiltonian=H_matrix,
            num_qubits=num_qubits,
            max_bond_dim=8,
            convergence_threshold=1e-4,
            shots=1024
        )
        
        np.random.seed(42)
        energy, params = solver.solve(max_iterations=50)
        
        # Should get reasonable energy
        assert len(solver.energy_history) > 0
        assert energy < 0  # Ground state should be negative
        
        # Should be within reasonable accuracy
        error = abs(energy - exact_energy)
        assert error < 0.1  # 0.1 Ha tolerance for short test
