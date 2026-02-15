"""
Tests for TensorNetwork class.
"""

import pytest
import numpy as np
from qatne.core.tensor_network import TensorNetwork


class TestTensorNetwork:
    
    def test_initialization(self):
        """Test tensor network initialization"""
        tn = TensorNetwork(num_sites=4, bond_dim=4, max_bond_dim=16)
        assert tn.num_sites == 4
        assert tn.bond_dim == 4
        assert tn.max_bond_dim == 16
        assert len(tn.bond_dims) == 3  # num_sites - 1
    
    def test_num_layers(self):
        """Test layer computation"""
        tn = TensorNetwork(num_sites=4, bond_dim=4)
        assert tn.num_layers == 3  # ceil(log2(4)) + 1
        
        tn = TensorNetwork(num_sites=8, bond_dim=4)
        assert tn.num_layers == 4  # ceil(log2(8)) + 1
    
    def test_entanglement_pairs(self):
        """Test entanglement structure"""
        tn = TensorNetwork(num_sites=4, bond_dim=4)
        
        # Layer 0 (even): (0,1), (2,3)
        pairs_0 = tn.get_entanglement_pairs(0)
        assert (0, 1) in pairs_0
        assert (2, 3) in pairs_0
        
        # Layer 1 (odd): (1,2)
        pairs_1 = tn.get_entanglement_pairs(1)
        assert (1, 2) in pairs_1
    
    def test_increase_bond_dim(self):
        """Test bond dimension increase"""
        tn = TensorNetwork(num_sites=4, bond_dim=4, max_bond_dim=16)
        
        initial_dim = tn.get_bond_dim(0)
        tn.increase_bond_dim(0)
        new_dim = tn.get_bond_dim(0)
        
        assert new_dim == min(initial_dim * 2, 16)
    
    def test_bond_dim_cap(self):
        """Test that bond dimension doesn't exceed max"""
        tn = TensorNetwork(num_sites=4, bond_dim=8, max_bond_dim=16)
        
        tn.increase_bond_dim(0)  # 8 -> 16
        assert tn.get_bond_dim(0) == 16
        
        tn.increase_bond_dim(0)  # Should stay at 16
        assert tn.get_bond_dim(0) == 16
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy computation"""
        tn = TensorNetwork(num_sites=4, bond_dim=4)
        
        # Create maximally entangled state (2 qubits)
        state = np.zeros(16, dtype=complex)
        state[0] = 1/np.sqrt(2)
        state[15] = 1/np.sqrt(2)
        
        entropy = tn.compute_entanglement_entropy(state, partition=2)
        
        # Should be positive and bounded
        assert entropy >= 0
        assert entropy <= 2  # Max entropy for 2-qubit partition
    
    def test_complexity(self):
        """Test complexity computation"""
        tn = TensorNetwork(num_sites=4, bond_dim=4)
        complexity = tn.get_complexity()
        
        # Should be sum of bond_dim^2 for each bond
        expected = 3 * (4 ** 2)  # 3 bonds, bond_dim=4
        assert complexity == expected
