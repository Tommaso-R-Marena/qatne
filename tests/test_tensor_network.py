"""Tests for tensor network module."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qatne.core.tensor_network import TensorNetwork


class TestTensorNetwork:
    """Test suite for TensorNetwork class."""
    
    def test_initialization(self):
        """Test tensor network initialization."""
        tn = TensorNetwork(num_sites=4, bond_dim=2, max_bond_dim=16)
        
        assert tn.num_sites == 4
        assert tn.bond_dim == 2
        assert tn.max_bond_dim == 16
        assert tn.num_layers > 0
    
    def test_bond_dimensions(self):
        """Test bond dimension management."""
        tn = TensorNetwork(num_sites=4, bond_dim=2, max_bond_dim=8)
        
        # Check initial bond dimensions
        for i in range(3):
            assert tn.get_bond_dim(i) == 2
        
        # Increase bond dimension
        tn.increase_bond_dim(0)
        assert tn.get_bond_dim(0) == 4
        
        # Check max bound
        tn.increase_bond_dim(0)
        tn.increase_bond_dim(0)
        assert tn.get_bond_dim(0) <= 8
    
    def test_entanglement_pairs(self):
        """Test entanglement pair generation."""
        tn = TensorNetwork(num_sites=4, bond_dim=2)
        
        # Check pairs for each layer
        for layer in range(tn.num_layers):
            pairs = tn.get_entanglement_pairs(layer)
            assert isinstance(pairs, list)
            assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
            
            # Check pairs are valid
            for i, j in pairs:
                assert 0 <= i < tn.num_sites
                assert 0 <= j < tn.num_sites
                assert abs(i - j) == 1  # Nearest neighbor
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy computation."""
        tn = TensorNetwork(num_sites=4, bond_dim=4)
        
        # Create a simple product state (no entanglement)
        state_product = np.zeros(16, dtype=complex)
        state_product[0] = 1.0  # |0000>
        entropy_product = tn.compute_entanglement_entropy(state_product, partition=2)
        assert abs(entropy_product) < 1e-10  # Should be ~0
        
        # Create a maximally entangled state
        state_entangled = np.zeros(16, dtype=complex)
        state_entangled[0] = 1/np.sqrt(2)  # |0000>
        state_entangled[15] = 1/np.sqrt(2)  # |1111>
        entropy_entangled = tn.compute_entanglement_entropy(state_entangled, partition=2)
        assert entropy_entangled > 0.5  # Should be ~1 bit
    
    def test_truncation(self):
        """Test bond dimension truncation."""
        tn = TensorNetwork(num_sites=4, bond_dim=16, max_bond_dim=32)
        
        # Increase some bond dimensions
        for i in range(3):
            tn.increase_bond_dim(i)
        
        initial_max = max(tn.bond_dims.values())
        
        # Truncate
        tn.truncate(threshold=0.5)
        
        # Check that small bonds were reduced
        assert max(tn.bond_dims.values()) <= initial_max
    
    def test_layer_count(self):
        """Test number of layers computation."""
        # Small system
        tn4 = TensorNetwork(num_sites=4, bond_dim=2)
        assert tn4.num_layers >= 2
        
        # Larger system
        tn16 = TensorNetwork(num_sites=16, bond_dim=2)
        assert tn16.num_layers >= 4
        
        # Scaling check
        assert tn16.num_layers >= tn4.num_layers


class TestEntanglementStructure:
    """Test entanglement structure properties."""
    
    def test_connectivity(self):
        """Test that all qubits are connected."""
        tn = TensorNetwork(num_sites=8, bond_dim=2)
        
        # Collect all pairs across all layers
        all_pairs = set()
        for layer in range(tn.num_layers):
            pairs = tn.get_entanglement_pairs(layer)
            all_pairs.update(pairs)
        
        # Check connectivity (each qubit connected to at least one other)
        connected = set()
        for i, j in all_pairs:
            connected.add(i)
            connected.add(j)
        
        # All qubits should be reachable
        assert len(connected) == tn.num_sites
    
    def test_alternating_pattern(self):
        """Test alternating even/odd pattern in layers."""
        tn = TensorNetwork(num_sites=8, bond_dim=2)
        
        # Check first two layers have different patterns
        pairs_0 = tn.get_entanglement_pairs(0)
        pairs_1 = tn.get_entanglement_pairs(1)
        
        # Extract first elements of pairs
        first_0 = {p[0] for p in pairs_0}
        first_1 = {p[0] for p in pairs_1}
        
        # Should have minimal overlap (alternating pattern)
        overlap = first_0.intersection(first_1)
        assert len(overlap) <= 1


class TestAdaptiveBehavior:
    """Test adaptive bond dimension features."""
    
    def test_gradient_based_adaptation(self):
        """Test simulated gradient-based adaptation."""
        tn = TensorNetwork(num_sites=4, bond_dim=2, max_bond_dim=16)
        
        # Simulate high gradient at site 1
        initial_dim = tn.get_bond_dim(1)
        tn.increase_bond_dim(1)
        new_dim = tn.get_bond_dim(1)
        
        assert new_dim > initial_dim
        assert new_dim <= tn.max_bond_dim
    
    def test_max_bond_dim_constraint(self):
        """Test that bond dimensions respect maximum."""
        tn = TensorNetwork(num_sites=4, bond_dim=2, max_bond_dim=8)
        
        # Try to increase beyond max
        for _ in range(10):
            tn.increase_bond_dim(0)
        
        assert tn.get_bond_dim(0) <= tn.max_bond_dim
    
    def test_overall_bond_dim_update(self):
        """Test that overall bond_dim tracks maximum."""
        tn = TensorNetwork(num_sites=4, bond_dim=2, max_bond_dim=16)
        
        initial_overall = tn.bond_dim
        
        # Increase one bond significantly
        for _ in range(3):
            tn.increase_bond_dim(1)
        
        # Overall bond dim should reflect the increase
        assert tn.bond_dim >= initial_overall


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
