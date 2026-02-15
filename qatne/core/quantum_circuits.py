"""
Quantum circuit construction utilities.
"""

from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from typing import List, Tuple


class AdaptiveAnsatz:
    """
    Adaptive quantum circuit ansatz.
    
    Dynamically constructs circuits based on tensor network structure.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
    
    def build_circuit(self, params: np.ndarray, entanglement_pairs: List[Tuple[int, int]]) -> QuantumCircuit:
        """
        Build quantum circuit with specified entanglement structure
        
        Args:
            params: Circuit parameters
            entanglement_pairs: List of qubit pairs to entangle
        
        Returns:
            Quantum circuit
        """
        qr = QuantumRegister(self.num_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        param_idx = 0
        
        # Initial layer
        for i in range(self.num_qubits):
            circuit.ry(params[param_idx], qr[i])
            param_idx += 1
        
        # Entangling layers
        for i, j in entanglement_pairs:
            circuit.cx(qr[i], qr[j])
            if param_idx < len(params):
                circuit.ry(params[param_idx], qr[j])
                param_idx += 1
        
        return circuit
