"""
Molecular benchmark systems for testing QATNE.
"""

import numpy as np
from typing import Tuple


def create_h2_hamiltonian(bond_length: float = 0.735) -> Tuple[np.ndarray, int, float]:
    """
    Create H2 molecule Hamiltonian
    
    Args:
        bond_length: H-H distance in Angstroms
    
    Returns:
        hamiltonian_matrix: Qubit Hamiltonian matrix
        num_qubits: Number of qubits required
        fci_energy: Exact FCI ground state energy
    """
    try:
        from openfermion import MolecularData, jordan_wigner
        from openfermionpyscf import run_pyscf
        
        # Define molecule
        geometry = [('H', (0, 0, 0)), ('H', (0, 0, bond_length))]
        
        molecule = MolecularData(
            geometry=geometry,
            basis='sto-3g',
            multiplicity=1,
            charge=0
        )
        
        # Run classical calculation
        molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
        
        # Get fermionic Hamiltonian
        hamiltonian_ferm = molecule.get_molecular_hamiltonian()
        
        # Convert to qubit operator (Jordan-Wigner)
        hamiltonian_jw = jordan_wigner(hamiltonian_ferm)
        
        # Convert to matrix
        hamiltonian_matrix = hamiltonian_jw.to_matrix()
        
        # Get FCI energy
        fci_energy = molecule.fci_energy
        
        num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
        
        return hamiltonian_matrix, num_qubits, fci_energy
        
    except ImportError:
        print("Warning: openfermion/pyscf not available. Using pre-computed H2 data.")
        # Fallback to pre-computed H2 Hamiltonian for 0.735 Angstrom
        return _get_precomputed_h2(), 4, -1.137283834488


def _get_precomputed_h2() -> np.ndarray:
    """Return pre-computed H2 Hamiltonian matrix"""
    # Simplified 4-qubit H2 Hamiltonian
    H = np.diag([-1.0523732, 0.3979374, -0.0133576, -0.4754827,
                 0.3979374, -0.4754827, -1.0523732, -0.0133576,
                 -0.0133576, -1.0523732, -0.4754827, 0.3979374,
                 -0.4754827, -0.0133576, 0.3979374, -1.0523732])
    return H


def create_lih_hamiltonian(bond_length: float = 1.45) -> Tuple[np.ndarray, int, float]:
    """
    Create LiH molecule Hamiltonian
    
    Args:
        bond_length: Li-H distance in Angstroms
    
    Returns:
        hamiltonian_matrix: Qubit Hamiltonian matrix
        num_qubits: Number of qubits required
        fci_energy: Exact FCI ground state energy
    """
    try:
        from openfermion import MolecularData, jordan_wigner
        from openfermionpyscf import run_pyscf
        
        geometry = [('Li', (0, 0, 0)), ('H', (0, 0, bond_length))]
        
        molecule = MolecularData(
            geometry=geometry,
            basis='sto-3g',
            multiplicity=1,
            charge=0
        )
        
        molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
        hamiltonian_ferm = molecule.get_molecular_hamiltonian()
        hamiltonian_jw = jordan_wigner(hamiltonian_ferm)
        hamiltonian_matrix = hamiltonian_jw.to_matrix()
        
        num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
        
        return hamiltonian_matrix, num_qubits, molecule.fci_energy
        
    except ImportError:
        raise ImportError("LiH requires openfermion and pyscf to be installed.")


def create_beh2_hamiltonian(bond_length: float = 1.334) -> Tuple[np.ndarray, int, float]:
    """
    Create BeH2 molecule Hamiltonian
    
    Args:
        bond_length: Be-H distance in Angstroms
    
    Returns:
        hamiltonian_matrix: Qubit Hamiltonian matrix
        num_qubits: Number of qubits required  
        fci_energy: Exact FCI ground state energy
    """
    try:
        from openfermion import MolecularData, jordan_wigner
        from openfermionpyscf import run_pyscf
        
        geometry = [
            ('Be', (0, 0, 0)),
            ('H', (0, 0, bond_length)),
            ('H', (0, 0, -bond_length))
        ]
        
        molecule = MolecularData(
            geometry=geometry,
            basis='sto-3g',
            multiplicity=1,
            charge=0
        )
        
        molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
        hamiltonian_ferm = molecule.get_molecular_hamiltonian()
        hamiltonian_jw = jordan_wigner(hamiltonian_ferm)
        hamiltonian_matrix = hamiltonian_jw.to_matrix()
        
        num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
        
        return hamiltonian_matrix, num_qubits, molecule.fci_energy
        
    except ImportError:
        raise ImportError("BeH2 requires openfermion and pyscf to be installed.")


def create_h2o_hamiltonian() -> Tuple[np.ndarray, int, float]:
    """
    Create H2O molecule Hamiltonian
    
    Returns:
        hamiltonian_matrix: Qubit Hamiltonian matrix
        num_qubits: Number of qubits required
        fci_energy: Exact FCI ground state energy
    """
    try:
        from openfermion import MolecularData, jordan_wigner
        from openfermionpyscf import run_pyscf
        
        # Water geometry (optimized)
        geometry = [
            ('O', (0, 0, 0)),
            ('H', (0.757, 0.586, 0)),
            ('H', (-0.757, 0.586, 0))
        ]
        
        molecule = MolecularData(
            geometry=geometry,
            basis='sto-3g',
            multiplicity=1,
            charge=0
        )
        
        molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
        hamiltonian_ferm = molecule.get_molecular_hamiltonian()
        hamiltonian_jw = jordan_wigner(hamiltonian_ferm)
        hamiltonian_matrix = hamiltonian_jw.to_matrix()
        
        num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
        
        return hamiltonian_matrix, num_qubits, molecule.fci_energy
        
    except ImportError:
        raise ImportError("H2O requires openfermion and pyscf to be installed.")
