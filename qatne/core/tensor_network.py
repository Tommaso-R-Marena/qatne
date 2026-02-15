"""Adaptive tensor-network primitives used by the QATNE solver."""

from typing import List, Tuple

import numpy as np


class TensorNetwork:
    """Adaptive Matrix Product State style representation."""

    def __init__(
        self,
        num_sites: int,
        bond_dim: int = 4,
        max_bond_dim: int = 32,
    ) -> None:
        self.num_sites = num_sites
        self.bond_dim = bond_dim
        self.max_bond_dim = max_bond_dim
        self.num_layers = self._compute_num_layers()

        self.bond_dims = {(i, i + 1): bond_dim for i in range(num_sites - 1)}
        self.entanglement_pairs = self._initialize_entanglement()

    def _compute_num_layers(self) -> int:
        """Compute number of alternating entangling layers."""
        return int(np.ceil(np.log2(self.num_sites))) + 1

    def _initialize_entanglement(self) -> List[List[Tuple[int, int]]]:
        """Initialize nearest-neighbor pairs for each layer."""
        pairs_by_layer: List[List[Tuple[int, int]]] = []

        for layer in range(self.num_layers):
            pairs: List[Tuple[int, int]] = []
            if layer % 2 == 0:
                for i in range(0, self.num_sites - 1, 2):
                    pairs.append((i, i + 1))
            else:
                for i in range(1, self.num_sites - 1, 2):
                    pairs.append((i, i + 1))
            pairs_by_layer.append(pairs)

        return pairs_by_layer

    def get_entanglement_pairs(self, layer: int) -> List[Tuple[int, int]]:
        """Return two-qubit interaction pairs for a layer."""
        if layer < len(self.entanglement_pairs):
            return self.entanglement_pairs[layer]
        return []

    def increase_bond_dim(self, site: int) -> None:
        """Increase bond dimension at a specific bond (up to max)."""
        if site < self.num_sites - 1:
            current_dim = self.bond_dims.get((site, site + 1), self.bond_dim)
            self.bond_dims[(site, site + 1)] = min(current_dim * 2, self.max_bond_dim)
            self.bond_dim = max(self.bond_dims.values())

    def get_bond_dim(self, site: int) -> int:
        """Get bond dimension at a specific bond."""
        return self.bond_dims.get((site, site + 1), self.bond_dim)

    def compute_entanglement_entropy(self, state_vector: np.ndarray, partition: int) -> float:
        """Compute von Neumann entropy across a bipartition."""
        n = int(np.log2(len(state_vector)))
        state_matrix = state_vector.reshape(2**partition, 2 ** (n - partition))
        rho_a = state_matrix @ state_matrix.conj().T
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    def truncate(self, threshold: float = 1e-10) -> None:
        """Reduce very small bond dimensions to simplify the network."""
        for key in list(self.bond_dims.keys()):
            if self.bond_dims[key] < threshold * self.max_bond_dim:
                self.bond_dims[key] = max(2, self.bond_dims[key] // 2)

    def get_complexity(self) -> int:
        """Estimate parameter complexity from bond dimensions."""
        return sum(bond_dim**2 for bond_dim in self.bond_dims.values())

    def visualize_structure(self) -> str:
        """Create an ASCII rendering of layers and bond dimensions."""
        lines = ["", "Tensor Network Structure:", "=" * 40]

        for layer in range(self.num_layers):
            pairs = self.get_entanglement_pairs(layer)
            pair_str = ", ".join(f"({i},{j})" for i, j in pairs)
            lines.append(f"Layer {layer}: {pair_str}")

        lines.append("\nBond Dimensions:")
        for (i, j), dim in sorted(self.bond_dims.items()):
            lines.append(f"  Site {i}-{j}: χ = {dim}")

        lines.append(f"\nTotal complexity: {self.get_complexity()} parameters")
        return "\n".join(lines)
