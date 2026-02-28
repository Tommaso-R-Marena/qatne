"""Adaptive tensor-network primitives used by the QATNE solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from qatne.core.exceptions import TensorNetworkError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bond:
    """A nearest-neighbor bond in a 1D tensor network.

    Parameters
    ----------
    left : int
        Left site index.
    right : int
        Right site index.
    """

    left: int
    right: int


class TensorNetwork:
    """Adaptive Matrix Product State style representation.

    Parameters
    ----------
    num_sites : int
        Number of sites/qubits.
    bond_dim : int, default=4
        Initial bond dimension.
    max_bond_dim : int, default=32
        Maximum allowed bond dimension.
    """

    def __init__(
        self,
        num_sites: int,
        bond_dim: int = 4,
        max_bond_dim: int = 32,
    ) -> None:
        if num_sites < 1:
            raise TensorNetworkError("num_sites must be >= 1")
        if bond_dim < 1 or max_bond_dim < bond_dim:
            raise TensorNetworkError("Require 1 <= bond_dim <= max_bond_dim")

        self.num_sites = num_sites
        self.bond_dim = bond_dim
        self.max_bond_dim = max_bond_dim
        self.num_layers = self._compute_num_layers()

        self.bond_dims: dict[tuple[int, int], int] = {
            (i, i + 1): bond_dim for i in range(num_sites - 1)
        }
        self.entanglement_pairs = self._initialize_entanglement()

    def _compute_num_layers(self) -> int:
        """Compute number of alternating entangling layers."""
        return int(np.ceil(np.log2(self.num_sites))) + 1

    def _initialize_entanglement(self) -> list[list[tuple[int, int]]]:
        """Initialize nearest-neighbor pairs for each layer."""
        pairs_by_layer: list[list[tuple[int, int]]] = []

        for layer in range(self.num_layers):
            pairs: list[tuple[int, int]] = []
            start = 0 if layer % 2 == 0 else 1
            for i in range(start, self.num_sites - 1, 2):
                pairs.append((i, i + 1))
            pairs_by_layer.append(pairs)

        return pairs_by_layer

    def get_entanglement_pairs(self, layer: int) -> list[tuple[int, int]]:
        """Return two-qubit interaction pairs for a layer."""
        if layer < 0:
            raise TensorNetworkError("layer index must be non-negative")
        if layer < len(self.entanglement_pairs):
            return self.entanglement_pairs[layer]
        return []

    def increase_bond_dim(self, site: int) -> None:
        """Increase bond dimension at a specific bond (up to max)."""
        if site < 0 or site >= self.num_sites - 1:
            raise TensorNetworkError(f"invalid site index {site}")

        current_dim = self.bond_dims.get((site, site + 1), self.bond_dim)
        new_dim = min(current_dim * 2, self.max_bond_dim)
        self.bond_dims[(site, site + 1)] = new_dim
        self.bond_dim = max(self.bond_dims.values(), default=self.bond_dim)
        LOGGER.debug(
            "Increased bond dimension for (%d,%d): %d -> %d",
            site,
            site + 1,
            current_dim,
            new_dim,
        )

    def get_bond_dim(self, site: int) -> int:
        """Get bond dimension at a specific bond."""
        if site < 0 or site >= self.num_sites - 1:
            raise TensorNetworkError(f"invalid site index {site}")
        return self.bond_dims.get((site, site + 1), self.bond_dim)

    def compute_entanglement_entropy(
        self, state_vector: np.ndarray, partition: int
    ) -> float:
        """Compute von Neumann entropy across a bipartition."""
        if partition <= 0 or partition >= self.num_sites:
            raise TensorNetworkError("partition must satisfy 0 < partition < num_sites")

        expected_dim = 2**self.num_sites
        if state_vector.size != expected_dim:
            raise TensorNetworkError(
                f"state_vector has size {state_vector.size}, expected {expected_dim}"
            )

        state_matrix = state_vector.reshape(
            2**partition, 2 ** (self.num_sites - partition)
        )
        rho_a = state_matrix @ state_matrix.conj().T
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def truncate(self, threshold: float = 1e-10) -> None:
        """Reduce very small bond dimensions to simplify the network."""
        if threshold <= 0:
            raise TensorNetworkError("threshold must be positive")

        for key in list(self.bond_dims.keys()):
            if self.bond_dims[key] < threshold * self.max_bond_dim:
                self.bond_dims[key] = max(2, self.bond_dims[key] // 2)

    def get_complexity(self) -> int:
        """Estimate parameter complexity from bond dimensions."""
        return int(sum(bond_dim**2 for bond_dim in self.bond_dims.values()))

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
