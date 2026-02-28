"""Standard metrics for evaluating QATNE performance."""

from __future__ import annotations

import numpy as np


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute quantum state fidelity between two state vectors.

    Parameters
    ----------
    state1 : np.ndarray
        First state vector.
    state2 : np.ndarray
        Second state vector.
    """
    return float(np.abs(np.vdot(state1, state2)) ** 2)


def compute_energy_error(approx_energy: float, exact_energy: float) -> float:
    """Compute absolute error in energy (Ha)."""
    return float(np.abs(approx_energy - exact_energy))


def compute_statistical_metrics(energies: list[float], exact_energy: float) -> dict[str, float]:
    """Compute comprehensive statistical metrics from multiple trials.

    Parameters
    ----------
    energies : list[float]
        Final energies from multiple trials.
    exact_energy : float
        Exact ground state energy.
    """
    energies_arr = np.array(energies)
    errors = np.abs(energies_arr - exact_energy)

    return {
        "mean_energy": float(np.mean(energies_arr)),
        "std_energy": float(np.std(energies_arr)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "median_error": float(np.median(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
    }
