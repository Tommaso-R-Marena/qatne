"""State-of-the-art visualization utilities for QATNE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from scipy import stats

if TYPE_CHECKING:
    from qatne.core.tensor_network import TensorNetwork

sns.set_style("whitegrid")


def plot_convergence(
    energy_history: list[float],
    exact_energy: float | None = None,
    title: str = "QATNE Convergence",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot energy convergence history.

    Parameters
    ----------
    energy_history : list[float]
        List of energies from each iteration.
    exact_energy : float, optional
        Exact ground state energy for comparison.
    title : str, default="QATNE Convergence"
        Plot title.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(len(energy_history))
    ax.plot(iterations, energy_history, "b-", linewidth=2, label="QATNE Energy")

    if exact_energy is not None:
        ax.axhline(exact_energy, color="red", linestyle="--", linewidth=2, label="Exact (FCI)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_energy_landscape(
    theta1_range: np.ndarray,
    theta2_range: np.ndarray,
    energy_surface: np.ndarray,
    title: str = "QATNE Energy Landscape",
) -> go.Figure:
    """Create a 3D interactive energy landscape.

    Parameters
    ----------
    theta1_range : np.ndarray
        Range of values for first parameter.
    theta2_range : np.ndarray
        Range of values for second parameter.
    energy_surface : np.ndarray
        2D array of energy values.
    title : str, default="QATNE Energy Landscape"
        Plot title.
    """
    Theta1, Theta2 = np.meshgrid(theta1_range, theta2_range)

    fig = go.Figure(
        data=[
            go.Surface(
                x=Theta1,
                y=Theta2,
                z=energy_surface,
                colorscale="Viridis",
                contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}},
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Parameter θ₁",
            yaxis_title="Parameter θ₂",
            zaxis_title="Energy (Ha)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=800,
        height=600,
    )
    return fig


def plot_tensor_network_structure(
    tn: TensorNetwork,
    title: str = "Adaptive Tensor Network Structure",
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize the tensor network graph and bond dimensions.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network object to visualize.
    title : str, default="Adaptive Tensor Network Structure"
        Plot title.
    save_path : str, optional
        Path to save the figure.
    """
    G = nx.Graph()
    num_sites = tn.num_sites

    # Add nodes
    for i in range(num_sites):
        G.add_node(f"T{i}")

    # Add edges with bond dimensions
    for (i, j), dim in tn.bond_dims.items():
        G.add_edge(f"T{i}", f"T{j}", weight=dim)

    fig, ax = plt.subplots(figsize=(12, 7))
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1200, alpha=0.9, ax=ax)

    # Draw edges with width proportional to bond dimension
    edges = G.edges()
    weights = [G[u][v]["weight"] / 4 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color="gray", ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # Add edge labels
    edge_labels = {(u, v): f"χ={G[u][v]['weight']}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    ax.set_title(title + "\n(Edge width ∝ bond dimension)", fontsize=14, fontweight="bold")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_circuit_evolution(
    circuits: list[QuantumCircuit],
    titles: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize evolution of the quantum circuit ansatz.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        List of circuits at different stages.
    titles : list[str], optional
        Titles for each stage.
    save_path : str, optional
        Path to save the figure.
    """
    n = len(circuits)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    if titles is None:
        titles = [f"Stage {i}" for i in range(n)]

    for idx, (circ, title) in enumerate(zip(circuits, titles)):
        circuit_drawer(circ, output="mpl", style="iqx", ax=axes[idx])
        axes[idx].set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_statistical_analysis(
    trial_energies: list[float],
    exact_energy: float,
    save_path: str | None = None,
) -> plt.Figure:
    """Comprehensive statistical analysis of multiple QATNE trials.

    Parameters
    ----------
    trial_energies : list[float]
        Final energies from multiple independent runs.
    exact_energy : float
        Exact ground state energy.
    save_path : str, optional
        Path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distribution
    ax = axes[0, 0]
    ax.hist(trial_energies, bins=12, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(exact_energy, color="red", linestyle="--", linewidth=2, label="Exact")
    ax.axvline(np.mean(trial_energies), color="green", linestyle="-", linewidth=2, label="Mean")
    ax.set_xlabel("Energy (Ha)")
    ax.set_ylabel("Frequency")
    ax.set_title("Energy Distribution")
    ax.legend()

    # Q-Q plot
    ax = axes[0, 1]
    stats.probplot(trial_energies, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Normality Test)")

    # Box plot
    ax = axes[1, 0]
    ax.boxplot([trial_energies], labels=["QATNE"], vert=True)
    ax.axhline(exact_energy, color="red", linestyle="--", linewidth=2, label="Exact")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("Box Plot")
    ax.legend()

    # Cumulative distribution
    ax = axes[1, 1]
    sorted_energies = np.sort(trial_energies)
    cumulative = np.arange(1, len(sorted_energies) + 1) / len(sorted_energies)
    ax.plot(sorted_energies, cumulative, "b-", linewidth=2)
    ax.axvline(exact_energy, color="red", linestyle="--", linewidth=2, label="Exact")
    ax.set_xlabel("Energy (Ha)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_bond_dimension_evolution(
    bond_dim_history: np.ndarray,
    title: str = "Adaptive Bond Dimension Evolution",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot heatmap of bond dimension evolution over iterations.

    Parameters
    ----------
    bond_dim_history : np.ndarray
        2D array of shape (num_bonds, num_iterations).
    title : str, default="Adaptive Bond Dimension Evolution"
        Plot title.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(bond_dim_history, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bond Index")
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Bond Dimension χ")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_entanglement_spectrum(
    singular_values: np.ndarray,
    title: str = "Entanglement Spectrum",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot entanglement spectrum (singular values).

    Parameters
    ----------
    singular_values : np.ndarray
        Singular values from a bipartition.
    title : str, default="Entanglement Spectrum"
        Plot title.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(singular_values, "bo", markersize=8)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def create_interactive_dashboard(
    iterations: np.ndarray,
    energy: np.ndarray,
    gradient_norm: np.ndarray,
    bond_dim_avg: np.ndarray,
    params_trajectory: np.ndarray | None = None,
    exact_energy: float | None = None,
) -> go.Figure:
    """Create an interactive dashboard for QATNE optimization tracking.

    Parameters
    ----------
    iterations : np.ndarray
        Iteration indices.
    energy : np.ndarray
        Energy history.
    gradient_norm : np.ndarray
        Gradient norm history.
    bond_dim_avg : np.ndarray
        Average bond dimension history.
    params_trajectory : np.ndarray, optional
        2D array of parameter trajectory for visualization.
    exact_energy : float, optional
        Exact energy for baseline.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Energy Convergence", "Gradient Norm", "Average Bond Dimension", "Parameter Space Trajectory"),
    )

    # Energy
    fig.add_trace(go.Scatter(x=iterations, y=energy, mode="lines", name="Energy", line=dict(color="blue")), row=1, col=1)
    if exact_energy is not None:
        fig.add_hline(y=exact_energy, line_dash="dash", line_color="red", annotation_text="Exact", row=1, col=1)

    # Gradient
    fig.add_trace(
        go.Scatter(x=iterations, y=gradient_norm, mode="lines", name="||∇E||", line=dict(color="green")), row=1, col=2
    )

    # Bond dimension
    fig.add_trace(
        go.Scatter(x=iterations, y=bond_dim_avg, mode="lines", name="Avg χ", line=dict(color="purple")), row=2, col=1
    )

    # Trajectory
    if params_trajectory is not None and params_trajectory.shape[1] >= 2:
        fig.add_trace(
            go.Scatter(
                x=params_trajectory[:, 0],
                y=params_trajectory[:, 1],
                mode="lines+markers",
                name="Trajectory",
                marker=dict(size=4, color=iterations, colorscale="Viridis", showscale=True),
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=800, title_text="QATNE Interactive Optimization Dashboard", showlegend=False)
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_xaxes(title_text="θ₁", row=2, col=2)

    fig.update_yaxes(title_text="Energy (Ha)", row=1, col=1)
    fig.update_yaxes(title_text="||∇E||", row=1, col=2)
    fig.update_yaxes(title_text="Bond Dim", row=2, col=1)
    fig.update_yaxes(title_text="θ₂", row=2, col=2)

    return fig


# Aliases for compatibility with __init__.py names
plot_circuit_diagram = plot_circuit_evolution
