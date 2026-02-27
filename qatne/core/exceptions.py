"""Domain-specific exceptions for the QATNE package."""


class QATNEError(Exception):
    """Base exception for QATNE-specific failures."""


class TensorNetworkError(QATNEError):
    """Raised when tensor-network construction or adaptation fails."""


class QuantumCircuitError(QATNEError):
    """Raised when ansatz circuit creation receives invalid inputs."""


class ConvergenceError(QATNEError):
    """Raised when the optimizer fails to converge within iteration limits."""
