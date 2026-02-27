"""Adaptive optimization strategies for QATNE."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for optimizers used by QATNE."""

    @abstractmethod
    def step(self, params: np.ndarray, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """Perform one optimization step and return updated parameters."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal optimizer state."""


class AdaptiveOptimizer(BaseOptimizer):
    """Adaptive gradient descent with momentum and LR scheduling."""

    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """Perform single optimization step."""
        lr = self.learning_rate / np.sqrt(iteration + 1)

        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity - lr * gradient
        return params + self.velocity

    def reset(self) -> None:
        """Reset optimizer state."""
        self.velocity = None
