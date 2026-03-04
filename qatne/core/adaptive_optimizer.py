"""Adaptive optimization strategies for QATNE."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for optimizers used by QATNE."""

    @abstractmethod
    def step(
        self, params: np.ndarray, gradient: np.ndarray, iteration: int
    ) -> np.ndarray:
        """Perform one optimization step and return updated parameters."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal optimizer state."""


class GradientDescentOptimizer(BaseOptimizer):
    """Standard gradient descent with optional learning rate decay."""

    def __init__(self, learning_rate: float = 0.1, decay: float = 0.0):
        self.learning_rate = learning_rate
        self.decay = decay

    def step(
        self, params: np.ndarray, gradient: np.ndarray, iteration: int
    ) -> np.ndarray:
        lr = self.learning_rate / (1.0 + self.decay * iteration)
        lr = lr / np.sqrt(iteration + 1)  # Base sqrt decay
        return params - lr * gradient

    def reset(self) -> None:
        pass


class AdamOptimizer(BaseOptimizer):
    """Adaptive Moment Estimation (Adam) optimizer with optional decay."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        decay: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay

        self.m = None
        self.v = None

    def step(
        self, params: np.ndarray, gradient: np.ndarray, iteration: int
    ) -> np.ndarray:
        current_lr = self.learning_rate / (1.0 + self.decay * iteration)

        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        t = iteration + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

        m_hat = self.m / (1 - self.beta1**t)
        v_hat = self.v / (1 - self.beta2**t)

        return params - current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self) -> None:
        self.m = None
        self.v = None


# For backward compatibility
AdaptiveOptimizer = AdamOptimizer
