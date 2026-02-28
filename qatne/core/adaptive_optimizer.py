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
    """Standard gradient descent with decaying learning rate."""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

    def step(
        self, params: np.ndarray, gradient: np.ndarray, iteration: int
    ) -> np.ndarray:
        lr = self.learning_rate / np.sqrt(iteration + 1)
        return params - lr * gradient

    def reset(self) -> None:
        pass


class AdamOptimizer(BaseOptimizer):
    """Adaptive Moment Estimation (Adam) optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None

    def step(
        self, params: np.ndarray, gradient: np.ndarray, iteration: int
    ) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        t = iteration + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

        m_hat = self.m / (1 - self.beta1**t)
        v_hat = self.v / (1 - self.beta2**t)

        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self) -> None:
        self.m = None
        self.v = None


# For backward compatibility
AdaptiveOptimizer = AdamOptimizer
