"""
Adaptive optimization strategies for QATNE.
"""

import numpy as np
from typing import Callable, Tuple


class AdaptiveOptimizer:
    """
    Adaptive gradient descent with learning rate scheduling.
    """
    
    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params: np.ndarray, gradient: np.ndarray, iteration: int) -> np.ndarray:
        """
        Perform single optimization step
        
        Args:
            params: Current parameters
            gradient: Gradient vector
            iteration: Current iteration number
        
        Returns:
            Updated parameters
        """
        # Adaptive learning rate
        lr = self.learning_rate / np.sqrt(iteration + 1)
        
        # Momentum update
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity - lr * gradient
        params_new = params + self.velocity
        
        return params_new
    
    def reset(self):
        """Reset optimizer state"""
        self.velocity = None
