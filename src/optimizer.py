"""
Optimizers for Training Neural Networks

This module implements the AdamW optimizer, which is the standard optimizer
for training transformer models. It also includes learning rate scheduling
and gradient clipping utilities.

AdamW (Adam with decoupled Weight decay) is an improvement over the original
Adam optimizer that properly implements L2 regularization.

Reference:
    - "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
    - "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)

Classes:
    AdamW: AdamW optimizer implementation

Functions:
    clip_gradient_norm: Clip gradients to prevent explosion
    get_learning_rate_with_warmup: Learning rate scheduling with warmup
"""

from typing import Dict, Optional

import numpy as np


class AdamW:
    """
    AdamW Optimizer (Adam with decoupled Weight Decay).

    AdamW is the go-to optimizer for training transformers. It combines:
    1. Adaptive learning rates per parameter (from Adam)
    2. Momentum for faster convergence
    3. Properly decoupled weight decay (L2 regularization)

    Algorithm (at each step t):
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # Momentum
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2       # Velocity (squared gradient)
        m_hat = m_t / (1 - beta1^t)                        # Bias correction
        v_hat = v_t / (1 - beta2^t)                        # Bias correction
        theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})

    Why AdamW over Adam?
        In original Adam, weight decay is applied to the gradient before
        the adaptive scaling. AdamW applies weight decay directly to the
        weights, which is the correct implementation of L2 regularization.

    Attributes:
        learning_rate: Step size for updates
        beta1: Exponential decay rate for first moment (momentum)
        beta2: Exponential decay rate for second moment (velocity)
        epsilon: Small constant for numerical stability
        weight_decay: L2 regularization coefficient
        momentum: First moment estimates for each parameter
        velocity: Second moment estimates for each parameter
        step_count: Number of optimization steps taken
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            learning_rate: Learning rate (alpha). Typical: 1e-4 to 1e-3 for LLMs
            beta1: First moment decay (momentum coefficient). Default 0.9
            beta2: Second moment decay (RMSprop-like). Default 0.999
            epsilon: Numerical stability constant. Default 1e-8
            weight_decay: L2 regularization strength. Typical: 0.01 to 0.1
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # State variables (initialized later)
        self.momentum: Dict[str, np.ndarray] = {}  # First moment (mean of gradients)
        self.velocity: Dict[
            str, np.ndarray
        ] = {}  # Second moment (variance of gradients)
        self.step_count: int = 0

        # Reference to parameters (for in-place updates)
        self._params: Optional[Dict[str, np.ndarray]] = None

    def initialize(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Initialize optimizer state for the given parameters.

        This creates zero-initialized momentum and velocity buffers
        for each parameter.

        Args:
            parameters: Dictionary of parameter name -> parameter array
        """
        self._params = parameters
        self.step_count = 0

        for name, param in parameters.items():
            # Initialize momentum (first moment) to zeros
            self.momentum[name] = np.zeros_like(param)
            # Initialize velocity (second moment) to zeros
            self.velocity[name] = np.zeros_like(param)

    def step(
        self, gradients: Dict[str, np.ndarray], learning_rate: Optional[float] = None
    ) -> None:
        """
        Perform a single optimization step.

        Updates all parameters in-place based on their gradients.

        Args:
            gradients: Dictionary of parameter name -> gradient array
            learning_rate: Optional override for learning rate (for scheduling)
        """
        if self._params is None:
            raise RuntimeError("Optimizer not initialized. Call initialize() first.")

        self.step_count += 1
        lr = learning_rate if learning_rate is not None else self.learning_rate

        # Bias correction factors
        # These correct for the fact that m and v are initialized to 0
        bias_correction_1 = 1.0 - (self.beta1**self.step_count)
        bias_correction_2 = 1.0 - (self.beta2**self.step_count)

        for name, gradient in gradients.items():
            if name not in self._params:
                continue

            param = self._params[name]

            # Update momentum (first moment estimate)
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.momentum[name] = (
                self.beta1 * self.momentum[name] + (1.0 - self.beta1) * gradient
            )

            # Update velocity (second moment estimate)
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            self.velocity[name] = self.beta2 * self.velocity[name] + (
                1.0 - self.beta2
            ) * np.square(gradient)

            # Bias-corrected estimates
            momentum_corrected = self.momentum[name] / bias_correction_1
            velocity_corrected = self.velocity[name] / bias_correction_2

            # Compute update: m_hat / (sqrt(v_hat) + eps)
            update = momentum_corrected / (np.sqrt(velocity_corrected) + self.epsilon)

            # Apply update with weight decay (AdamW: decay applied to weights directly)
            # theta = theta - lr * (update + wd * theta)
            self._params[name] -= lr * (update + self.weight_decay * param)

    def get_state(self) -> dict:
        """Get optimizer state for checkpointing."""
        return {
            "momentum": {k: v.copy() for k, v in self.momentum.items()},
            "velocity": {k: v.copy() for k, v in self.velocity.items()},
            "step_count": self.step_count,
        }

    def load_state(self, state: dict) -> None:
        """Load optimizer state from checkpoint."""
        self.momentum = {k: v.copy() for k, v in state["momentum"].items()}
        self.velocity = {k: v.copy() for k, v in state["velocity"].items()}
        self.step_count = state["step_count"]


def clip_gradient_norm(
    gradients: Dict[str, np.ndarray], max_norm: float
) -> Dict[str, np.ndarray]:
    """
    Clip gradients by global norm.

    If the total norm of all gradients exceeds max_norm, scale them down
    proportionally so the total norm equals max_norm.

    This prevents exploding gradients, which can cause training instability.

    Algorithm:
        total_norm = sqrt(sum(norm(g)^2 for g in gradients))
        if total_norm > max_norm:
            scale = max_norm / total_norm
            gradients = {name: g * scale for name, g in gradients}

    Args:
        gradients: Dictionary of parameter name -> gradient array
        max_norm: Maximum allowed gradient norm

    Returns:
        Clipped gradients (new dictionary, original unchanged)
    """
    # Compute total gradient norm
    total_norm_squared = 0.0
    for gradient in gradients.values():
        total_norm_squared += np.sum(np.square(gradient))
    total_norm = np.sqrt(total_norm_squared)

    # Compute clipping coefficient
    if total_norm > max_norm:
        clip_coefficient = max_norm / total_norm
    else:
        clip_coefficient = 1.0

    # Apply clipping
    clipped_gradients = {}
    for name, gradient in gradients.items():
        clipped_gradients[name] = gradient * clip_coefficient

    return clipped_gradients


def get_learning_rate_with_warmup(
    current_step: int,
    base_learning_rate: float,
    warmup_steps: int,
    total_steps: Optional[int] = None,
    min_learning_rate: float = 0.0,
) -> float:
    """
    Compute learning rate with linear warmup and optional cosine decay.

    Learning rate schedule:
    1. Warmup phase (0 to warmup_steps): Linear increase from 0 to base_lr
    2. Decay phase (warmup_steps to total_steps): Cosine decay to min_lr

    This schedule is standard for training transformers:
    - Warmup prevents early training instability
    - Cosine decay helps fine-tune in later stages

    Args:
        current_step: Current training step
        base_learning_rate: Target learning rate after warmup
        warmup_steps: Number of steps for linear warmup
        total_steps: Total training steps (for cosine decay). If None, no decay.
        min_learning_rate: Minimum learning rate at end of training

    Returns:
        Learning rate for the current step
    """
    if current_step < warmup_steps:
        # Linear warmup: lr = base_lr * (step / warmup_steps)
        warmup_factor = current_step / warmup_steps
        return base_learning_rate * warmup_factor

    if total_steps is None:
        # No decay, just return base learning rate
        return base_learning_rate

    # Cosine decay after warmup
    # progress goes from 0 to 1 as we go from warmup_steps to total_steps
    decay_steps = total_steps - warmup_steps
    progress = (current_step - warmup_steps) / decay_steps
    progress = min(1.0, progress)  # Clamp to [0, 1]

    # Cosine decay formula
    # lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    learning_rate = (
        min_learning_rate + (base_learning_rate - min_learning_rate) * cosine_factor
    )

    return learning_rate
