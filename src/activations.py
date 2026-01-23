"""
Activation Functions for Neural Networks

This module implements common activation functions used in transformer models,
with both forward and backward passes for gradient computation during training.

All implementations are in pure NumPy for educational purposes.

Functions:
    softmax: Converts logits to probability distribution
    gelu: Gaussian Error Linear Unit (used in GPT/BERT)
    relu: Rectified Linear Unit

Gradient Functions:
    softmax_backward: Gradient of softmax
    gelu_backward: Gradient of GELU
    relu_backward: Gradient of ReLU

Reference:
    - "Attention Is All You Need" (Vaswani et al., 2017) - Softmax in attention
    - "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016) - GELU activation
"""

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax activation function.

    Converts a vector of arbitrary real values (logits) into a probability
    distribution where all values are positive and sum to 1.

    Mathematical Formula:
        softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    Numerical Stability:
        We subtract max(x) from all values before exponentiation to prevent
        overflow. This doesn't change the result because:
        exp(x_i - max) / sum(exp(x_j - max)) = exp(x_i) / sum(exp(x_j))

    Reference:
        "Attention Is All You Need" Section 3.2.1 - Scaled Dot-Product Attention
        uses softmax to convert attention scores to attention weights.

    Args:
        logits: Input array of any shape. Softmax is applied along the
                specified axis.
        axis: The axis along which to compute softmax. Default is -1 (last axis),
              which is standard for attention mechanisms.

    Returns:
        probabilities: Array of same shape as input, with softmax applied along
                      the specified axis. Values along that axis sum to 1.

    Example:
        >>> logits = np.array([1.0, 2.0, 3.0])
        >>> probs = softmax(logits)
        >>> print(probs)  # [0.09, 0.24, 0.67]
        >>> print(np.sum(probs))  # 1.0
    """
    # Step 1: Subtract maximum for numerical stability
    # This prevents exp() from returning inf for large values
    max_logit = np.max(logits, axis=axis, keepdims=True)
    stable_logits = logits - max_logit

    # Step 2: Compute exponentials
    # exp(x - max) is always <= 1, preventing overflow
    exponentials = np.exp(stable_logits)

    # Step 3: Normalize to get probabilities
    # Sum along the specified axis and divide
    sum_of_exponentials = np.sum(exponentials, axis=axis, keepdims=True)
    probabilities = exponentials / sum_of_exponentials

    return probabilities


def softmax_backward(
    upstream_gradient: np.ndarray, softmax_output: np.ndarray
) -> np.ndarray:
    """
    Compute the gradient of softmax with respect to its input.

    The Jacobian of softmax is:
        d(softmax_i) / d(x_j) = softmax_i * (kronecker_delta_ij - softmax_j)

    Where kronecker_delta_ij = 1 if i==j, else 0.

    For efficiency, we use the vector form:
        d_loss/d_x = softmax * (upstream_grad - sum(upstream_grad * softmax))

    This avoids explicitly constructing the full Jacobian matrix.

    Args:
        upstream_gradient: Gradient flowing back from the next layer.
                          Shape must match softmax_output.
        softmax_output: The output from the forward pass of softmax.

    Returns:
        input_gradient: Gradient with respect to the input logits.
                       Same shape as upstream_gradient.

    Mathematical Derivation:
        Let s = softmax(x), and L be the loss.
        d_L/d_x_i = sum_j (d_L/d_s_j * d_s_j/d_x_i)
                  = sum_j (upstream_j * s_j * (delta_ij - s_i))
                  = s_i * upstream_i - s_i * sum_j(upstream_j * s_j)
                  = s_i * (upstream_i - sum_j(upstream_j * s_j))
    """
    # Compute the dot product of upstream gradient and softmax output
    # This is sum_j(upstream_j * softmax_j) for each sample
    weighted_sum = np.sum(upstream_gradient * softmax_output, axis=-1, keepdims=True)

    # Apply the gradient formula
    input_gradient = softmax_output * (upstream_gradient - weighted_sum)

    return input_gradient


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Compute GELU (Gaussian Error Linear Unit) activation.

    GELU is the activation function used in GPT-2, GPT-3, and BERT.
    It provides a smooth approximation to ReLU that allows small negative
    values to pass through.

    Mathematical Formula (exact):
        GELU(x) = x * Phi(x)
        where Phi(x) is the CDF of the standard normal distribution.

    Approximation (used here for efficiency):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This approximation is accurate to within 0.1% of the exact GELU.

    Properties:
        - GELU(0) = 0
        - GELU(x) ≈ x for large positive x
        - GELU(x) ≈ 0 for large negative x
        - Smooth and differentiable everywhere

    Reference:
        "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
        https://arxiv.org/abs/1606.08415

    Args:
        x: Input array of any shape.

    Returns:
        Output array of same shape with GELU applied element-wise.

    Example:
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> gelu(x)
        array([-0.159, 0.0, 0.841])
    """
    # Constants for the tanh approximation
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)  # ≈ 0.7979
    cubic_coefficient = 0.044715

    # Compute the inner expression: sqrt(2/pi) * (x + 0.044715 * x^3)
    cubic_term = cubic_coefficient * np.power(x, 3)
    inner_expression = sqrt_2_over_pi * (x + cubic_term)

    # Apply tanh and scale
    tanh_result = np.tanh(inner_expression)

    # Final GELU formula: 0.5 * x * (1 + tanh(...))
    gelu_output = 0.5 * x * (1.0 + tanh_result)

    return gelu_output


def gelu_backward(upstream_gradient: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of GELU with respect to its input.

    The derivative of GELU (using tanh approximation) is:
        d(GELU)/dx = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx

    Where:
        z = sqrt(2/pi) * (x + 0.044715 * x^3)
        dz/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        sech^2(z) = 1 - tanh^2(z)

    Args:
        upstream_gradient: Gradient flowing back from the next layer.
        x: The original input to the forward pass.

    Returns:
        input_gradient: Gradient with respect to the input x.
    """
    # Constants
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    cubic_coefficient = 0.044715

    # Forward pass values we need
    cubic_term = cubic_coefficient * np.power(x, 3)
    z = sqrt_2_over_pi * (x + cubic_term)
    tanh_z = np.tanh(z)

    # Derivative of z with respect to x
    dz_dx = sqrt_2_over_pi * (1.0 + 3.0 * cubic_coefficient * np.power(x, 2))

    # sech^2(z) = 1 - tanh^2(z)
    sech_squared_z = 1.0 - np.power(tanh_z, 2)

    # Full derivative of GELU
    # d(GELU)/dx = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
    gelu_derivative = 0.5 * (1.0 + tanh_z) + 0.5 * x * sech_squared_z * dz_dx

    # Chain rule: multiply by upstream gradient
    input_gradient = upstream_gradient * gelu_derivative

    return input_gradient


def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute ReLU (Rectified Linear Unit) activation.

    ReLU is the simplest and most commonly used activation function.
    It passes positive values unchanged and sets negative values to zero.

    Mathematical Formula:
        ReLU(x) = max(0, x)

    Properties:
        - Non-linear (creates decision boundaries)
        - Computationally efficient
        - Can cause "dying ReLU" problem (neurons stuck at 0)
        - Non-differentiable at x=0 (we use 0 as the subgradient)

    Args:
        x: Input array of any shape.

    Returns:
        Output array of same shape with ReLU applied element-wise.

    Example:
        >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> relu(x)
        array([0., 0., 0., 1., 2.])
    """
    return np.maximum(0, x)


def relu_backward(upstream_gradient: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of ReLU with respect to its input.

    The derivative of ReLU is:
        d(ReLU)/dx = 1 if x > 0, else 0

    At x=0, the derivative is technically undefined, but we use 0
    as a subgradient (common convention).

    Args:
        upstream_gradient: Gradient flowing back from the next layer.
        x: The original input to the forward pass.

    Returns:
        input_gradient: Gradient with respect to the input x.
    """
    # Gradient is 1 where x > 0, else 0
    relu_mask = (x > 0).astype(np.float64)

    # Chain rule
    input_gradient = upstream_gradient * relu_mask

    return input_gradient
