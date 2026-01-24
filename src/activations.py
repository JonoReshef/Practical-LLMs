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


# =============================================================================
# EDUCATIONAL DEMO
# Run with: python -m src.activations
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ACTIVATION FUNCTIONS DEMO")
    print("=" * 70)
    print()
    print("This module provides activation functions - nonlinear transformations")
    print("that allow neural networks to learn complex patterns.")
    print()
    print("Dependencies: None (this is a foundational module)")
    print()

    # -------------------------------------------------------------------------
    # SOFTMAX: Converting scores to probabilities
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("1. SOFTMAX - Converting logits to probabilities")
    print("-" * 70)
    print()
    print("Softmax converts arbitrary numbers into a probability distribution.")
    print("Used in attention (to weight which tokens to focus on) and output")
    print("(to predict which token comes next).")
    print()

    # Example: attention scores for 4 tokens
    attention_scores = np.array([2.0, 1.0, 0.5, -1.0])
    print(f"Input (attention scores): {attention_scores}")
    print("  These could be 'how relevant is each token to the current token'")
    print()

    attention_weights = softmax(attention_scores)
    print(f"Output (attention weights): {attention_weights}")
    print(f"  Sum of weights: {attention_weights.sum():.6f} (always 1.0)")
    print()
    print("Notice: Higher input scores get higher probabilities.")
    print(
        "        Score 2.0 -> {:.1%}, Score -1.0 -> {:.1%}".format(
            attention_weights[0], attention_weights[3]
        )
    )
    print()

    # Numerical stability demonstration
    print("Numerical stability:")
    large_values = np.array([1000.0, 1001.0, 1002.0])
    print(f"  Large inputs: {large_values}")
    print(f"  Softmax handles them: {softmax(large_values)}")
    print("  (Without the max-subtraction trick, this would overflow)")
    print()

    # -------------------------------------------------------------------------
    # GELU: The activation function used in transformers
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("2. GELU - Gaussian Error Linear Unit")
    print("-" * 70)
    print()
    print("GELU is used in the feed-forward networks inside transformer blocks.")
    print("Unlike ReLU which harshly cuts off negatives, GELU has a smooth curve.")
    print()

    x_values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    gelu_values = gelu(x_values)
    relu_values = relu(x_values)

    print("Comparing GELU vs ReLU:")
    print()
    print("    x     |   GELU   |   ReLU   ")
    print("  --------|----------|----------")
    for x, g, r in zip(x_values, gelu_values, relu_values):
        print(f"  {x:6.2f}  | {g:8.4f} | {r:8.4f}")
    print()
    print("Key insight: GELU allows small negative values through,")
    print("             which can help the model learn more nuanced patterns.")
    print()

    # -------------------------------------------------------------------------
    # BACKWARD PASS: How gradients flow
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("3. BACKWARD PASS - Computing gradients for learning")
    print("-" * 70)
    print()
    print("During training, we need to compute how changing the input would")
    print("affect the output. This is the 'gradient' or 'derivative'.")
    print()

    x = np.array([[-1.0, 0.5, 2.0]])
    upstream_grad = np.array([[1.0, 1.0, 1.0]])  # Gradient from next layer

    print(f"Input x: {x[0]}")
    print(f"Upstream gradient (from next layer): {upstream_grad[0]}")
    print()

    gelu_grad = gelu_backward(upstream_grad, x)
    relu_grad = relu_backward(upstream_grad, x)

    print("Gradients (how much each input affects the output):")
    print(f"  GELU gradient: {gelu_grad[0]}")
    print(f"  ReLU gradient: {relu_grad[0]}")
    print()
    print("Notice: ReLU has gradient 0 for x=-1 (completely blocked),")
    print("        while GELU has a small gradient (still learning).")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("- Softmax: Turns scores into probabilities (for attention & prediction)")
    print("- GELU: Smooth activation function (inside transformer blocks)")
    print("- Backward functions: Enable learning by computing gradients")
    print()
    print("Next step: Run 'python -m src.layers' to see how these are used")
    print("           in neural network layers.")
