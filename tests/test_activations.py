"""
Tests for activation functions module.

Tests cover:
- Softmax: numerical stability, probability distribution properties
- GELU: approximation accuracy, gradient computation
- ReLU: basic functionality, gradient computation

Following TDD: these tests are written BEFORE the implementation.
"""

import numpy as np


class TestSoftmax:
    """
    Test suite for the softmax activation function.

    Softmax converts a vector of real numbers into a probability distribution.
    Formula: softmax(x)_i = exp(x_i) / sum(exp(x_j))

    Reference: "Attention Is All You Need" Section 3.2.1
    """

    def test_softmax_output_sums_to_one(self):
        """Softmax output should be a valid probability distribution (sums to 1)."""
        from src.activations import softmax

        input_logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        output_probabilities = softmax(input_logits)

        assert np.isclose(np.sum(output_probabilities), 1.0), (
            "Softmax output must sum to 1.0"
        )

    def test_softmax_output_is_positive(self):
        """All softmax outputs should be positive (valid probabilities)."""
        from src.activations import softmax

        input_logits = np.array([-10.0, 0.0, 10.0])
        output_probabilities = softmax(input_logits)

        assert np.all(output_probabilities > 0), "All softmax outputs must be positive"

    def test_softmax_numerical_stability_large_values(self):
        """Softmax should not overflow with large input values."""
        from src.activations import softmax

        # Large values that would cause overflow in naive implementation
        input_logits = np.array([1000.0, 1001.0, 1002.0])
        output_probabilities = softmax(input_logits)

        assert not np.any(np.isnan(output_probabilities)), (
            "Softmax should handle large values without NaN"
        )
        assert not np.any(np.isinf(output_probabilities)), (
            "Softmax should handle large values without Inf"
        )
        assert np.isclose(np.sum(output_probabilities), 1.0), (
            "Softmax output must still sum to 1.0"
        )

    def test_softmax_numerical_stability_negative_values(self):
        """Softmax should handle very negative values without underflow."""
        from src.activations import softmax

        input_logits = np.array([-1000.0, -999.0, -998.0])
        output_probabilities = softmax(input_logits)

        assert not np.any(np.isnan(output_probabilities)), (
            "Softmax should handle negative values without NaN"
        )
        assert np.isclose(np.sum(output_probabilities), 1.0), (
            "Softmax output must still sum to 1.0"
        )

    def test_softmax_2d_along_last_axis(self):
        """Softmax on 2D array should apply along the last axis by default."""
        from src.activations import softmax

        input_logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        output_probabilities = softmax(input_logits, axis=-1)

        # Each row should sum to 1
        row_sums = np.sum(output_probabilities, axis=-1)
        assert np.allclose(row_sums, 1.0), (
            "Each row should sum to 1.0 when applying softmax along last axis"
        )

    def test_softmax_3d_attention_scores(self):
        """Softmax should work on 3D tensors (batch of attention score matrices)."""
        from src.activations import softmax

        batch_size = 2
        sequence_length = 4
        # Shape: (batch_size, seq_len, seq_len) - typical attention scores shape
        attention_scores = np.random.randn(batch_size, sequence_length, sequence_length)
        attention_weights = softmax(attention_scores, axis=-1)

        # Each attention distribution (last axis) should sum to 1
        assert attention_weights.shape == attention_scores.shape
        assert np.allclose(np.sum(attention_weights, axis=-1), 1.0), (
            "Attention weights must sum to 1.0 along the last axis"
        )


class TestGELU:
    """
    Test suite for GELU (Gaussian Error Linear Unit) activation.

    GELU is used in transformer models like GPT and BERT.
    Formula: GELU(x) = x * Phi(x) where Phi is the CDF of standard normal.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Reference: "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
    """

    def test_gelu_zero_input(self):
        """GELU(0) should be 0."""
        from src.activations import gelu

        result = gelu(np.array([0.0]))
        assert np.isclose(result[0], 0.0, atol=1e-6), "GELU(0) should be 0"

    def test_gelu_positive_values(self):
        """GELU should be approximately linear for large positive values."""
        from src.activations import gelu

        large_positive = np.array([10.0])
        result = gelu(large_positive)

        # For large x, GELU(x) ≈ x
        assert np.isclose(result[0], 10.0, atol=0.01), (
            "GELU should approximate identity for large positive values"
        )

    def test_gelu_negative_values(self):
        """GELU should be close to 0 for large negative values."""
        from src.activations import gelu

        large_negative = np.array([-10.0])
        result = gelu(large_negative)

        # For large negative x, GELU(x) ≈ 0
        assert np.isclose(result[0], 0.0, atol=0.01), (
            "GELU should approach 0 for large negative values"
        )

    def test_gelu_shape_preserved(self):
        """GELU should preserve input shape."""
        from src.activations import gelu

        input_tensor = np.random.randn(2, 3, 4)
        output_tensor = gelu(input_tensor)

        assert output_tensor.shape == input_tensor.shape, (
            "GELU must preserve input shape"
        )

    def test_gelu_known_values(self):
        """Test GELU against known reference values."""
        from src.activations import gelu

        # Test at x = 1.0, GELU(1) ≈ 0.8413 (using tanh approximation)
        result = gelu(np.array([1.0]))
        assert np.isclose(result[0], 0.8413, atol=0.01), (
            f"GELU(1.0) should be approximately 0.8413, got {result[0]}"
        )


class TestReLU:
    """
    Test suite for ReLU (Rectified Linear Unit) activation.

    Formula: ReLU(x) = max(0, x)
    """

    def test_relu_positive_unchanged(self):
        """Positive values should pass through unchanged."""
        from src.activations import relu

        positive_values = np.array([1.0, 2.0, 3.0])
        result = relu(positive_values)

        assert np.allclose(result, positive_values), (
            "ReLU should not change positive values"
        )

    def test_relu_negative_to_zero(self):
        """Negative values should become zero."""
        from src.activations import relu

        negative_values = np.array([-1.0, -2.0, -3.0])
        result = relu(negative_values)

        assert np.allclose(result, 0.0), "ReLU should convert negative values to zero"

    def test_relu_mixed_values(self):
        """Test ReLU with mixed positive and negative values."""
        from src.activations import relu

        mixed_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        result = relu(mixed_values)

        assert np.allclose(result, expected), (
            "ReLU should zero out negatives and keep positives"
        )

    def test_relu_shape_preserved(self):
        """ReLU should preserve input shape."""
        from src.activations import relu

        input_tensor = np.random.randn(2, 3, 4)
        output_tensor = relu(input_tensor)

        assert output_tensor.shape == input_tensor.shape, (
            "ReLU must preserve input shape"
        )


class TestSoftmaxGradient:
    """
    Test suite for softmax backward pass (gradient computation).

    The Jacobian of softmax: d(softmax_i)/d(x_j) = softmax_i * (delta_ij - softmax_j)
    """

    def test_softmax_backward_shape(self):
        """Backward pass should return gradient with same shape as input."""
        from src.activations import softmax_backward

        logits = np.array([1.0, 2.0, 3.0])
        probs = np.exp(logits) / np.sum(np.exp(logits))
        upstream_gradient = np.ones_like(probs)  # Gradient from next layer

        gradient = softmax_backward(upstream_gradient, probs)

        assert gradient.shape == logits.shape, "Gradient shape must match input shape"

    def test_softmax_backward_numerical_gradient(self):
        """Verify backward pass against numerical gradient."""
        from src.activations import softmax, softmax_backward

        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits)

        # Compute numerical gradient
        epsilon = 1e-5
        numerical_grad = np.zeros_like(logits)

        for i in range(len(logits)):
            logits_plus = logits.copy()
            logits_plus[i] += epsilon
            logits_minus = logits.copy()
            logits_minus[i] -= epsilon

            probs_plus = softmax(logits_plus)
            probs_minus = softmax(logits_minus)

            # Using sum as the scalar loss function
            numerical_grad[i] = (np.sum(probs_plus) - np.sum(probs_minus)) / (
                2 * epsilon
            )

        # Analytical gradient (upstream gradient is all ones for sum loss)
        upstream_gradient = np.ones_like(probs)
        analytical_grad = softmax_backward(upstream_gradient, probs)

        assert np.allclose(numerical_grad, analytical_grad, atol=1e-4), (
            "Analytical gradient should match numerical gradient"
        )


class TestGELUGradient:
    """
    Test suite for GELU backward pass (gradient computation).
    """

    def test_gelu_backward_shape(self):
        """Backward pass should return gradient with same shape as input."""
        from src.activations import gelu_backward

        input_values = np.array([1.0, 2.0, 3.0])
        upstream_gradient = np.ones_like(input_values)

        gradient = gelu_backward(upstream_gradient, input_values)

        assert gradient.shape == input_values.shape, (
            "Gradient shape must match input shape"
        )

    def test_gelu_backward_numerical_gradient(self):
        """Verify GELU backward pass against numerical gradient."""
        from src.activations import gelu, gelu_backward

        input_values = np.array([-1.0, 0.0, 1.0, 2.0])

        # Compute numerical gradient
        epsilon = 1e-5
        numerical_grad = np.zeros_like(input_values)

        for i in range(len(input_values)):
            x_plus = input_values.copy()
            x_plus[i] += epsilon
            x_minus = input_values.copy()
            x_minus[i] -= epsilon

            y_plus = gelu(x_plus)
            y_minus = gelu(x_minus)

            # Using sum as the scalar loss function
            numerical_grad[i] = (np.sum(y_plus) - np.sum(y_minus)) / (2 * epsilon)

        # Analytical gradient
        upstream_gradient = np.ones_like(input_values)
        analytical_grad = gelu_backward(upstream_gradient, input_values)

        assert np.allclose(numerical_grad, analytical_grad, atol=1e-4), (
            "GELU analytical gradient should match numerical gradient"
        )
