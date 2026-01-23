"""
Tests for multi-head attention module.

Tests cover:
- Scaled dot-product attention
- Multi-head attention mechanism
- Causal (autoregressive) masking
- Forward and backward passes

Following TDD: these tests are written BEFORE the implementation.

Reference: "Attention Is All You Need" Section 3.2
"""

import numpy as np
import pytest


class TestScaledDotProductAttention:
    """
    Test suite for scaled dot-product attention.

    Formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where:
        Q = Query matrix (what we're looking for)
        K = Key matrix (what we're matching against)
        V = Value matrix (what we want to retrieve)
        d_k = dimension of keys (for scaling)

    Reference: "Attention Is All You Need" Section 3.2.1
    """

    def test_attention_output_shape(self):
        """Attention output should have shape (batch, seq_len, d_v)."""
        from src.attention import scaled_dot_product_attention

        batch_size = 2
        sequence_length = 10
        d_k = 64  # Key/Query dimension
        d_v = 64  # Value dimension

        query = np.random.randn(batch_size, sequence_length, d_k)
        key = np.random.randn(batch_size, sequence_length, d_k)
        value = np.random.randn(batch_size, sequence_length, d_v)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        expected_output_shape = (batch_size, sequence_length, d_v)
        assert output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, got {output.shape}"
        )

    def test_attention_weights_shape(self):
        """Attention weights should have shape (batch, seq_len, seq_len)."""
        from src.attention import scaled_dot_product_attention

        batch_size = 2
        sequence_length = 10
        d_k = 64

        query = np.random.randn(batch_size, sequence_length, d_k)
        key = np.random.randn(batch_size, sequence_length, d_k)
        value = np.random.randn(batch_size, sequence_length, d_k)

        output, attention_weights = scaled_dot_product_attention(query, key, value)

        expected_weights_shape = (batch_size, sequence_length, sequence_length)
        assert attention_weights.shape == expected_weights_shape, (
            f"Expected weights shape {expected_weights_shape}, got {attention_weights.shape}"
        )

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 along the key dimension."""
        from src.attention import scaled_dot_product_attention

        query = np.random.randn(2, 5, 32)
        key = np.random.randn(2, 5, 32)
        value = np.random.randn(2, 5, 32)

        _, attention_weights = scaled_dot_product_attention(query, key, value)

        # Sum along the last axis (attending over keys)
        weight_sums = np.sum(attention_weights, axis=-1)

        assert np.allclose(weight_sums, 1.0), (
            "Attention weights should sum to 1 along the key dimension"
        )

    def test_attention_scaling(self):
        """Attention should scale by sqrt(d_k) to prevent large dot products."""
        from src.attention import scaled_dot_product_attention

        d_k = 64  # Larger dimension
        query = np.ones((1, 1, d_k))  # All ones
        key = np.ones((1, 2, d_k))  # All ones
        value = np.array([[[1.0] * d_k], [[2.0] * d_k]])  # Different values
        value = value.transpose(1, 0, 2)  # Shape: (1, 2, d_k)

        _, weights = scaled_dot_product_attention(query, key, value)

        # Without scaling, large d_k would make softmax nearly one-hot
        # With proper scaling, weights should be more balanced
        # (both keys are identical, so weights should be equal)
        assert np.allclose(weights[0, 0, 0], weights[0, 0, 1], atol=0.01), (
            "Equal keys should get roughly equal attention weights with proper scaling"
        )

    def test_causal_mask_prevents_future_attention(self):
        """Causal mask should prevent attending to future positions."""
        from src.attention import scaled_dot_product_attention

        sequence_length = 5
        query = np.random.randn(1, sequence_length, 32)
        key = np.random.randn(1, sequence_length, 32)
        value = np.random.randn(1, sequence_length, 32)

        # Create causal mask: True where attention is allowed
        # Position i can only attend to positions <= i
        causal_mask = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))

        _, attention_weights = scaled_dot_product_attention(
            query, key, value, mask=causal_mask
        )

        # Check that future positions have zero attention weight
        for i in range(sequence_length):
            for j in range(i + 1, sequence_length):
                assert attention_weights[0, i, j] < 1e-6, (
                    f"Position {i} should not attend to future position {j}"
                )

    def test_padding_mask(self):
        """Padding mask should prevent attending to padded positions."""
        from src.attention import scaled_dot_product_attention

        batch_size = 1
        sequence_length = 5

        query = np.random.randn(batch_size, sequence_length, 32)
        key = np.random.randn(batch_size, sequence_length, 32)
        value = np.random.randn(batch_size, sequence_length, 32)

        # Mask out last 2 positions (padding)
        padding_mask = np.ones((sequence_length, sequence_length), dtype=bool)
        padding_mask[:, 3:] = False  # Positions 3 and 4 are padding

        _, attention_weights = scaled_dot_product_attention(
            query, key, value, mask=padding_mask
        )

        # Padded positions should have zero attention weight
        assert np.allclose(attention_weights[0, :, 3:], 0.0, atol=1e-6), (
            "Padded positions should have zero attention weight"
        )


class TestMultiHeadAttention:
    """
    Test suite for multi-head attention.

    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
    where head_i = Attention(Q @ W^Q_i, K @ W^K_i, V @ W^V_i)

    Reference: "Attention Is All You Need" Section 3.2.2
    """

    def test_multihead_output_shape(self):
        """Multi-head attention output should match embedding dimension."""
        from src.attention import MultiHeadAttention

        batch_size = 2
        sequence_length = 10
        embedding_dimension = 128
        num_heads = 4

        mha = MultiHeadAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        query = np.random.randn(batch_size, sequence_length, embedding_dimension)
        key = np.random.randn(batch_size, sequence_length, embedding_dimension)
        value = np.random.randn(batch_size, sequence_length, embedding_dimension)

        output = mha.forward(query, key, value)

        expected_shape = (batch_size, sequence_length, embedding_dimension)
        assert output.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {output.shape}"
        )

    def test_multihead_self_attention(self):
        """Self-attention should work when Q, K, V are the same."""
        from src.attention import MultiHeadAttention

        batch_size = 2
        sequence_length = 10
        embedding_dimension = 64
        num_heads = 4

        mha = MultiHeadAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        # Self-attention: Q = K = V = input
        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)

        output = mha.forward(input_tensor, input_tensor, input_tensor)

        assert output.shape == input_tensor.shape, (
            "Self-attention output should have same shape as input"
        )

    def test_multihead_causal_attention(self):
        """Multi-head attention should support causal masking."""
        from src.attention import MultiHeadAttention

        sequence_length = 8
        embedding_dimension = 64
        num_heads = 4

        mha = MultiHeadAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        input_tensor = np.random.randn(1, sequence_length, embedding_dimension)

        # Create causal mask
        causal_mask = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))

        output = mha.forward(input_tensor, input_tensor, input_tensor, mask=causal_mask)

        assert output.shape == input_tensor.shape, (
            "Causal attention output should have same shape as input"
        )

    def test_multihead_head_dimension(self):
        """Embedding dimension should be divisible by number of heads."""
        from src.attention import MultiHeadAttention

        # This should work
        mha = MultiHeadAttention(embedding_dimension=128, num_heads=4)
        assert mha.head_dimension == 32, "Head dimension should be d_model / num_heads"

        # This should raise an error
        with pytest.raises(ValueError):
            MultiHeadAttention(embedding_dimension=100, num_heads=3)

    def test_multihead_backward_shapes(self):
        """Backward pass should produce correct gradient shapes."""
        from src.attention import MultiHeadAttention

        batch_size = 2
        sequence_length = 8
        embedding_dimension = 64
        num_heads = 4

        mha = MultiHeadAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        _ = mha.forward(input_tensor, input_tensor, input_tensor)

        upstream_gradient = np.random.randn(
            batch_size, sequence_length, embedding_dimension
        )
        input_gradient = mha.backward(upstream_gradient)

        assert input_gradient.shape == input_tensor.shape, (
            "Input gradient should have same shape as input"
        )

    def test_multihead_has_projections(self):
        """Multi-head attention should have Q, K, V, and output projections."""
        from src.attention import MultiHeadAttention

        embedding_dimension = 128
        num_heads = 4

        mha = MultiHeadAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        # Should have projection layers
        assert hasattr(mha, "query_projection"), "Should have query projection"
        assert hasattr(mha, "key_projection"), "Should have key projection"
        assert hasattr(mha, "value_projection"), "Should have value projection"
        assert hasattr(mha, "output_projection"), "Should have output projection"


class TestAttentionNumericalGradient:
    """
    Test gradients against numerical gradients to verify correctness.
    """

    def test_attention_backward_numerical(self):
        """Verify attention backward pass against numerical gradient."""
        from src.attention import attention_backward, scaled_dot_product_attention

        np.random.seed(42)

        batch_size = 1
        seq_len = 3
        d_k = 4

        query = np.random.randn(batch_size, seq_len, d_k)
        key = np.random.randn(batch_size, seq_len, d_k)
        value = np.random.randn(batch_size, seq_len, d_k)

        # Forward pass
        output, attention_weights = scaled_dot_product_attention(query, key, value)

        # Upstream gradient (gradient of loss w.r.t. output)
        upstream_grad = np.ones_like(output)

        # Analytical gradient
        d_query, d_key, d_value = attention_backward(
            upstream_grad, query, key, value, attention_weights
        )

        # Numerical gradient for query
        epsilon = 1e-5
        numerical_grad = np.zeros_like(query)

        for i in range(query.shape[1]):
            for j in range(query.shape[2]):
                query_plus = query.copy()
                query_plus[0, i, j] += epsilon
                output_plus, _ = scaled_dot_product_attention(query_plus, key, value)

                query_minus = query.copy()
                query_minus[0, i, j] -= epsilon
                output_minus, _ = scaled_dot_product_attention(query_minus, key, value)

                numerical_grad[0, i, j] = np.sum(output_plus - output_minus) / (
                    2 * epsilon
                )

        assert np.allclose(d_query, numerical_grad, atol=1e-4), (
            "Analytical query gradient should match numerical gradient"
        )
