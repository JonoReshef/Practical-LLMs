"""
Tests for neural network layers module.

Tests cover:
- Linear: weight initialization, forward pass, backward pass
- LayerNorm: normalization, learnable parameters, backward pass
- Embedding: lookup, gradient computation

Following TDD: these tests are written BEFORE the implementation.
"""

import numpy as np


class TestLinear:
    """
    Test suite for Linear (fully connected) layer.

    Linear layer computes: y = x @ W^T + b
    where W has shape (output_features, input_features)

    Reference: Basic neural network building block used throughout transformers.
    """

    def test_linear_output_shape(self):
        """Linear layer should produce correct output shape."""
        from src.layers import Linear

        batch_size = 4
        input_features = 8
        output_features = 16

        linear_layer = Linear(
            input_features=input_features, output_features=output_features
        )

        input_tensor = np.random.randn(batch_size, input_features)
        output_tensor = linear_layer.forward(input_tensor)

        expected_shape = (batch_size, output_features)
        assert output_tensor.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output_tensor.shape}"
        )

    def test_linear_3d_input(self):
        """Linear layer should handle 3D input (batch, seq_len, features)."""
        from src.layers import Linear

        batch_size = 2
        sequence_length = 10
        input_features = 8
        output_features = 16

        linear_layer = Linear(
            input_features=input_features, output_features=output_features
        )

        input_tensor = np.random.randn(batch_size, sequence_length, input_features)
        output_tensor = linear_layer.forward(input_tensor)

        expected_shape = (batch_size, sequence_length, output_features)
        assert output_tensor.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output_tensor.shape}"
        )

    def test_linear_no_bias(self):
        """Linear layer without bias should work correctly."""
        from src.layers import Linear

        linear_layer = Linear(input_features=4, output_features=8, use_bias=False)

        assert linear_layer.bias is None, "Bias should be None when use_bias=False"

        input_tensor = np.random.randn(2, 4)
        output_tensor = linear_layer.forward(input_tensor)

        # Manual computation without bias
        expected = input_tensor @ linear_layer.weights.T
        assert np.allclose(output_tensor, expected), (
            "Forward pass should match manual computation without bias"
        )

    def test_linear_with_bias(self):
        """Linear layer with bias should add bias correctly."""
        from src.layers import Linear

        linear_layer = Linear(input_features=4, output_features=8, use_bias=True)

        assert linear_layer.bias is not None, "Bias should exist when use_bias=True"

        input_tensor = np.random.randn(2, 4)
        output_tensor = linear_layer.forward(input_tensor)

        # Manual computation with bias
        expected = input_tensor @ linear_layer.weights.T + linear_layer.bias
        assert np.allclose(output_tensor, expected), (
            "Forward pass should match manual computation with bias"
        )

    def test_linear_backward_shapes(self):
        """Backward pass should produce correct gradient shapes."""
        from src.layers import Linear

        batch_size = 4
        input_features = 8
        output_features = 16

        linear_layer = Linear(
            input_features=input_features,
            output_features=output_features,
            use_bias=True,
        )

        input_tensor = np.random.randn(batch_size, input_features)
        _ = linear_layer.forward(input_tensor)

        upstream_gradient = np.random.randn(batch_size, output_features)
        input_gradient = linear_layer.backward(upstream_gradient)

        # Check gradient shapes
        assert input_gradient.shape == input_tensor.shape, (
            "Input gradient shape must match input shape"
        )
        assert linear_layer.weight_gradient.shape == linear_layer.weights.shape, (
            "Weight gradient shape must match weight shape"
        )
        assert linear_layer.bias_gradient.shape == linear_layer.bias.shape, (
            "Bias gradient shape must match bias shape"
        )

    def test_linear_backward_numerical_gradient(self):
        """Verify backward pass against numerical gradient for weights."""
        from src.layers import Linear

        np.random.seed(42)  # For reproducibility

        linear_layer = Linear(input_features=3, output_features=2, use_bias=False)
        input_tensor = np.array([[1.0, 2.0, 3.0]])

        # Forward pass
        _ = linear_layer.forward(input_tensor)

        # Compute analytical gradient
        upstream_gradient = np.ones((1, 2))
        _ = linear_layer.backward(upstream_gradient)
        analytical_grad = linear_layer.weight_gradient.copy()

        # Compute numerical gradient
        epsilon = 1e-5
        numerical_grad = np.zeros_like(linear_layer.weights)

        for i in range(linear_layer.weights.shape[0]):
            for j in range(linear_layer.weights.shape[1]):
                # Plus epsilon
                linear_layer.weights[i, j] += epsilon
                output_plus = linear_layer.forward(input_tensor)
                loss_plus = np.sum(output_plus)

                # Minus epsilon
                linear_layer.weights[i, j] -= 2 * epsilon
                output_minus = linear_layer.forward(input_tensor)
                loss_minus = np.sum(output_minus)

                # Restore
                linear_layer.weights[i, j] += epsilon

                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        assert np.allclose(analytical_grad, numerical_grad, atol=1e-4), (
            "Analytical gradient should match numerical gradient"
        )


class TestLayerNorm:
    """
    Test suite for Layer Normalization.

    LayerNorm normalizes across the feature dimension:
        y = gamma * (x - mean) / sqrt(var + eps) + beta

    Reference: "Layer Normalization" (Ba et al., 2016)
               Used extensively in transformers for stable training.
    """

    def test_layernorm_output_shape(self):
        """LayerNorm should preserve input shape."""
        from src.layers import LayerNorm

        batch_size = 4
        sequence_length = 10
        embedding_dimension = 128

        layer_norm = LayerNorm(normalized_shape=embedding_dimension)

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        output_tensor = layer_norm.forward(input_tensor)

        assert output_tensor.shape == input_tensor.shape, (
            "LayerNorm must preserve input shape"
        )

    def test_layernorm_normalized_statistics(self):
        """LayerNorm output should have mean≈0 and variance≈1 along normalized axis."""
        from src.layers import LayerNorm

        embedding_dimension = 128
        layer_norm = LayerNorm(normalized_shape=embedding_dimension)

        input_tensor = np.random.randn(4, 10, embedding_dimension) * 5 + 3
        output_tensor = layer_norm.forward(input_tensor)

        # Check mean is close to 0 along last axis
        output_mean = np.mean(output_tensor, axis=-1)
        assert np.allclose(output_mean, 0.0, atol=1e-5), (
            "Normalized output should have mean ≈ 0"
        )

        # Check variance is close to 1 along last axis
        output_var = np.var(output_tensor, axis=-1)
        assert np.allclose(output_var, 1.0, atol=1e-4), (
            "Normalized output should have variance ≈ 1"
        )

    def test_layernorm_learnable_parameters(self):
        """LayerNorm should have learnable gamma and beta parameters."""
        from src.layers import LayerNorm

        embedding_dimension = 64
        layer_norm = LayerNorm(normalized_shape=embedding_dimension)

        assert layer_norm.gamma.shape == (embedding_dimension,), (
            "Gamma should have shape (normalized_shape,)"
        )
        assert layer_norm.beta.shape == (embedding_dimension,), (
            "Beta should have shape (normalized_shape,)"
        )

        # Initial values: gamma=1, beta=0
        assert np.allclose(layer_norm.gamma, 1.0), "Gamma should be initialized to 1"
        assert np.allclose(layer_norm.beta, 0.0), "Beta should be initialized to 0"

    def test_layernorm_affine_transformation(self):
        """LayerNorm should apply gamma scaling and beta shift."""
        from src.layers import LayerNorm

        embedding_dimension = 64
        layer_norm = LayerNorm(normalized_shape=embedding_dimension)

        # Set custom gamma and beta
        layer_norm.gamma = np.full(embedding_dimension, 2.0)  # Scale by 2
        layer_norm.beta = np.full(embedding_dimension, 1.0)  # Shift by 1

        input_tensor = np.random.randn(2, 5, embedding_dimension)
        output_tensor = layer_norm.forward(input_tensor)

        # Output mean should be approximately 1 (due to beta)
        output_mean = np.mean(output_tensor, axis=-1)
        assert np.allclose(output_mean, 1.0, atol=1e-4), (
            "Output mean should shift by beta"
        )

        # Output std should be approximately 2 (due to gamma)
        output_std = np.std(output_tensor, axis=-1)
        assert np.allclose(output_std, 2.0, atol=1e-3), (
            "Output std should scale by gamma"
        )

    def test_layernorm_backward_shapes(self):
        """LayerNorm backward should produce correct gradient shapes."""
        from src.layers import LayerNorm

        embedding_dimension = 64
        layer_norm = LayerNorm(normalized_shape=embedding_dimension)

        input_tensor = np.random.randn(4, 10, embedding_dimension)
        _ = layer_norm.forward(input_tensor)

        upstream_gradient = np.random.randn(4, 10, embedding_dimension)
        input_gradient = layer_norm.backward(upstream_gradient)

        assert input_gradient.shape == input_tensor.shape, (
            "Input gradient shape must match input shape"
        )
        assert layer_norm.gamma_gradient.shape == layer_norm.gamma.shape, (
            "Gamma gradient shape must match gamma shape"
        )
        assert layer_norm.beta_gradient.shape == layer_norm.beta.shape, (
            "Beta gradient shape must match beta shape"
        )


class TestEmbedding:
    """
    Test suite for Embedding layer.

    Embedding layer converts token IDs to dense vectors by table lookup.

    Reference: "Attention Is All You Need" Section 3.4 - Embeddings and Softmax
    """

    def test_embedding_output_shape(self):
        """Embedding should produce correct output shape."""
        from src.layers import Embedding

        vocabulary_size = 1000
        embedding_dimension = 128
        batch_size = 4
        sequence_length = 10

        embedding_layer = Embedding(
            vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension
        )

        token_ids = np.random.randint(0, vocabulary_size, (batch_size, sequence_length))
        embeddings = embedding_layer.forward(token_ids)

        expected_shape = (batch_size, sequence_length, embedding_dimension)
        assert embeddings.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {embeddings.shape}"
        )

    def test_embedding_lookup_correct(self):
        """Embedding should correctly look up vectors for token IDs."""
        from src.layers import Embedding

        vocabulary_size = 10
        embedding_dimension = 4

        embedding_layer = Embedding(
            vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension
        )

        # Test single token lookup
        token_id = 5
        token_ids = np.array([[token_id]])
        embedding = embedding_layer.forward(token_ids)

        expected = embedding_layer.embedding_table[token_id]
        assert np.allclose(embedding[0, 0], expected), (
            "Embedding lookup should return correct vector"
        )

    def test_embedding_1d_input(self):
        """Embedding should handle 1D input (single sequence)."""
        from src.layers import Embedding

        vocabulary_size = 100
        embedding_dimension = 32
        sequence_length = 5

        embedding_layer = Embedding(
            vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension
        )

        token_ids = np.random.randint(0, vocabulary_size, (sequence_length,))
        embeddings = embedding_layer.forward(token_ids)

        expected_shape = (sequence_length, embedding_dimension)
        assert embeddings.shape == expected_shape, (
            f"Expected shape {expected_shape} for 1D input, got {embeddings.shape}"
        )

    def test_embedding_backward_shapes(self):
        """Embedding backward should produce correct gradient shape."""
        from src.layers import Embedding

        vocabulary_size = 100
        embedding_dimension = 32
        batch_size = 4
        sequence_length = 10

        embedding_layer = Embedding(
            vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension
        )

        token_ids = np.random.randint(0, vocabulary_size, (batch_size, sequence_length))
        _ = embedding_layer.forward(token_ids)

        upstream_gradient = np.random.randn(
            batch_size, sequence_length, embedding_dimension
        )
        embedding_layer.backward(upstream_gradient)

        expected_shape = (vocabulary_size, embedding_dimension)
        assert embedding_layer.embedding_gradient.shape == expected_shape, (
            f"Embedding gradient should have shape {expected_shape}"
        )

    def test_embedding_gradient_accumulation(self):
        """Repeated tokens should accumulate gradients correctly."""
        from src.layers import Embedding

        vocabulary_size = 10
        embedding_dimension = 4

        embedding_layer = Embedding(
            vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension
        )

        # Use same token ID twice
        token_ids = np.array([[3, 3, 3]])  # Token 3 appears 3 times
        _ = embedding_layer.forward(token_ids)

        upstream_gradient = np.ones((1, 3, embedding_dimension))
        embedding_layer.backward(upstream_gradient)

        # Gradient for token 3 should be 3x the upstream gradient
        expected_grad_for_token_3 = 3 * np.ones(embedding_dimension)
        assert np.allclose(
            embedding_layer.embedding_gradient[3], expected_grad_for_token_3
        ), "Gradients should accumulate for repeated tokens"


class TestPositionalEncoding:
    """
    Test suite for Positional Encoding.

    Positional encoding adds position information using sine and cosine functions.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Reference: "Attention Is All You Need" Section 3.5 - Positional Encoding
    """

    def test_positional_encoding_shape(self):
        """Positional encoding should produce correct shape."""
        from src.layers import PositionalEncoding

        max_sequence_length = 512
        embedding_dimension = 128

        pos_encoding = PositionalEncoding(
            max_sequence_length=max_sequence_length,
            embedding_dimension=embedding_dimension,
        )

        # Get encoding for a specific sequence length
        sequence_length = 100
        encoding = pos_encoding.get_encoding(sequence_length)

        expected_shape = (sequence_length, embedding_dimension)
        assert encoding.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {encoding.shape}"
        )

    def test_positional_encoding_unique_positions(self):
        """Each position should have a unique encoding."""
        from src.layers import PositionalEncoding

        pos_encoding = PositionalEncoding(
            max_sequence_length=100, embedding_dimension=64
        )

        encoding = pos_encoding.get_encoding(sequence_length=50)

        # Check that no two positions have the same encoding
        for i in range(encoding.shape[0]):
            for j in range(i + 1, encoding.shape[0]):
                assert not np.allclose(encoding[i], encoding[j]), (
                    f"Positions {i} and {j} should have different encodings"
                )

    def test_positional_encoding_bounded(self):
        """Positional encoding values should be bounded in [-1, 1]."""
        from src.layers import PositionalEncoding

        pos_encoding = PositionalEncoding(
            max_sequence_length=1000, embedding_dimension=256
        )

        encoding = pos_encoding.get_encoding(sequence_length=1000)

        assert np.all(encoding >= -1.0) and np.all(encoding <= 1.0), (
            "Positional encoding values should be in [-1, 1]"
        )

    def test_positional_encoding_deterministic(self):
        """Positional encoding should be deterministic (same input = same output)."""
        from src.layers import PositionalEncoding

        pos_encoding = PositionalEncoding(
            max_sequence_length=100, embedding_dimension=64
        )

        encoding1 = pos_encoding.get_encoding(sequence_length=50)
        encoding2 = pos_encoding.get_encoding(sequence_length=50)

        assert np.allclose(encoding1, encoding2), (
            "Positional encoding should be deterministic"
        )

    def test_positional_encoding_add_to_embeddings(self):
        """Positional encoding should be addable to embeddings."""
        from src.layers import Embedding, PositionalEncoding

        vocabulary_size = 100
        embedding_dimension = 64
        sequence_length = 10

        embedding_layer = Embedding(vocabulary_size, embedding_dimension)
        pos_encoding = PositionalEncoding(
            max_sequence_length=100, embedding_dimension=embedding_dimension
        )

        token_ids = np.random.randint(0, vocabulary_size, (2, sequence_length))
        embeddings = embedding_layer.forward(token_ids)
        position_encodings = pos_encoding.get_encoding(sequence_length)

        # Should be able to add them together
        combined = embeddings + position_encodings

        assert combined.shape == embeddings.shape, (
            "Combined shape should match embedding shape"
        )
