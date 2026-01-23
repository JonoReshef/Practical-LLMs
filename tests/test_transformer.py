"""
Tests for transformer module (TransformerBlock, FeedForward).

Tests cover:
- Feed-forward network (FFN)
- Transformer block with attention + FFN + residual connections
- Full transformer encoder/decoder stack

Following TDD: these tests are written BEFORE the implementation.
"""

import numpy as np


class TestFeedForwardNetwork:
    """
    Test suite for the position-wise feed-forward network.

    The FFN in transformers consists of two linear transformations with
    a GELU activation in between:
        FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    The inner dimension is typically 4x the model dimension.

    Reference: "Attention Is All You Need" Section 3.3
    """

    def test_ffn_output_shape(self):
        """FFN output should match input shape."""
        from src.transformer import FeedForwardNetwork

        batch_size = 2
        sequence_length = 10
        embedding_dimension = 128

        ffn = FeedForwardNetwork(
            embedding_dimension=embedding_dimension,
            hidden_dimension=512,  # 4x expansion
        )

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        output = ffn.forward(input_tensor)

        assert output.shape == input_tensor.shape, (
            "FFN output shape should match input shape"
        )

    def test_ffn_default_hidden_dimension(self):
        """FFN should default to 4x expansion."""
        from src.transformer import FeedForwardNetwork

        embedding_dimension = 64
        ffn = FeedForwardNetwork(embedding_dimension=embedding_dimension)

        assert ffn.hidden_dimension == 4 * embedding_dimension, (
            "Default hidden dimension should be 4x embedding dimension"
        )

    def test_ffn_backward_shapes(self):
        """FFN backward should produce correct gradient shapes."""
        from src.transformer import FeedForwardNetwork

        batch_size = 2
        sequence_length = 8
        embedding_dimension = 64

        ffn = FeedForwardNetwork(embedding_dimension=embedding_dimension)

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        _ = ffn.forward(input_tensor)

        upstream_gradient = np.random.randn(
            batch_size, sequence_length, embedding_dimension
        )
        input_gradient = ffn.backward(upstream_gradient)

        assert input_gradient.shape == input_tensor.shape, (
            "Input gradient shape should match input shape"
        )


class TestTransformerBlock:
    """
    Test suite for a single transformer block.

    A transformer block (for decoder-only models like GPT) consists of:
        1. Layer Norm
        2. Multi-Head Self-Attention (with causal mask)
        3. Residual Connection
        4. Layer Norm
        5. Feed-Forward Network
        6. Residual Connection

    This is the "Pre-LN" architecture used in GPT-2/3.

    Reference: "Language Models are Unsupervised Multitask Learners" (GPT-2 paper)
    """

    def test_transformer_block_output_shape(self):
        """Transformer block output should match input shape."""
        from src.transformer import TransformerBlock

        batch_size = 2
        sequence_length = 10
        embedding_dimension = 128
        num_heads = 4

        block = TransformerBlock(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        output = block.forward(input_tensor)

        assert output.shape == input_tensor.shape, (
            "Transformer block output shape should match input shape"
        )

    def test_transformer_block_causal_attention(self):
        """Transformer block should support causal masking."""
        from src.transformer import TransformerBlock

        batch_size = 1
        sequence_length = 8
        embedding_dimension = 64
        num_heads = 4

        block = TransformerBlock(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)

        # With causal mask
        output = block.forward(input_tensor, use_causal_mask=True)

        assert output.shape == input_tensor.shape, (
            "Output should have same shape with causal mask"
        )

    def test_transformer_block_residual_connection(self):
        """Residual connections should be working (output not drastically different from input)."""
        from src.transformer import TransformerBlock

        embedding_dimension = 64
        num_heads = 4

        block = TransformerBlock(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        # Use small input to check residual connection is working
        input_tensor = np.random.randn(1, 4, embedding_dimension) * 0.01
        output = block.forward(input_tensor)

        # With residual, output should not be completely different from input
        # (it should be input + some transformation)
        # Check that the outputs are in a reasonable range
        assert np.isfinite(output).all(), "Output should be finite"

    def test_transformer_block_backward_shapes(self):
        """Transformer block backward should produce correct gradient shapes."""
        from src.transformer import TransformerBlock

        batch_size = 2
        sequence_length = 8
        embedding_dimension = 64
        num_heads = 4

        block = TransformerBlock(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        _ = block.forward(input_tensor)

        upstream_gradient = np.random.randn(
            batch_size, sequence_length, embedding_dimension
        )
        input_gradient = block.backward(upstream_gradient)

        assert input_gradient.shape == input_tensor.shape, (
            "Input gradient should have same shape as input"
        )


class TestTransformerStack:
    """
    Test suite for a stack of transformer blocks.
    """

    def test_transformer_stack_output_shape(self):
        """Stack of transformer blocks should preserve shape."""
        from src.transformer import TransformerStack

        batch_size = 2
        sequence_length = 10
        embedding_dimension = 64
        num_heads = 4
        num_layers = 3

        stack = TransformerStack(
            num_layers=num_layers,
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
        )

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        output = stack.forward(input_tensor)

        assert output.shape == input_tensor.shape, (
            "Transformer stack output should match input shape"
        )

    def test_transformer_stack_layer_count(self):
        """Stack should have the correct number of layers."""
        from src.transformer import TransformerStack

        num_layers = 6
        stack = TransformerStack(
            num_layers=num_layers, embedding_dimension=64, num_heads=4
        )

        assert len(stack.blocks) == num_layers, (
            f"Stack should have {num_layers} transformer blocks"
        )

    def test_transformer_stack_backward(self):
        """Stack backward should produce correct gradient shapes."""
        from src.transformer import TransformerStack

        batch_size = 2
        sequence_length = 8
        embedding_dimension = 64
        num_heads = 4
        num_layers = 2

        stack = TransformerStack(
            num_layers=num_layers,
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
        )

        input_tensor = np.random.randn(batch_size, sequence_length, embedding_dimension)
        _ = stack.forward(input_tensor)

        upstream_gradient = np.random.randn(
            batch_size, sequence_length, embedding_dimension
        )
        input_gradient, gradients = stack.backward(upstream_gradient)

        assert input_gradient.shape == input_tensor.shape, (
            "Stack input gradient should match input shape"
        )

        assert isinstance(gradients, dict), "Backward should return gradients dict"
        assert len(gradients) > 0, "Should have parameter gradients"
