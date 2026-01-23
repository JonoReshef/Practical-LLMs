"""
Tests for the GPT Model Module

Tests the complete GPT model implementation including:
- Token and position embedding
- Forward pass through transformer stack
- Loss computation (cross-entropy)
- Backward pass (gradient computation)
- Text generation (autoregressive)
"""

import numpy as np
import pytest

from src.model import (
    GPTConfig,
    GPTModel,
    cross_entropy_loss,
    cross_entropy_loss_backward,
)


class TestGPTConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test that default config has sensible values."""
        config = GPTConfig()

        assert config.vocab_size > 0
        assert config.embedding_dim > 0
        assert config.num_heads > 0
        assert config.num_layers > 0
        assert config.ffn_hidden_dim > 0
        assert config.max_sequence_length > 0
        assert 0 <= config.dropout_prob < 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = GPTConfig(
            vocab_size=1000,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            ffn_hidden_dim=256,
            max_sequence_length=64,
        )

        assert config.vocab_size == 1000
        assert config.embedding_dim == 64
        assert config.num_heads == 4
        assert config.num_layers == 2
        assert config.ffn_hidden_dim == 256
        assert config.max_sequence_length == 64


class TestGPTModel:
    """Test the complete GPT model."""

    @pytest.fixture
    def small_config(self):
        """Small model config for testing."""
        return GPTConfig(
            vocab_size=100,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_hidden_dim=64,
            max_sequence_length=16,
        )

    @pytest.fixture
    def model(self, small_config):
        """Create a small model for testing."""
        return GPTModel(small_config)

    def test_model_initialization(self, model, small_config):
        """Test that model initializes with correct shapes."""
        # Check token embedding shape
        assert model.token_embedding.weight.shape == (
            small_config.vocab_size,
            small_config.embedding_dim,
        )

        # Check output projection shape
        assert model.output_projection.weight.shape == (
            small_config.vocab_size,
            small_config.embedding_dim,
        )

    def test_forward_output_shape(self, model, small_config):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        sequence_length = 10

        # Create dummy input (batch of token IDs)
        input_tokens = np.random.randint(
            0, small_config.vocab_size, size=(batch_size, sequence_length)
        )

        # Forward pass
        logits = model.forward(input_tokens)

        # Output should be (batch, sequence, vocab_size)
        expected_shape = (batch_size, sequence_length, small_config.vocab_size)
        assert logits.shape == expected_shape

    def test_forward_different_sequence_lengths(self, model, small_config):
        """Test forward pass with different sequence lengths."""
        batch_size = 2

        for seq_len in [1, 5, 10, small_config.max_sequence_length]:
            input_tokens = np.random.randint(
                0, small_config.vocab_size, size=(batch_size, seq_len)
            )
            logits = model.forward(input_tokens)

            assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_forward_produces_valid_logits(self, model, small_config):
        """Test that forward pass produces finite logits."""
        input_tokens = np.array([[1, 2, 3, 4, 5]])

        logits = model.forward(input_tokens)

        assert np.all(np.isfinite(logits))

    def test_backward_computes_gradients(self, model, small_config):
        """Test that backward pass computes gradients."""
        input_tokens = np.array([[1, 2, 3, 4, 5]])

        # Forward pass
        logits = model.forward(input_tokens)

        # Create gradient (pretend it's from loss)
        grad_logits = np.random.randn(*logits.shape) * 0.01

        # Backward pass
        gradients = model.backward(grad_logits)

        # Check that gradients were computed
        assert "token_embedding.weight" in gradients
        assert "output_projection.weight" in gradients
        assert "output_projection.bias" in gradients

        # Check gradient shapes
        assert (
            gradients["token_embedding.weight"].shape
            == model.token_embedding.weight.shape
        )
        assert (
            gradients["output_projection.weight"].shape
            == model.output_projection.weight.shape
        )

    def test_get_parameters(self, model):
        """Test that get_parameters returns all model parameters."""
        params = model.get_parameters()

        # Should have token embedding, positional encoding, transformer layers, output
        assert "token_embedding.weight" in params
        assert "output_projection.weight" in params
        assert "output_projection.bias" in params

        # Should have transformer layer parameters
        assert any("transformer" in name for name in params.keys())

    def test_forward_backward_consistency(self, model, small_config):
        """Test that forward and backward are consistent."""
        np.random.seed(42)
        input_tokens = np.random.randint(0, small_config.vocab_size, size=(2, 8))

        # Two forward passes should give same result
        logits1 = model.forward(input_tokens)
        logits2 = model.forward(input_tokens)

        np.testing.assert_array_equal(logits1, logits2)


class TestCrossEntropyLoss:
    """Test cross-entropy loss function."""

    def test_loss_shape(self):
        """Test that loss returns a scalar."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        logits = np.random.randn(batch_size, seq_len, vocab_size)
        targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

        loss = cross_entropy_loss(logits, targets)

        assert np.isscalar(loss) or loss.shape == ()

    def test_loss_positive(self):
        """Test that loss is non-negative."""
        logits = np.random.randn(2, 5, 50)
        targets = np.random.randint(0, 50, size=(2, 5))

        loss = cross_entropy_loss(logits, targets)

        assert loss >= 0

    def test_loss_correct_prediction(self):
        """Test that loss is low when predictions are correct."""
        vocab_size = 10

        # Create logits where the correct class has very high value
        logits = np.full((1, 1, vocab_size), -10.0)
        targets = np.array([[5]])  # Target is class 5
        logits[0, 0, 5] = 10.0  # Make class 5 have high logit

        loss = cross_entropy_loss(logits, targets)

        # Loss should be very small (close to 0)
        assert loss < 0.1

    def test_loss_wrong_prediction(self):
        """Test that loss is high when predictions are wrong."""
        vocab_size = 10

        # Create logits where wrong class has high value
        logits = np.full((1, 1, vocab_size), -10.0)
        targets = np.array([[5]])  # Target is class 5
        logits[0, 0, 0] = 10.0  # Make class 0 have high logit (wrong!)

        loss = cross_entropy_loss(logits, targets)

        # Loss should be high
        assert loss > 10.0

    def test_loss_with_padding(self):
        """Test that loss can handle padding tokens (ignored)."""
        vocab_size = 100
        pad_token = 0

        logits = np.random.randn(2, 5, vocab_size)
        targets = np.array(
            [
                [1, 2, 3, 0, 0],  # Last 2 are padding
                [4, 5, 0, 0, 0],  # Last 3 are padding
            ]
        )

        loss = cross_entropy_loss(logits, targets, ignore_index=pad_token)

        assert np.isfinite(loss)
        assert loss >= 0


class TestCrossEntropyLossBackward:
    """Test cross-entropy loss gradient."""

    def test_backward_shape(self):
        """Test gradient has same shape as logits."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        logits = np.random.randn(batch_size, seq_len, vocab_size)
        targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

        grad = cross_entropy_loss_backward(logits, targets)

        assert grad.shape == logits.shape

    def test_backward_gradient_direction(self):
        """Test that gradient points toward correct class."""
        vocab_size = 5

        # All equal logits
        logits = np.zeros((1, 1, vocab_size))
        targets = np.array([[2]])  # Target is class 2

        grad = cross_entropy_loss_backward(logits, targets)

        # Gradient for correct class should be negative (we want to increase it)
        # Gradient for other classes should be positive (we want to decrease them)
        assert grad[0, 0, 2] < 0  # Target class
        for i in range(vocab_size):
            if i != 2:
                assert grad[0, 0, i] > 0

    def test_backward_gradient_finite(self):
        """Test that gradients are finite."""
        logits = np.random.randn(2, 5, 50)
        targets = np.random.randint(0, 50, size=(2, 5))

        grad = cross_entropy_loss_backward(logits, targets)

        assert np.all(np.isfinite(grad))


class TestTextGeneration:
    """Test text generation functionality."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for generation testing."""
        config = GPTConfig(
            vocab_size=50,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_hidden_dim=64,
            max_sequence_length=32,
        )
        return GPTModel(config)

    def test_generate_output_shape(self, small_model):
        """Test that generate produces correct number of tokens."""
        prompt = np.array([[1, 2, 3]])  # Starting tokens
        max_new_tokens = 5

        generated = small_model.generate(prompt, max_new_tokens=max_new_tokens)

        # Should have original + new tokens
        expected_length = prompt.shape[1] + max_new_tokens
        assert generated.shape == (1, expected_length)

    def test_generate_preserves_prompt(self, small_model):
        """Test that generation preserves the original prompt."""
        prompt = np.array([[1, 2, 3]])

        generated = small_model.generate(prompt, max_new_tokens=3)

        # First tokens should be the prompt
        np.testing.assert_array_equal(generated[0, :3], prompt[0])

    def test_generate_valid_tokens(self, small_model):
        """Test that generated tokens are valid vocabulary indices."""
        prompt = np.array([[1, 2, 3]])

        generated = small_model.generate(prompt, max_new_tokens=10)

        # All tokens should be in valid range
        assert np.all(generated >= 0)
        assert np.all(generated < small_model.config.vocab_size)

    def test_generate_with_temperature(self, small_model):
        """Test generation with different temperatures."""
        prompt = np.array([[1, 2, 3]])

        # Low temperature (more deterministic)
        gen_low = small_model.generate(prompt, max_new_tokens=5, temperature=0.1)

        # High temperature (more random)
        gen_high = small_model.generate(prompt, max_new_tokens=5, temperature=2.0)

        # Both should be valid
        assert gen_low.shape == gen_high.shape
        assert np.all(gen_low >= 0)
        assert np.all(gen_high >= 0)

    def test_generate_deterministic_with_seed(self, small_model):
        """Test that generation is deterministic with same seed."""
        prompt = np.array([[1, 2, 3]])

        np.random.seed(42)
        gen1 = small_model.generate(prompt, max_new_tokens=5)

        np.random.seed(42)
        gen2 = small_model.generate(prompt, max_new_tokens=5)

        np.testing.assert_array_equal(gen1, gen2)
