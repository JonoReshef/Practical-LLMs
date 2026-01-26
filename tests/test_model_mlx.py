"""
Tests for MLX-accelerated GPT model implementation.

These tests verify the Apple MLX version of the GPT model functions correctly.
Tests are skipped automatically when MLX is not available (non-Apple Silicon systems).
"""

import pytest

# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Import MLX model components only if available
if MLX_AVAILABLE:
    from src.model_mlx import (
        FeedForwardMLX,
        GPTConfigMLX,
        GPTModelMLX,
        TransformerBlockMLX,
        count_parameters_mlx,
        create_model_mlx,
        cross_entropy_loss_mlx,
    )
else:
    # Import config class which doesn't need MLX
    from src.model_mlx import GPTConfigMLX


# Decorator for MLX-specific tests
mlx_required = pytest.mark.skipif(
    not MLX_AVAILABLE, reason="MLX not available (requires Apple Silicon Mac)"
)


# GPTConfigMLX is a dataclass that doesn't require MLX
class TestGPTConfigMLX:
    """Tests for GPTConfigMLX."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GPTConfigMLX()

        assert config.vocab_size == 2000
        assert config.embedding_dim == 128
        assert config.num_heads == 4
        assert config.num_layers == 4
        assert config.ffn_hidden_dim == 512
        assert config.max_sequence_length == 128
        assert config.dropout == 0.1

    def test_from_size_tiny(self):
        """Test tiny preset configuration."""
        config = GPTConfigMLX.from_size(vocab_size=1000, size="tiny")

        assert config.vocab_size == 1000
        assert config.embedding_dim == 64
        assert config.num_heads == 2
        assert config.num_layers == 2
        assert config.ffn_hidden_dim == 128
        assert config.max_sequence_length == 64

    def test_from_size_small(self):
        """Test small preset configuration."""
        config = GPTConfigMLX.from_size(vocab_size=2000, size="small")

        assert config.vocab_size == 2000
        assert config.embedding_dim == 128
        assert config.num_heads == 4
        assert config.num_layers == 4

    def test_from_size_medium(self):
        """Test medium preset configuration."""
        config = GPTConfigMLX.from_size(vocab_size=2000, size="medium")

        assert config.embedding_dim == 256
        assert config.num_heads == 8
        assert config.num_layers == 6

    def test_from_size_large(self):
        """Test large preset configuration."""
        config = GPTConfigMLX.from_size(vocab_size=2000, size="large")

        assert config.embedding_dim == 512
        assert config.num_heads == 8
        assert config.num_layers == 8

    def test_from_size_invalid(self):
        """Test that invalid size raises error."""
        with pytest.raises(ValueError, match="Unknown size"):
            GPTConfigMLX.from_size(vocab_size=1000, size="xlarge")


@mlx_required
class TestFeedForwardMLX:
    """Tests for feed-forward network."""

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        ffn = FeedForwardMLX(embedding_dim=64, ffn_hidden_dim=256)
        x = mx.random.normal((2, 10, 64))

        output = ffn(x)
        mx.eval(output)

        assert output.shape == (2, 10, 64)

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        ffn = FeedForwardMLX(embedding_dim=32, ffn_hidden_dim=128)

        for batch_size in [1, 4, 16]:
            x = mx.random.normal((batch_size, 8, 32))
            output = ffn(x)
            mx.eval(output)
            assert output.shape == (batch_size, 8, 32)


@mlx_required
class TestTransformerBlockMLX:
    """Tests for transformer block."""

    def test_forward_shape(self):
        """Test output shape preservation."""
        block = TransformerBlockMLX(
            embedding_dim=64,
            num_heads=4,
            ffn_hidden_dim=256,
        )
        x = mx.random.normal((2, 10, 64))

        output = block(x)
        mx.eval(output)

        assert output.shape == (2, 10, 64)

    def test_with_causal_mask(self):
        """Test forward pass with causal mask."""
        block = TransformerBlockMLX(
            embedding_dim=64,
            num_heads=4,
            ffn_hidden_dim=256,
        )
        x = mx.random.normal((2, 16, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(16)

        output = block(x, mask=mask)
        mx.eval(output)

        assert output.shape == (2, 16, 64)


@mlx_required
class TestGPTModelMLX:
    """Tests for the full GPT model."""

    def test_forward_shape(self):
        """Test model output shape."""
        config = GPTConfigMLX(
            vocab_size=100,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_hidden_dim=64,
            max_sequence_length=32,
        )
        model = GPTModelMLX(config)

        input_ids = mx.random.randint(0, 100, (2, 16))
        logits = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 16, 100)

    def test_single_sequence(self):
        """Test with single sequence."""
        config = GPTConfigMLX.from_size(vocab_size=100, size="tiny")
        model = GPTModelMLX(config)

        input_ids = mx.random.randint(0, 100, (1, 8))
        logits = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (1, 8, 100)

    def test_generate_length(self):
        """Test generation produces correct number of tokens."""
        config = GPTConfigMLX(
            vocab_size=100,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_hidden_dim=64,
            max_sequence_length=64,
        )
        model = GPTModelMLX(config)

        prompt = mx.array([[1, 2, 3, 4, 5]])
        max_new_tokens = 10

        generated = model.generate(prompt, max_new_tokens=max_new_tokens)
        mx.eval(generated)

        assert generated.shape == (1, 5 + max_new_tokens)

    def test_generate_with_temperature(self):
        """Test generation with different temperatures."""
        config = GPTConfigMLX.from_size(vocab_size=100, size="tiny")
        model = GPTModelMLX(config)

        prompt = mx.array([[1, 2, 3]])

        # Low temperature (more deterministic)
        gen_low = model.generate(prompt, max_new_tokens=5, temperature=0.1)
        mx.eval(gen_low)

        # High temperature (more random)
        gen_high = model.generate(prompt, max_new_tokens=5, temperature=2.0)
        mx.eval(gen_high)

        assert gen_low.shape == (1, 8)
        assert gen_high.shape == (1, 8)

    def test_generate_with_top_k(self):
        """Test generation with top-k sampling."""
        config = GPTConfigMLX.from_size(vocab_size=100, size="tiny")
        model = GPTModelMLX(config)

        prompt = mx.array([[1, 2, 3]])
        generated = model.generate(prompt, max_new_tokens=5, top_k=10)
        mx.eval(generated)

        assert generated.shape == (1, 8)


@mlx_required
class TestCrossEntropyLossMLX:
    """Tests for cross-entropy loss function."""

    def test_perfect_prediction_low_loss(self):
        """Test that correct predictions have low loss."""
        vocab_size = 10
        batch_size = 2
        seq_len = 5

        # Create targets
        targets = mx.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        # Create logits that strongly favor correct tokens
        logits = mx.zeros((batch_size, seq_len, vocab_size))
        for b in range(batch_size):
            for s in range(seq_len):
                # One-hot-ish with high value for correct token
                target_idx = targets[b, s].item()
                logits = logits.at[b, s, target_idx].add(10.0)

        loss = cross_entropy_loss_mlx(logits, targets)
        mx.eval(loss)

        # Loss should be low for good predictions
        assert loss.item() < 0.1

    def test_random_prediction_higher_loss(self):
        """Test that random logits have higher loss."""
        vocab_size = 100
        logits = mx.random.normal((2, 10, vocab_size))
        targets = mx.random.randint(0, vocab_size, (2, 10))

        loss = cross_entropy_loss_mlx(logits, targets)
        mx.eval(loss)

        # Random predictions should have loss around -log(1/100) = ~4.6
        assert 2.0 < loss.item() < 10.0

    def test_ignore_index(self):
        """Test that ignore_index excludes tokens from loss."""
        vocab_size = 10
        logits = mx.random.normal((2, 5, vocab_size))

        # Targets with some padding tokens
        targets = mx.array([[1, 2, -100, -100, -100], [1, 2, 3, -100, -100]])

        loss = cross_entropy_loss_mlx(logits, targets, ignore_index=-100)
        mx.eval(loss)

        # Loss should be computed only for non-padding tokens
        assert loss.item() > 0


@mlx_required
class TestCreateModelMLX:
    """Tests for model factory function."""

    def test_create_tiny_model(self):
        """Test creating tiny model."""
        model, config = create_model_mlx(vocab_size=1000, size="tiny")

        assert config.embedding_dim == 64
        assert config.vocab_size == 1000

    def test_create_and_forward(self):
        """Test that created model works."""
        model, config = create_model_mlx(vocab_size=500, size="small")

        input_ids = mx.random.randint(0, 500, (1, 16))
        logits = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (1, 16, 500)


@mlx_required
class TestCountParametersMLX:
    """Tests for parameter counting."""

    def test_parameter_count_increases_with_size(self):
        """Test that larger models have more parameters."""
        tiny_model, _ = create_model_mlx(vocab_size=1000, size="tiny")
        small_model, _ = create_model_mlx(vocab_size=1000, size="small")

        tiny_params = count_parameters_mlx(tiny_model)
        small_params = count_parameters_mlx(small_model)

        assert small_params > tiny_params

    def test_parameter_count_positive(self):
        """Test that parameter count is positive."""
        model, _ = create_model_mlx(vocab_size=1000, size="tiny")
        params = count_parameters_mlx(model)

        assert params > 0


@mlx_required
class TestGradientComputation:
    """Tests for automatic differentiation."""

    def test_value_and_grad(self):
        """Test that gradients can be computed."""
        config = GPTConfigMLX(
            vocab_size=50,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            ffn_hidden_dim=32,
            max_sequence_length=16,
        )
        model = GPTModelMLX(config)

        def loss_fn(model, x, y):
            logits = model(x)
            return cross_entropy_loss_mlx(logits, y)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        inputs = mx.random.randint(0, 50, (2, 8))
        targets = mx.random.randint(0, 50, (2, 8))

        loss, grads = loss_and_grad(model, inputs, targets)
        mx.eval(loss, grads)

        # Loss should be a scalar
        assert loss.shape == ()

        # Gradients should exist
        assert len(grads) > 0

    def test_optimizer_update(self):
        """Test that optimizer can update model parameters."""
        import mlx.optimizers as optim

        config = GPTConfigMLX(
            vocab_size=50,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            ffn_hidden_dim=32,
            max_sequence_length=16,
        )
        model = GPTModelMLX(config)
        optimizer = optim.AdamW(learning_rate=1e-3)

        def loss_fn(model, x, y):
            logits = model(x)
            return cross_entropy_loss_mlx(logits, y)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        inputs = mx.random.randint(0, 50, (2, 8))
        targets = mx.random.randint(0, 50, (2, 8))

        # Get initial loss
        initial_loss, grads = loss_and_grad(model, inputs, targets)
        mx.eval(initial_loss)

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Get new loss (should be different)
        new_loss, _ = loss_and_grad(model, inputs, targets)
        mx.eval(new_loss)

        # Parameters should have changed (loss likely decreased)
        assert initial_loss.item() != new_loss.item()


# Tests for parse_data_size helper (imported from run_demo if available)
# These tests don't require MLX
class TestParseDataSize:
    """Tests for data size parsing helper."""

    @pytest.fixture
    def parse_fn(self):
        """Import parse_data_size from run_demo."""
        import sys

        sys.path.insert(0, ".")
        from run_demo import parse_data_size

        return parse_data_size

    def test_fraction(self, parse_fn):
        """Test parsing fraction values."""
        assert parse_fn("0.1", 100000) == 10000
        assert parse_fn("0.5", 100000) == 50000
        assert parse_fn("1.0", 100000) == 100000

    def test_k_suffix(self, parse_fn):
        """Test parsing k suffix."""
        assert parse_fn("100k", 1000000) == 100000
        assert parse_fn("50k", 1000000) == 50000

    def test_m_suffix(self, parse_fn):
        """Test parsing m suffix."""
        assert parse_fn("1m", 10000000) == 1000000
        assert parse_fn("2.5m", 10000000) == 2500000

    def test_absolute_count(self, parse_fn):
        """Test parsing absolute count."""
        assert parse_fn("50000", 100000) == 50000

    def test_invalid_raises(self, parse_fn):
        """Test that invalid input raises error."""
        with pytest.raises(ValueError):
            parse_fn("invalid", 100000)
