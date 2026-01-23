"""
Tests for LoRA (Low-Rank Adaptation) Module

Tests the LoRA implementation including:
- Low-rank decomposition
- Forward pass computation
- Gradient computation
- Merging LoRA weights into base model
- Applying LoRA to GPT model
"""

import numpy as np
import pytest

from src.lora import (
    LoRALinear,
    apply_lora_to_model,
    get_lora_gradients,
    get_lora_parameters,
    merge_lora_weights,
)
from src.model import GPTConfig, GPTModel


class TestLoRALinear:
    """Test the LoRA-enhanced linear layer."""

    def test_lora_output_shape(self):
        """Test that LoRA linear produces correct output shape."""
        input_dim = 64
        output_dim = 64
        batch_size = 4
        seq_len = 10

        lora = LoRALinear(
            input_features=input_dim, output_features=output_dim, rank=4, alpha=8
        )

        input_tensor = np.random.randn(batch_size, seq_len, input_dim)
        output = lora.forward(input_tensor)

        assert output.shape == (batch_size, seq_len, output_dim)

    def test_lora_low_rank_shapes(self):
        """Test that LoRA matrices have correct low-rank shapes."""
        input_dim = 64
        output_dim = 64
        rank = 4

        lora = LoRALinear(
            input_features=input_dim, output_features=output_dim, rank=rank, alpha=8
        )

        # A: (rank, input_dim)
        assert lora.lora_A.shape == (rank, input_dim)
        # B: (output_dim, rank)
        assert lora.lora_B.shape == (output_dim, rank)

    def test_lora_initialized_correctly(self):
        """Test that LoRA A is random and B is zero."""
        input_dim = 64
        output_dim = 64
        rank = 4

        lora = LoRALinear(
            input_features=input_dim, output_features=output_dim, rank=rank, alpha=8
        )

        # A should be non-zero (random)
        assert np.any(lora.lora_A != 0)

        # B should be all zeros initially
        assert np.all(lora.lora_B == 0)

    def test_lora_forward_initially_matches_base(self):
        """Test that with B=0, LoRA output equals base linear output."""
        input_dim = 64
        output_dim = 64
        rank = 4

        lora = LoRALinear(
            input_features=input_dim, output_features=output_dim, rank=rank, alpha=8
        )

        input_tensor = np.random.randn(2, 5, input_dim)

        # Forward through LoRA layer
        lora_output = lora.forward(input_tensor)

        # Forward through just base weights
        base_output = np.matmul(input_tensor, lora.base_weights.T)
        if lora.base_bias is not None:
            base_output += lora.base_bias

        # Should be equal since B=0
        np.testing.assert_allclose(lora_output, base_output, rtol=1e-5)

    def test_lora_backward_computes_gradients(self):
        """Test that backward computes gradients for A and B."""
        input_dim = 32
        output_dim = 32
        rank = 4

        lora = LoRALinear(
            input_features=input_dim, output_features=output_dim, rank=rank, alpha=8
        )

        # Make B non-zero so we have signal
        lora.lora_B = np.random.randn(*lora.lora_B.shape) * 0.1

        input_tensor = np.random.randn(2, 5, input_dim)
        _ = lora.forward(input_tensor)

        upstream_gradient = np.random.randn(2, 5, output_dim)
        _ = lora.backward(upstream_gradient)

        gradients = lora.get_lora_gradients()

        assert "lora_A" in gradients
        assert "lora_B" in gradients
        assert gradients["lora_A"].shape == lora.lora_A.shape
        assert gradients["lora_B"].shape == lora.lora_B.shape

    def test_lora_scaling(self):
        """Test that scaling factor alpha/rank is applied correctly."""
        input_dim = 64
        output_dim = 64
        rank = 4
        alpha = 8  # scaling = 8/4 = 2

        lora = LoRALinear(
            input_features=input_dim, output_features=output_dim, rank=rank, alpha=alpha
        )

        expected_scaling = alpha / rank
        assert lora.scaling == expected_scaling


class TestLoRAApplyToModel:
    """Test applying LoRA to a GPT model."""

    @pytest.fixture
    def small_model(self):
        """Create a small GPT model for testing."""
        config = GPTConfig(
            vocab_size=100,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_hidden_dim=64,
            max_sequence_length=16,
        )
        return GPTModel(config)

    def test_apply_lora_returns_model(self, small_model):
        """Test that apply_lora returns a modified model."""
        lora_model = apply_lora_to_model(
            small_model, rank=4, alpha=8, target_modules=["query", "value"]
        )

        assert lora_model is not None

    def test_apply_lora_forward_works(self, small_model):
        """Test that LoRA model can run forward pass."""
        lora_model = apply_lora_to_model(
            small_model, rank=4, alpha=8, target_modules=["query", "value"]
        )

        input_tokens = np.array([[1, 2, 3, 4, 5]])
        logits = lora_model.forward(input_tokens)

        assert logits.shape == (1, 5, small_model.config.vocab_size)

    def test_lora_reduces_trainable_params(self, small_model):
        """Test that LoRA has fewer trainable parameters than full model."""
        base_params = sum(p.size for p in small_model.get_parameters().values())

        lora_model = apply_lora_to_model(
            small_model, rank=4, alpha=8, target_modules=["query", "value"]
        )

        lora_params = get_lora_parameters(lora_model)
        lora_param_count = sum(p.size for p in lora_params.values())

        # LoRA should have much fewer parameters
        assert lora_param_count < base_params


class TestLoRAMerge:
    """Test merging LoRA weights into base model."""

    def test_merge_lora_weights(self):
        """Test that merged weights produce same output as LoRA layer."""
        input_dim = 64
        output_dim = 64
        rank = 4

        lora = LoRALinear(
            input_features=input_dim, output_features=output_dim, rank=rank, alpha=8
        )

        # Set some non-zero LoRA weights
        lora.lora_A = np.random.randn(*lora.lora_A.shape) * 0.1
        lora.lora_B = np.random.randn(*lora.lora_B.shape) * 0.1

        input_tensor = np.random.randn(2, 5, input_dim)

        # Get output with LoRA
        lora_output = lora.forward(input_tensor)

        # Merge weights
        merged_weights = merge_lora_weights(
            lora.base_weights, lora.lora_A, lora.lora_B, lora.scaling
        )

        # Get output with merged weights
        merged_output = np.matmul(input_tensor, merged_weights.T)
        if lora.base_bias is not None:
            merged_output += lora.base_bias

        # Should be equal
        np.testing.assert_allclose(lora_output, merged_output, rtol=1e-5)


class TestLoRAGradients:
    """Test LoRA gradient collection."""

    @pytest.fixture
    def small_lora_model(self):
        """Create a small LoRA-enhanced model."""
        config = GPTConfig(
            vocab_size=50,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_hidden_dim=64,
            max_sequence_length=16,
        )
        model = GPTModel(config)
        return apply_lora_to_model(
            model, rank=4, alpha=8, target_modules=["query", "value"]
        )

    def test_get_lora_gradients(self, small_lora_model):
        """Test that LoRA gradients can be retrieved."""
        input_tokens = np.array([[1, 2, 3, 4, 5]])

        # Forward pass
        logits = small_lora_model.forward(input_tokens)

        # Create dummy gradient
        grad_logits = np.random.randn(*logits.shape) * 0.01

        # Backward pass
        _ = small_lora_model.backward(grad_logits)

        # Get LoRA gradients
        lora_grads = get_lora_gradients(small_lora_model)

        assert len(lora_grads) > 0

        # All gradients should have 'lora' in the name
        for name in lora_grads.keys():
            assert "lora" in name.lower()
