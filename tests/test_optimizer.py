"""
Tests for optimizer module.

Tests cover:
- AdamW optimizer implementation
- Weight decay (L2 regularization)
- Gradient clipping

Following TDD: these tests are written BEFORE the implementation.
"""

import numpy as np


class TestAdamWOptimizer:
    """
    Test suite for AdamW optimizer.

    AdamW is the optimizer used for training most modern LLMs.
    It combines Adam (adaptive learning rates) with decoupled weight decay.

    Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
    """

    def test_adamw_updates_parameters(self):
        """Optimizer should update parameters when step is called."""
        from src.optimizer import AdamW

        # Create simple parameter and gradient
        params = {"weight": np.array([1.0, 2.0, 3.0])}
        grads = {"weight": np.array([0.1, 0.1, 0.1])}

        optimizer = AdamW(learning_rate=0.01)
        optimizer.initialize(params)

        original_params = params["weight"].copy()
        optimizer.step(grads)

        # Parameters should have changed
        assert not np.allclose(params["weight"], original_params), (
            "Parameters should be updated after optimizer step"
        )

    def test_adamw_reduces_loss_direction(self):
        """Optimizer should move parameters in negative gradient direction."""
        from src.optimizer import AdamW

        # Simple case: parameter = 1.0, gradient = positive
        params = {"weight": np.array([1.0])}
        grads = {"weight": np.array([1.0])}  # Positive gradient

        optimizer = AdamW(learning_rate=0.1)
        optimizer.initialize(params)
        optimizer.step(grads)

        # With positive gradient, parameter should decrease
        assert params["weight"][0] < 1.0, (
            "Parameter should decrease with positive gradient"
        )

    def test_adamw_weight_decay(self):
        """Weight decay should reduce parameter magnitudes."""
        from src.optimizer import AdamW

        # Large initial parameter
        params = {"weight": np.array([10.0, 10.0])}
        grads = {"weight": np.array([0.0, 0.0])}  # Zero gradient

        optimizer = AdamW(learning_rate=0.01, weight_decay=0.1)
        optimizer.initialize(params)

        original_magnitude = np.linalg.norm(params["weight"])
        optimizer.step(grads)
        new_magnitude = np.linalg.norm(params["weight"])

        # Weight decay should reduce magnitude even with zero gradient
        assert new_magnitude < original_magnitude, (
            "Weight decay should reduce parameter magnitude"
        )

    def test_adamw_momentum_accumulation(self):
        """Adam momentum should accumulate over steps."""
        from src.optimizer import AdamW

        params = {"weight": np.array([0.0])}
        grads = {"weight": np.array([1.0])}  # Constant gradient

        optimizer = AdamW(learning_rate=0.01, beta1=0.9, beta2=0.999)
        optimizer.initialize(params)

        # Multiple steps with same gradient
        for _ in range(10):
            optimizer.step(grads)

        # Should have accumulated momentum
        assert optimizer.momentum["weight"] is not None, "Momentum should be tracked"

    def test_adamw_handles_multiple_parameters(self):
        """Optimizer should handle multiple parameter groups."""
        from src.optimizer import AdamW

        params = {
            "weight1": np.array([1.0, 2.0]),
            "weight2": np.array([3.0, 4.0, 5.0]),
            "bias": np.array([0.1]),
        }
        grads = {
            "weight1": np.array([0.1, 0.1]),
            "weight2": np.array([0.1, 0.1, 0.1]),
            "bias": np.array([0.1]),
        }

        optimizer = AdamW(learning_rate=0.01)
        optimizer.initialize(params)

        # Should not raise any errors
        optimizer.step(grads)

        # All parameters should be updated
        for name in params:
            assert name in optimizer.momentum, (
                f"Parameter {name} should have momentum state"
            )


class TestGradientClipping:
    """
    Test suite for gradient clipping.

    Gradient clipping prevents exploding gradients by scaling them down
    when their norm exceeds a threshold.
    """

    def test_clip_gradient_norm(self):
        """Gradients exceeding max norm should be clipped."""
        from src.optimizer import clip_gradient_norm

        # Create gradient with large norm
        grads = {"weight": np.array([10.0, 10.0])}  # Norm ≈ 14.14
        max_norm = 1.0

        clipped = clip_gradient_norm(grads, max_norm)

        # Check norm is now <= max_norm
        clipped_norm = np.linalg.norm(clipped["weight"])
        assert clipped_norm <= max_norm + 1e-6, (
            f"Clipped gradient norm {clipped_norm} should be <= {max_norm}"
        )

    def test_clip_preserves_direction(self):
        """Clipping should preserve gradient direction."""
        from src.optimizer import clip_gradient_norm

        grads = {"weight": np.array([3.0, 4.0])}  # Norm = 5
        max_norm = 1.0

        original_direction = grads["weight"] / np.linalg.norm(grads["weight"])
        clipped = clip_gradient_norm(grads, max_norm)
        clipped_direction = clipped["weight"] / np.linalg.norm(clipped["weight"])

        assert np.allclose(original_direction, clipped_direction), (
            "Clipping should preserve gradient direction"
        )

    def test_clip_no_change_small_gradient(self):
        """Small gradients should not be changed by clipping."""
        from src.optimizer import clip_gradient_norm

        grads = {"weight": np.array([0.1, 0.1])}  # Norm ≈ 0.14
        max_norm = 1.0

        clipped = clip_gradient_norm(grads, max_norm)

        assert np.allclose(grads["weight"], clipped["weight"]), (
            "Small gradients should not be modified by clipping"
        )


class TestLearningRateScheduler:
    """
    Test suite for learning rate scheduling.
    """

    def test_warmup_schedule(self):
        """Learning rate should increase during warmup."""
        from src.optimizer import get_learning_rate_with_warmup

        base_lr = 0.001
        warmup_steps = 100

        lr_step_0 = get_learning_rate_with_warmup(0, base_lr, warmup_steps)
        lr_step_50 = get_learning_rate_with_warmup(50, base_lr, warmup_steps)
        lr_step_100 = get_learning_rate_with_warmup(100, base_lr, warmup_steps)

        assert lr_step_0 < lr_step_50 < lr_step_100, (
            "Learning rate should increase during warmup"
        )
        assert np.isclose(lr_step_100, base_lr), (
            "Learning rate should reach base_lr at end of warmup"
        )

    def test_cosine_decay(self):
        """Learning rate should decay after warmup with cosine schedule."""
        from src.optimizer import get_learning_rate_with_warmup

        base_lr = 0.001
        warmup_steps = 100
        total_steps = 1000

        lr_at_warmup = get_learning_rate_with_warmup(
            100, base_lr, warmup_steps, total_steps
        )
        lr_mid = get_learning_rate_with_warmup(500, base_lr, warmup_steps, total_steps)
        lr_end = get_learning_rate_with_warmup(1000, base_lr, warmup_steps, total_steps)

        assert lr_at_warmup > lr_mid > lr_end, "Learning rate should decay after warmup"
