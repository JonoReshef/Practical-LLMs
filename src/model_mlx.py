"""
MLX-Accelerated GPT Model Implementation

This module provides a GPU-accelerated implementation of the GPT model using
Apple's MLX framework. It's designed for Apple Silicon (M1/M2/M3/M4) and
leverages unified memory and the GPU for significant speedups over NumPy.

This serves as a comparison to the educational NumPy implementation:
- NumPy version: Shows every computation step, great for learning
- MLX version: Shows how frameworks abstract away complexity for speed

Key differences from NumPy version:
- Automatic differentiation (no manual backward pass)
- GPU-accelerated matrix operations
- Lazy evaluation with explicit mx.eval() calls
- Built-in optimizer implementations

Usage:
    from src.model_mlx import GPTModelMLX, GPTConfigMLX, create_model_mlx

    model = create_model_mlx(vocab_size=2000, size="medium")
    # Training uses mlx.nn.value_and_grad for automatic differentiation

Reference:
    - MLX Documentation: https://ml-explore.github.io/mlx/
    - Apple Silicon optimization for ML workloads
"""

from dataclasses import dataclass
from typing import Optional, Tuple

# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None


def check_mlx_available():
    """Check if MLX is available and raise helpful error if not."""
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX is not installed. To use Mac-accelerated training:\n"
            "  pip install mlx\n\n"
            "Note: MLX only works on Apple Silicon Macs (M1/M2/M3/M4)."
        )


@dataclass
class GPTConfigMLX:
    """
    Configuration for MLX GPT model.

    Provides preset sizes for easy experimentation:
    - tiny: ~100K params, fast iteration
    - small: ~500K params, quick training
    - medium: ~2M params, balanced
    - large: ~8M params, best quality (slower)
    """

    vocab_size: int = 2000
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_hidden_dim: int = 512
    max_sequence_length: int = 128
    dropout: float = 0.1

    @classmethod
    def from_size(cls, vocab_size: int, size: str = "medium") -> "GPTConfigMLX":
        """
        Create config from preset size.

        Args:
            vocab_size: Vocabulary size from tokenizer
            size: One of "tiny", "small", "medium", "large"

        Returns:
            GPTConfigMLX with appropriate hyperparameters
        """
        presets = {
            "tiny": {
                "embedding_dim": 64,
                "num_heads": 2,
                "num_layers": 2,
                "ffn_hidden_dim": 128,
                "max_sequence_length": 64,
            },
            "small": {
                "embedding_dim": 128,
                "num_heads": 4,
                "num_layers": 4,
                "ffn_hidden_dim": 512,
                "max_sequence_length": 128,
            },
            "medium": {
                "embedding_dim": 256,
                "num_heads": 8,
                "num_layers": 6,
                "ffn_hidden_dim": 1024,
                "max_sequence_length": 256,
            },
            "large": {
                "embedding_dim": 512,
                "num_heads": 8,
                "num_layers": 8,
                "ffn_hidden_dim": 2048,
                "max_sequence_length": 512,
            },
        }

        if size not in presets:
            raise ValueError(
                f"Unknown size '{size}'. Choose from: {list(presets.keys())}"
            )

        return cls(vocab_size=vocab_size, **presets[size])


# =============================================================================
# MLX-SPECIFIC CLASSES (only defined when MLX is available)
# =============================================================================

if MLX_AVAILABLE:

    class FeedForwardMLX(nn.Module):
        """Feed-forward network using MLX layers."""

        def __init__(
            self, embedding_dim: int, ffn_hidden_dim: int, dropout: float = 0.1
        ):
            super().__init__()
            self.linear1 = nn.Linear(embedding_dim, ffn_hidden_dim)
            self.linear2 = nn.Linear(ffn_hidden_dim, embedding_dim)
            self.dropout = nn.Dropout(dropout)
            self.gelu = nn.GELU()

        def __call__(self, x):
            x = self.linear1(x)
            x = self.gelu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            x = self.dropout(x)
            return x

    class TransformerBlockMLX(nn.Module):
        """Single transformer block using MLX."""

        def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            ffn_hidden_dim: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.attention = nn.MultiHeadAttention(
                dims=embedding_dim,
                num_heads=num_heads,
            )
            self.ffn = FeedForwardMLX(embedding_dim, ffn_hidden_dim, dropout)
            self.ln1 = nn.LayerNorm(embedding_dim)
            self.ln2 = nn.LayerNorm(embedding_dim)
            self.dropout = nn.Dropout(dropout)

        def __call__(self, x, mask=None):
            # Pre-norm architecture (more stable for training)
            # Attention with residual
            normed = self.ln1(x)
            # MLX MultiHeadAttention returns only output, not (output, weights)
            attn_out = self.attention(normed, normed, normed, mask=mask)
            x = x + self.dropout(attn_out)

            # FFN with residual
            normed = self.ln2(x)
            ffn_out = self.ffn(normed)
            x = x + ffn_out

            return x

    class GPTModelMLX(nn.Module):
        """
        GPT Language Model using Apple MLX.

        This implements the same architecture as the NumPy GPTModel but uses
        MLX for GPU acceleration on Apple Silicon.

        Architecture:
            Token IDs -> Embedding + Positional -> Transformer Blocks -> LM Head -> Logits
        """

        def __init__(self, config: GPTConfigMLX):
            super().__init__()
            self.config = config

            # Token embedding
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

            # Positional encoding (learnable in this version for simplicity)
            self.position_embedding = nn.Embedding(
                config.max_sequence_length, config.embedding_dim
            )

            # Transformer blocks
            self.blocks = [
                TransformerBlockMLX(
                    config.embedding_dim,
                    config.num_heads,
                    config.ffn_hidden_dim,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]

            # Final layer norm
            self.ln_f = nn.LayerNorm(config.embedding_dim)

            # Language model head (project to vocabulary)
            self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

            # Dropout
            self.dropout = nn.Dropout(config.dropout)

        def __call__(self, input_ids):
            """
            Forward pass.

            Args:
                input_ids: Token IDs, shape (batch_size, sequence_length)

            Returns:
                logits: Shape (batch_size, sequence_length, vocab_size)
            """
            batch_size, seq_len = input_ids.shape

            # Create position indices
            positions = mx.arange(seq_len)

            # Embeddings
            token_emb = self.token_embedding(input_ids)
            pos_emb = self.position_embedding(positions)
            x = self.dropout(token_emb + pos_emb)

            # Create causal mask
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(x.dtype)

            # Transformer blocks
            for block in self.blocks:
                x = block(x, mask=mask)

            # Final norm and projection
            x = self.ln_f(x)
            logits = self.lm_head(x)

            return logits

        def generate(
            self,
            input_ids,
            max_new_tokens: int = 50,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
        ):
            """
            Generate text autoregressively.

            Args:
                input_ids: Starting token IDs, shape (batch_size, seq_len)
                max_new_tokens: Number of new tokens to generate
                temperature: Sampling temperature (higher = more random)
                top_k: If set, only sample from top-k tokens

            Returns:
                Generated token IDs including the prompt
            """
            for _ in range(max_new_tokens):
                # Truncate to max sequence length
                idx_cond = input_ids[:, -self.config.max_sequence_length :]

                # Forward pass
                logits = self(idx_cond)

                # Get logits for the last position
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    # Get top-k values and indices
                    top_values = mx.topk(logits, k=top_k)
                    # Create mask for values below top-k threshold
                    threshold = top_values[:, -1:]
                    logits = mx.where(logits < threshold, float("-inf"), logits)

                # Convert to probabilities
                probs = mx.softmax(logits, axis=-1)

                # Sample
                next_token = mx.random.categorical(probs)
                next_token = next_token.reshape(-1, 1)

                # Append
                input_ids = mx.concatenate([input_ids, next_token], axis=1)

                # Evaluate to avoid graph buildup
                mx.eval(input_ids)

            return input_ids

    def count_parameters_mlx(model) -> int:
        """Count total parameters in MLX model."""
        total = 0
        for name, param in model.parameters().items():
            if isinstance(param, mx.array):
                total += param.size
            elif isinstance(param, dict):
                for p in param.values():
                    if isinstance(p, mx.array):
                        total += p.size
            elif isinstance(param, list):
                for item in param:
                    if isinstance(item, dict):
                        for p in item.values():
                            if isinstance(p, mx.array):
                                total += p.size
        return total

    def create_model_mlx(
        vocab_size: int, size: str = "medium"
    ) -> Tuple["GPTModelMLX", GPTConfigMLX]:
        """
        Create an MLX GPT model with preset size.

        Args:
            vocab_size: Vocabulary size from tokenizer
            size: One of "tiny", "small", "medium", "large"

        Returns:
            Tuple of (model, config)
        """
        check_mlx_available()

        config = GPTConfigMLX.from_size(vocab_size, size)
        model = GPTModelMLX(config)

        # Initialize parameters
        mx.eval(model.parameters())

        return model, config

    def cross_entropy_loss_mlx(logits, targets, ignore_index: int = -100):
        """
        Compute cross-entropy loss for language modeling.

        Args:
            logits: Model output, shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs, shape (batch_size, seq_len)
            ignore_index: Token ID to ignore in loss computation (e.g., padding)

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for cross-entropy
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Create mask for valid tokens
        mask = targets_flat != ignore_index

        # Compute log softmax manually: log_softmax(x) = x - logsumexp(x)
        log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)

        # Gather log probabilities for target tokens
        # Use advanced indexing
        batch_indices = mx.arange(targets_flat.size)
        target_log_probs = log_probs[batch_indices, targets_flat]

        # Apply mask and compute mean
        masked_log_probs = mx.where(
            mask, target_log_probs, mx.zeros_like(target_log_probs)
        )
        num_valid = mx.sum(mask.astype(mx.float32))

        # Avoid division by zero
        loss = -mx.sum(masked_log_probs) / mx.maximum(num_valid, mx.array(1.0))

        return loss

else:
    # Stubs for when MLX is not available
    FeedForwardMLX = None
    TransformerBlockMLX = None
    GPTModelMLX = None

    def count_parameters_mlx(model) -> int:
        raise ImportError("MLX not available")

    def create_model_mlx(vocab_size: int, size: str = "medium"):
        check_mlx_available()  # Will raise ImportError

    def cross_entropy_loss_mlx(logits, targets, ignore_index: int = -100):
        raise ImportError("MLX not available")


# =============================================================================
# EDUCATIONAL DEMO
# Run with: python -m src.model_mlx
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MLX GPT MODEL DEMO - GPU-Accelerated Language Model")
    print("=" * 70)
    print()

    if not MLX_AVAILABLE:
        print("ERROR: MLX is not installed.")
        print("To install: pip install mlx")
        print("Note: MLX only works on Apple Silicon Macs (M1/M2/M3/M4)")
        exit(1)

    print("MLX is available! Running on Apple Silicon GPU.")
    print()

    # -------------------------------------------------------------------------
    # MODEL SIZE COMPARISON
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("1. MODEL SIZE PRESETS")
    print("-" * 70)
    print()
    print("MLX GPT supports different sizes for experimentation:")
    print()

    for size in ["tiny", "small", "medium", "large"]:
        config = GPTConfigMLX.from_size(vocab_size=2000, size=size)
        print(f"  {size.upper()}")
        print(f"    Embedding dim: {config.embedding_dim}")
        print(f"    Attention heads: {config.num_heads}")
        print(f"    Transformer layers: {config.num_layers}")
        print(f"    FFN hidden dim: {config.ffn_hidden_dim}")
        print(f"    Max sequence length: {config.max_sequence_length}")
        print()

    # -------------------------------------------------------------------------
    # CREATE AND TEST MODEL
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("2. MODEL CREATION AND FORWARD PASS")
    print("-" * 70)
    print()

    model, config = create_model_mlx(vocab_size=2000, size="small")
    num_params = count_parameters_mlx(model)

    print(f"Created 'small' model with {num_params:,} parameters")
    print()

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    import time

    start = time.time()
    logits = model(input_ids)
    mx.eval(logits)  # Force computation
    elapsed = time.time() - start

    print("Forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Time: {elapsed * 1000:.2f}ms")
    print()

    # -------------------------------------------------------------------------
    # AUTOMATIC DIFFERENTIATION
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("3. AUTOMATIC DIFFERENTIATION (No manual backward!)")
    print("-" * 70)
    print()
    print("Unlike NumPy version, MLX computes gradients automatically:")
    print()
    print("  # NumPy (manual):")
    print("  grad_logits = cross_entropy_loss_backward(logits, targets)")
    print("  gradients = model.backward(grad_logits)  # 500+ lines of code")
    print()
    print("  # MLX (automatic):")
    print("  loss_and_grad = nn.value_and_grad(model, loss_fn)")
    print("  loss, grads = loss_and_grad(model, inputs, targets)")
    print()

    # Demo gradient computation
    targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    def loss_fn(model, x, y):
        logits = model(x)
        return cross_entropy_loss_mlx(logits, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    start = time.time()
    loss, grads = loss_and_grad(model, input_ids, targets)
    mx.eval(loss, grads)
    elapsed = time.time() - start

    print("Loss + Gradient computation:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Time: {elapsed * 1000:.2f}ms")
    print(f"  Gradient keys: {len(grads)} parameter groups")
    print()

    # -------------------------------------------------------------------------
    # TEXT GENERATION
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("4. TEXT GENERATION")
    print("-" * 70)
    print()

    prompt = mx.array([[1, 2, 3, 4, 5]])  # Dummy prompt tokens

    start = time.time()
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    elapsed = time.time() - start

    print(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    print(f"  Time: {elapsed * 1000:.2f}ms")
    print(f"  Tokens/sec: {20 / elapsed:.1f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("- MLX provides GPU acceleration on Apple Silicon")
    print("- Same architecture as NumPy version, but ~10-100x faster")
    print("- Automatic differentiation replaces manual backward pass")
    print("- Use size presets: tiny, small, medium, large")
    print()
    print("Run 'python run_demo.py mac-accel' to train with MLX!")
