"""
GPT Language Model Implementation

This module implements a complete GPT (Generative Pre-trained Transformer) model
from scratch using NumPy. GPT is a decoder-only transformer that generates text
autoregressively (predicting one token at a time).

Architecture Overview:
    Input Token IDs
           |
    [Token Embedding] + [Positional Encoding]
           |
    [Transformer Block] x N
       - LayerNorm
       - Multi-Head Self-Attention (causal)
       - Residual Connection
       - LayerNorm
       - Feed-Forward Network
       - Residual Connection
           |
    [LayerNorm]
           |
    [Linear Projection] -> Vocabulary Logits
           |
    [Softmax] -> Token Probabilities

Reference:
    - "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
    - "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)

Classes:
    GPTConfig: Configuration dataclass for model hyperparameters
    GPTModel: Complete GPT language model

Functions:
    cross_entropy_loss: Compute cross-entropy loss for language modeling
    cross_entropy_loss_backward: Gradient of cross-entropy loss
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.activations import softmax
from src.layers import Embedding, LayerNorm, Linear, PositionalEncoding
from src.transformer import TransformerStack


@dataclass
class GPTConfig:
    """
    Configuration for GPT Model.

    All hyperparameters that define the model architecture are stored here.
    This makes it easy to save/load model configurations and experiment
    with different settings.

    Attributes:
        vocab_size: Size of the token vocabulary
        embedding_dim: Dimension of token embeddings (d_model in papers)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers (blocks)
        ffn_hidden_dim: Hidden dimension of feed-forward network (usually 4x embedding_dim)
        max_sequence_length: Maximum sequence length the model can handle
        dropout_prob: Dropout probability (not used in NumPy implementation)

    Typical configurations:
        - GPT-2 Small: vocab=50257, embed=768, heads=12, layers=12, ffn=3072
        - Our educational model: vocab=2000, embed=128, heads=4, layers=4, ffn=512
    """

    vocab_size: int = 2000
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_hidden_dim: int = 512
    max_sequence_length: int = 128
    dropout_prob: float = 0.0  # Not implemented (would need stochastic behavior)


class GPTModel:
    """
    Complete GPT Language Model.

    This class assembles all components into a working language model:
    1. Token embedding: Converts token IDs to vectors
    2. Positional encoding: Adds position information
    3. Transformer stack: Processes through N transformer blocks
    4. Output projection: Maps back to vocabulary logits

    The model supports:
    - Forward pass: Input tokens -> Vocabulary logits
    - Backward pass: Gradient computation for training
    - Generation: Autoregressive text generation

    Example usage:
        config = GPTConfig(vocab_size=1000, embedding_dim=64)
        model = GPTModel(config)

        # Training
        tokens = np.array([[1, 2, 3, 4, 5]])  # Shape: (batch, seq_len)
        logits = model.forward(tokens)        # Shape: (batch, seq_len, vocab_size)

        # Generation
        prompt = np.array([[1, 2]])
        generated = model.generate(prompt, max_new_tokens=10)

    Attributes:
        config: Model configuration
        token_embedding: Token ID -> vector embedding
        positional_encoding: Position -> encoding vector
        transformer_stack: Stack of transformer blocks
        final_layer_norm: LayerNorm before output projection
        output_projection: Projects back to vocabulary size
    """

    def __init__(self, config: GPTConfig):
        """
        Initialize GPT model with given configuration.

        Args:
            config: GPTConfig with model hyperparameters
        """
        self.config = config

        # Token embedding: maps token IDs to vectors
        # Shape: (vocab_size, embedding_dim)
        self.token_embedding = Embedding(
            vocabulary_size=config.vocab_size, embedding_dimension=config.embedding_dim
        )

        # Positional encoding: adds position information
        # Shape: (max_seq_len, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            max_sequence_length=config.max_sequence_length,
            embedding_dimension=config.embedding_dim,
        )

        # Transformer stack: N layers of transformer blocks
        self.transformer_stack = TransformerStack(
            num_layers=config.num_layers,
            embedding_dimension=config.embedding_dim,
            num_heads=config.num_heads,
            ffn_hidden_dimension=config.ffn_hidden_dim,
        )

        # Final layer norm (Pre-LN style: applied after transformer stack)
        self.final_layer_norm = LayerNorm(config.embedding_dim)

        # Output projection: maps embeddings back to vocabulary
        # This is the "language model head"
        # Note: In some implementations, this shares weights with token embedding
        # We keep them separate for clarity
        self.output_projection = Linear(
            input_features=config.embedding_dim, output_features=config.vocab_size
        )

        # Cache for backward pass
        self._cache = {}

    def forward(self, input_tokens: np.ndarray) -> np.ndarray:
        """
        Forward pass: Convert input tokens to vocabulary logits.

        This is the main computation path:
        tokens -> embeddings -> transformer blocks -> logits

        Args:
            input_tokens: Integer token IDs, shape (batch_size, sequence_length)

        Returns:
            Logits over vocabulary, shape (batch_size, sequence_length, vocab_size)
            These are unnormalized log-probabilities for each position.
        """
        batch_size, sequence_length = input_tokens.shape

        # Step 1: Token embedding
        # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        token_embeddings = self.token_embedding.forward(input_tokens)

        # Step 2: Add positional encoding
        # (batch, seq_len, embedding_dim) + (seq_len, embedding_dim)
        position_encodings = self.positional_encoding.get_encoding(sequence_length)
        hidden_states = token_embeddings + position_encodings  # Broadcasting

        # Step 3: Pass through transformer stack
        # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)
        hidden_states = self.transformer_stack.forward(hidden_states)

        # Step 4: Final layer normalization
        hidden_states = self.final_layer_norm.forward(hidden_states)

        # Step 5: Project to vocabulary size
        # (batch, seq_len, embedding_dim) -> (batch, seq_len, vocab_size)
        logits = self.output_projection.forward(hidden_states)

        # Cache for backward pass
        self._cache = {
            "input_tokens": input_tokens,
            "token_embeddings": token_embeddings,
            "position_encodings": position_encodings,
            "pre_norm_hidden": hidden_states,
        }

        return logits

    def backward(self, grad_logits: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass: Compute gradients for all parameters.

        This propagates gradients from the loss back through all layers.

        Args:
            grad_logits: Gradient of loss w.r.t. output logits
                        Shape: (batch_size, sequence_length, vocab_size)

        Returns:
            Dictionary mapping parameter names to their gradients
        """
        gradients = {}

        # Backward through output projection
        grad_hidden = self.output_projection.backward(grad_logits)
        # Get gradients from the layer
        output_grads = self.output_projection.get_gradients()
        gradients["output_projection.weight"] = output_grads["weight"]
        gradients["output_projection.bias"] = output_grads["bias"]

        # Backward through final layer norm
        grad_hidden = self.final_layer_norm.backward(grad_hidden)

        # Backward through transformer stack
        grad_hidden, transformer_grads = self.transformer_stack.backward(grad_hidden)

        # Add transformer gradients with proper naming
        for name, grad in transformer_grads.items():
            gradients[f"transformer_stack.{name}"] = grad

        # Backward through positional encoding (no learnable parameters)
        # grad_hidden is gradient w.r.t. (token_embeddings + position_encodings)
        # Gradient w.r.t. token_embeddings is same as grad_hidden

        # Backward through token embedding
        self.token_embedding.backward(grad_hidden)
        token_grads = self.token_embedding.get_gradients()
        gradients["token_embedding.weight"] = token_grads["embedding_table"]

        return gradients

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get all model parameters.

        Returns:
            Dictionary mapping parameter names to parameter arrays
        """
        params = {}

        # Token embedding
        params["token_embedding.weight"] = self.token_embedding.weight

        # Transformer stack parameters
        transformer_params = self.transformer_stack.get_parameters()
        for name, param in transformer_params.items():
            params[f"transformer_stack.{name}"] = param

        # Final layer norm
        params["final_layer_norm.gamma"] = self.final_layer_norm.gamma
        params["final_layer_norm.beta"] = self.final_layer_norm.beta

        # Output projection
        params["output_projection.weight"] = self.output_projection.weight
        params["output_projection.bias"] = self.output_projection.bias

        return params

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """
        Set model parameters from dictionary.

        Args:
            params: Dictionary mapping parameter names to parameter arrays
        """
        if "token_embedding.weight" in params:
            self.token_embedding.weight = params["token_embedding.weight"]

        if "final_layer_norm.gamma" in params:
            self.final_layer_norm.gamma = params["final_layer_norm.gamma"]
        if "final_layer_norm.beta" in params:
            self.final_layer_norm.beta = params["final_layer_norm.beta"]

        if "output_projection.weight" in params:
            self.output_projection.weight = params["output_projection.weight"]
        if "output_projection.bias" in params:
            self.output_projection.bias = params["output_projection.bias"]

        # Set transformer stack parameters
        transformer_params = {}
        prefix = "transformer_stack."
        for name, param in params.items():
            if name.startswith(prefix):
                transformer_params[name[len(prefix) :]] = param

        if transformer_params:
            self.transformer_stack.set_parameters(transformer_params)

    def generate(
        self,
        prompt_tokens: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate text autoregressively.

        Starting from the prompt, generate new tokens one at a time by:
        1. Forward pass to get next token probabilities
        2. Sample from the distribution (with temperature)
        3. Append sampled token
        4. Repeat

        Args:
            prompt_tokens: Starting token IDs, shape (batch_size, prompt_length)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
                        0 = greedy (always pick most likely)
                        1 = sample from true distribution
                        >1 = more exploration
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated token sequence, shape (batch_size, prompt_length + max_new_tokens)
        """
        # Start with the prompt
        generated = prompt_tokens.copy()

        for _ in range(max_new_tokens):
            # Get the context (limited by max sequence length)
            context_length = min(generated.shape[1], self.config.max_sequence_length)
            context = generated[:, -context_length:]

            # Forward pass to get logits
            logits = self.forward(context)

            # Get logits for the last position
            # Shape: (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                # Set all logits outside top-k to -infinity
                top_k_indices = np.argsort(next_token_logits, axis=-1)[:, :-top_k]
                for b in range(next_token_logits.shape[0]):
                    next_token_logits[b, top_k_indices[b]] = float("-inf")

            # Convert to probabilities
            probs = softmax(next_token_logits)

            # Sample next token
            next_token = np.zeros((generated.shape[0], 1), dtype=np.int64)
            for b in range(probs.shape[0]):
                next_token[b, 0] = np.random.choice(self.config.vocab_size, p=probs[b])

            # Append to sequence
            generated = np.concatenate([generated, next_token], axis=1)

        return generated

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        total = 0
        for param in self.get_parameters().values():
            total += param.size
        return total


def cross_entropy_loss(
    logits: np.ndarray, targets: np.ndarray, ignore_index: Optional[int] = None
) -> float:
    """
    Compute cross-entropy loss for language modeling.

    Cross-entropy measures how well the predicted probability distribution
    matches the target distribution (which is a one-hot vector for classification).

    Formula:
        loss = -sum(y_true * log(softmax(logits)))

    For language modeling, we compute this for each position and average.

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size)
        targets: Target token IDs, shape (batch, seq_len)
        ignore_index: If set, ignore positions where target equals this value
                     (typically used for padding tokens)

    Returns:
        Scalar loss value (average cross-entropy per token)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for easier computation
    # (batch * seq_len, vocab_size)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Compute softmax probabilities (numerically stable)
    # Subtract max for numerical stability
    logits_stable = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Get probability of correct class for each position
    num_positions = logits_flat.shape[0]
    correct_class_probs = probs[np.arange(num_positions), targets_flat]

    # Compute log probabilities (clip to avoid log(0))
    log_probs = np.log(np.clip(correct_class_probs, 1e-10, 1.0))

    # Apply ignore mask if specified
    if ignore_index is not None:
        mask = (targets_flat != ignore_index).astype(np.float32)
        log_probs = log_probs * mask
        num_valid = np.sum(mask)
        if num_valid == 0:
            return 0.0
        loss = -np.sum(log_probs) / num_valid
    else:
        loss = -np.mean(log_probs)

    return float(loss)


def cross_entropy_loss_backward(
    logits: np.ndarray, targets: np.ndarray, ignore_index: Optional[int] = None
) -> np.ndarray:
    """
    Compute gradient of cross-entropy loss with respect to logits.

    The gradient of cross-entropy loss w.r.t. logits has a simple form:
        d_loss/d_logits = softmax(logits) - one_hot(targets)

    This means:
    - For the correct class: gradient is (predicted_prob - 1) [negative]
    - For wrong classes: gradient is (predicted_prob - 0) [positive]

    This pushes the model to increase probability of correct class
    and decrease probability of wrong classes.

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size)
        targets: Target token IDs, shape (batch, seq_len)
        ignore_index: If set, zero gradient where target equals this value

    Returns:
        Gradient w.r.t. logits, shape (batch, seq_len, vocab_size)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Compute softmax probabilities
    logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Gradient is: probs - one_hot(targets)
    grad = probs.copy()

    # Create one-hot encoding and subtract
    for b in range(batch_size):
        for t in range(seq_len):
            target_class = targets[b, t]
            grad[b, t, target_class] -= 1.0

    # Apply ignore mask if specified
    if ignore_index is not None:
        mask = (targets != ignore_index).astype(np.float32)
        mask = mask[:, :, np.newaxis]  # Shape: (batch, seq_len, 1)
        grad = grad * mask
        num_valid = np.sum(targets != ignore_index)
        if num_valid > 0:
            grad = grad / num_valid
    else:
        # Average over all positions
        grad = grad / (batch_size * seq_len)

    return grad


# =============================================================================
# EDUCATIONAL DEMO
# Run with: python -m src.model
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GPT MODEL DEMO - The Complete Language Model")
    print("=" * 70)
    print()
    print("This module assembles all components into a complete GPT model")
    print("that can be trained to generate text.")
    print()
    print("Dependencies (everything comes together here):")
    print("  - src.layers (Embedding, PositionalEncoding, Linear, LayerNorm)")
    print("  - src.transformer (TransformerStack)")
    print("  - src.attention (used by transformer)")
    print("  - src.activations (used by transformer)")
    print()

    # -------------------------------------------------------------------------
    # MODEL CONFIGURATION
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("1. MODEL CONFIGURATION")
    print("-" * 70)
    print()
    print("GPTConfig defines all hyperparameters:")
    print()

    config = GPTConfig(
        vocab_size=1000,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_hidden_dim=512,
        max_sequence_length=64,
    )

    print(f"  vocab_size: {config.vocab_size}")
    print("    - Number of unique tokens the model knows")
    print()
    print(f"  embedding_dim: {config.embedding_dim}")
    print("    - Size of vector representing each token")
    print()
    print(f"  num_heads: {config.num_heads}")
    print("    - Parallel attention patterns")
    print()
    print(f"  num_layers: {config.num_layers}")
    print("    - Depth of the model (transformer blocks)")
    print()
    print(f"  ffn_hidden_dim: {config.ffn_hidden_dim}")
    print("    - Width of feed-forward networks")
    print()
    print(f"  max_sequence_length: {config.max_sequence_length}")
    print("    - Maximum context window")
    print()

    # -------------------------------------------------------------------------
    # MODEL CREATION
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("2. MODEL ARCHITECTURE")
    print("-" * 70)
    print()

    model = GPTModel(config)

    print("GPT Architecture:")
    print()
    print("  Token IDs")
    print("      |")
    print("      v")
    print(
        f"  [Embedding]  ({config.vocab_size} tokens -> {config.embedding_dim}d vectors)"
    )
    print("      |")
    print("      v")
    print("  [+ PositionalEncoding]  (adds position information)")
    print("      |")
    print("      v")
    print(f"  [TransformerBlock x {config.num_layers}]")
    print(f"    - {config.num_heads}-head attention")
    print(
        f"    - FFN ({config.embedding_dim} -> {config.ffn_hidden_dim} -> {config.embedding_dim})"
    )
    print("      |")
    print("      v")
    print("  [LayerNorm]  (final normalization)")
    print("      |")
    print("      v")
    print(
        f"  [Output Projection]  ({config.embedding_dim}d -> {config.vocab_size} logits)"
    )
    print("      |")
    print("      v")
    print("  Logits (scores for next token)")
    print()

    # Count parameters
    params = model.get_parameters()
    total_params = sum(p.size for p in params.values())
    print(f"Total parameters: {total_params:,}")
    print()

    # Parameter breakdown
    embedding_params = params["token_embedding.weight"].size
    transformer_params = sum(
        p.size for k, p in params.items() if k.startswith("transformer_stack")
    )
    output_params = (
        params["final_layer_norm.gamma"].size
        + params["final_layer_norm.beta"].size
        + params["output_projection.weight"].size
        + params["output_projection.bias"].size
    )

    print("Parameter breakdown:")
    print(
        f"  Token embedding: {embedding_params:,} ({100 * embedding_params / total_params:.1f}%)"
    )
    print(
        f"  Transformer layers: {transformer_params:,} ({100 * transformer_params / total_params:.1f}%)"
    )
    print(
        f"  Output (norm + projection): {output_params:,} ({100 * output_params / total_params:.1f}%)"
    )
    print()

    # -------------------------------------------------------------------------
    # FORWARD PASS: Prediction
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("3. FORWARD PASS - Making predictions")
    print("-" * 70)
    print()

    # Create sample input
    np.random.seed(42)
    batch_size = 2
    seq_len = 8
    input_tokens = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_tokens.shape} (batch={batch_size}, seq_len={seq_len})")
    print(f"Sample tokens: {input_tokens[0]}")
    print()

    # Forward pass
    logits = model.forward(input_tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"  - batch_size: {logits.shape[0]}")
    print(f"  - seq_len: {logits.shape[1]}")
    print(f"  - vocab_size: {logits.shape[2]}")
    print()
    print("Each position outputs a score for every possible next token.")
    print()

    # Show prediction at last position
    last_logits = logits[0, -1, :]  # First batch, last position
    probs = softmax(last_logits)
    top_5_indices = np.argsort(probs)[-5:][::-1]

    print("Top 5 predicted next tokens (position 7):")
    for idx in top_5_indices:
        print(f"  Token {idx}: {probs[idx]:.4f} ({100 * probs[idx]:.2f}%)")
    print()

    # -------------------------------------------------------------------------
    # LOSS COMPUTATION: Measuring error
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("4. LOSS COMPUTATION - How wrong are we?")
    print("-" * 70)
    print()
    print("Cross-entropy loss measures prediction quality:")
    print("  - Low loss: model assigned high probability to correct token")
    print("  - High loss: model assigned low probability to correct token")
    print()

    # Create target (shifted by 1 for language modeling)
    targets = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

    loss = cross_entropy_loss(logits, targets)
    print(f"Cross-entropy loss: {loss:.4f}")
    print()

    # Expected loss for random model
    random_loss = -np.log(1.0 / config.vocab_size)
    print(f"Random model would have loss: {random_loss:.4f}")
    print(f"  (uniform probability = 1/{config.vocab_size})")
    print()
    print("As training progresses, loss should decrease.")
    print()

    # -------------------------------------------------------------------------
    # BACKWARD PASS: Learning
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("5. BACKWARD PASS - Computing gradients")
    print("-" * 70)
    print()
    print("Backpropagation computes how each parameter should change")
    print("to reduce the loss.")
    print()

    # Compute gradients
    grad_logits = cross_entropy_loss_backward(logits, targets)
    gradients = model.backward(grad_logits)

    print(f"Number of gradient tensors: {len(gradients)}")
    print()
    print("Sample gradients:")
    for name in list(gradients.keys())[:4]:
        grad = gradients[name]
        grad_norm = np.sqrt(np.sum(grad**2))
        print(f"  {name}: shape={grad.shape}, norm={grad_norm:.6f}")
    print("  ...")
    print()

    # -------------------------------------------------------------------------
    # TEXT GENERATION: Creating output
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("6. TEXT GENERATION - Autoregressive decoding")
    print("-" * 70)
    print()
    print("To generate text, we predict one token at a time:")
    print("  1. Input: [token_1, token_2, token_3]")
    print("  2. Model predicts distribution over next token")
    print("  3. Sample from distribution -> token_4")
    print("  4. Input: [token_1, token_2, token_3, token_4]")
    print("  5. Repeat...")
    print()

    # Generate some tokens
    prompt = np.array([[42, 100, 256]])  # Some seed tokens
    print(f"Prompt tokens: {prompt[0]}")
    print()

    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)

    print(f"Generated sequence: {generated[0]}")
    print(
        f"  (original {len(prompt[0])} + {len(generated[0]) - len(prompt[0])} new tokens)"
    )
    print()

    # Show effect of temperature
    print("Effect of temperature on generation:")
    for temp in [0.5, 1.0, 1.5]:
        np.random.seed(123)  # Same seed for comparison
        gen = model.generate(prompt.copy(), max_new_tokens=5, temperature=temp)
        print(f"  temp={temp}: {gen[0]}")
    print()
    print("Lower temperature = more deterministic (picks highest probability)")
    print("Higher temperature = more random (samples more evenly)")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The GPT model:")
    print("  1. Embeds input tokens into vectors")
    print("  2. Adds positional information")
    print("  3. Processes through transformer blocks (attention + FFN)")
    print("  4. Projects to vocabulary size for next-token prediction")
    print()
    print("Training:")
    print("  - Forward pass: predict next tokens")
    print("  - Loss: measure prediction error")
    print("  - Backward pass: compute gradients")
    print("  - Optimizer step: update weights")
    print()
    print("Generation:")
    print("  - Autoregressive: predict one token, append, repeat")
    print("  - Temperature and top-k control randomness")
    print()
    print("Next step: Run 'python -m src.lora' to see parameter-efficient")
    print("           fine-tuning with LoRA adapters.")
