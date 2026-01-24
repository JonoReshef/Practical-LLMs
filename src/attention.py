"""
Multi-Head Attention Mechanism

This module implements the core attention mechanism from the Transformer architecture,
including scaled dot-product attention and multi-head attention.

The attention mechanism is what allows transformers to weigh the importance of
different parts of the input when making predictions - the key innovation that
makes modern LLMs possible.

Reference: "Attention Is All You Need" (Vaswani et al., 2017) Section 3.2
           https://arxiv.org/abs/1706.03762

Functions:
    scaled_dot_product_attention: Core attention computation
    attention_backward: Gradient computation for attention

Classes:
    MultiHeadAttention: Multi-head attention layer with projections
"""

from typing import Optional, Tuple

import numpy as np

from src.activations import softmax, softmax_backward
from src.layers import Linear


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Scaled Dot-Product Attention.

    This is the core attention mechanism used in transformers. It computes a
    weighted sum of values, where the weights are determined by the similarity
    between queries and keys.

    Mathematical Formula (from the paper):
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Step-by-step:
        1. Compute attention scores: Q @ K^T (how similar each query is to each key)
        2. Scale by sqrt(d_k) to prevent gradient vanishing in softmax
        3. Apply mask (if provided) to prevent attending to certain positions
        4. Apply softmax to get attention weights (probability distribution)
        5. Multiply by V to get weighted combination of values

    Why scaling by sqrt(d_k)?
        For large d_k, the dot products grow large in magnitude, pushing the
        softmax into regions with extremely small gradients. Scaling keeps
        the variance of the dot products constant regardless of d_k.

    Args:
        query: Query tensor of shape (batch_size, seq_len_q, d_k)
               "What am I looking for?"
        key: Key tensor of shape (batch_size, seq_len_k, d_k)
             "What do I contain?"
        value: Value tensor of shape (batch_size, seq_len_k, d_v)
               "What information do I provide?"
        mask: Optional boolean mask of shape (seq_len_q, seq_len_k) or
              (batch_size, seq_len_q, seq_len_k)
              True = position can be attended to
              False = position should be masked out

    Returns:
        output: Attention output of shape (batch_size, seq_len_q, d_v)
        attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)

    Example:
        >>> Q = np.random.randn(2, 10, 64)  # 2 batches, 10 tokens, 64 dims
        >>> K = np.random.randn(2, 10, 64)
        >>> V = np.random.randn(2, 10, 64)
        >>> output, weights = scaled_dot_product_attention(Q, K, V)
    """
    # Get the dimension of keys for scaling
    d_k = query.shape[-1]

    # Step 1: Compute attention scores (Q @ K^T)
    # Shape: (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
    # This computes dot product between each query and all keys
    attention_scores = np.matmul(query, key.transpose(0, 2, 1))

    # Step 2: Scale by sqrt(d_k)
    # This prevents the dot products from becoming too large for softmax
    scaling_factor = np.sqrt(d_k)
    scaled_attention_scores = attention_scores / scaling_factor

    # Step 3: Apply mask (if provided)
    # Masked positions get a very large negative value so softmax makes them ~0
    if mask is not None:
        # Handle different mask shapes
        if mask.ndim == 2:
            # Shape (seq_q, seq_k) -> broadcast to (batch, seq_q, seq_k)
            mask = mask[np.newaxis, :, :]

        # Where mask is False, fill with large negative number
        # This makes softmax output approximately 0 for masked positions
        masked_attention_scores = np.where(
            mask,
            scaled_attention_scores,
            -1e9,  # Large negative number (will become ~0 after softmax)
        )
    else:
        masked_attention_scores = scaled_attention_scores

    # Step 4: Apply softmax to get attention weights
    # Softmax along last axis: each query position gets a probability distribution
    # over all key positions
    attention_weights = softmax(masked_attention_scores, axis=-1)

    # Step 5: Compute weighted sum of values
    # Shape: (batch, seq_q, seq_k) @ (batch, seq_k, d_v) -> (batch, seq_q, d_v)
    # Each query gets a weighted combination of all values
    attention_output = np.matmul(attention_weights, value)

    return attention_output, attention_weights


def attention_backward(
    upstream_gradient: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attention_weights: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradients for scaled dot-product attention.

    This implements the backward pass through the attention computation,
    computing gradients with respect to query, key, and value.

    Args:
        upstream_gradient: Gradient from next layer, shape (batch, seq_q, d_v)
        query: Original query input, shape (batch, seq_q, d_k)
        key: Original key input, shape (batch, seq_k, d_k)
        value: Original value input, shape (batch, seq_k, d_v)
        attention_weights: Computed attention weights, shape (batch, seq_q, seq_k)
        mask: Optional mask used in forward pass

    Returns:
        d_query: Gradient w.r.t. query
        d_key: Gradient w.r.t. key
        d_value: Gradient w.r.t. value
    """
    d_k = query.shape[-1]
    scaling_factor = np.sqrt(d_k)

    # Gradient w.r.t. value
    # output = attention_weights @ value
    # d_value = attention_weights^T @ upstream_gradient
    d_value = np.matmul(attention_weights.transpose(0, 2, 1), upstream_gradient)

    # Gradient w.r.t. attention weights
    # d_attention_weights = upstream_gradient @ value^T
    d_attention_weights = np.matmul(upstream_gradient, value.transpose(0, 2, 1))

    # Gradient through softmax
    d_scaled_scores = softmax_backward(d_attention_weights, attention_weights)

    # Apply mask gradient (masked positions have zero gradient)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]
        d_scaled_scores = np.where(mask, d_scaled_scores, 0)

    # Gradient through scaling
    d_scores = d_scaled_scores / scaling_factor

    # Gradient w.r.t. query and key
    # scores = query @ key^T
    # d_query = d_scores @ key
    # d_key = d_scores^T @ query
    d_query = np.matmul(d_scores, key)
    d_key = np.matmul(d_scores.transpose(0, 2, 1), query)

    return d_query, d_key, d_value


class MultiHeadAttention:
    """
    Multi-Head Attention Layer.

    Instead of performing a single attention function, multi-head attention
    projects queries, keys, and values h times with different learned
    projections, performs attention in parallel, and concatenates the results.

    This allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Mathematical Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
        where head_i = Attention(Q @ W^Q_i, K @ W^K_i, V @ W^V_i)

    Architecture:
        1. Linear projections: Q, K, V each projected to d_model dimensions
        2. Split into h heads: each head has dimension d_k = d_model / h
        3. Parallel attention: run attention on each head independently
        4. Concatenate: combine all heads back together
        5. Final projection: linear layer to produce output

    Attributes:
        embedding_dimension: Total dimension of the model (d_model)
        num_heads: Number of attention heads (h)
        head_dimension: Dimension of each head (d_k = d_model / h)
        query_projection: Linear layer for Q projection
        key_projection: Linear layer for K projection
        value_projection: Linear layer for V projection
        output_projection: Final linear layer after concatenation

    Reference: "Attention Is All You Need" Section 3.2.2
    """

    def __init__(
        self, embedding_dimension: int, num_heads: int, dropout_rate: float = 0.0
    ):
        """
        Initialize Multi-Head Attention layer.

        Args:
            embedding_dimension: Size of input/output embeddings (d_model)
            num_heads: Number of attention heads
            dropout_rate: Dropout rate (not used in NumPy implementation)

        Raises:
            ValueError: If embedding_dimension is not divisible by num_heads
        """
        if embedding_dimension % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embedding_dimension}) must be divisible by "
                f"number of heads ({num_heads})"
            )

        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads
        self.dropout_rate = dropout_rate

        # Projection layers
        # Each projects from d_model to d_model
        # W^Q, W^K, W^V in the paper
        self.query_projection = Linear(embedding_dimension, embedding_dimension)
        self.key_projection = Linear(embedding_dimension, embedding_dimension)
        self.value_projection = Linear(embedding_dimension, embedding_dimension)

        # Output projection (W^O in the paper)
        self.output_projection = Linear(embedding_dimension, embedding_dimension)

        # Cache for backward pass
        self._query_cache = None
        self._key_cache = None
        self._value_cache = None
        self._projected_query_cache = None
        self._projected_key_cache = None
        self._projected_value_cache = None
        self._attention_weights_cache = None
        self._attention_output_cache = None
        self._mask_cache = None

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass through multi-head attention.

        Args:
            query: Query tensor, shape (batch_size, seq_len_q, embedding_dim)
            key: Key tensor, shape (batch_size, seq_len_k, embedding_dim)
            value: Value tensor, shape (batch_size, seq_len_k, embedding_dim)
            mask: Optional attention mask

        Returns:
            output: Attention output, shape (batch_size, seq_len_q, embedding_dim)
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]

        # Cache inputs for backward pass
        self._query_cache = query
        self._key_cache = key
        self._value_cache = value
        self._mask_cache = mask

        # Step 1: Project Q, K, V through linear layers
        # Shape: (batch, seq, d_model) -> (batch, seq, d_model)
        projected_query = self.query_projection.forward(query)
        projected_key = self.key_projection.forward(key)
        projected_value = self.value_projection.forward(value)

        self._projected_query_cache = projected_query
        self._projected_key_cache = projected_key
        self._projected_value_cache = projected_value

        # Step 2: Reshape to separate heads
        # Shape: (batch, seq, d_model) -> (batch, seq, num_heads, head_dim)
        # Then transpose: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        # This allows parallel attention computation across heads

        projected_query = projected_query.reshape(
            batch_size, seq_len_q, self.num_heads, self.head_dimension
        )
        projected_query = projected_query.transpose(
            0, 2, 1, 3
        )  # (batch, heads, seq_q, head_dim)

        projected_key = projected_key.reshape(
            batch_size, seq_len_k, self.num_heads, self.head_dimension
        )
        projected_key = projected_key.transpose(
            0, 2, 1, 3
        )  # (batch, heads, seq_k, head_dim)

        projected_value = projected_value.reshape(
            batch_size, seq_len_k, self.num_heads, self.head_dimension
        )
        projected_value = projected_value.transpose(
            0, 2, 1, 3
        )  # (batch, heads, seq_k, head_dim)

        # Step 3: Compute attention for all heads in parallel
        # Reshape to (batch * heads, seq, head_dim) for attention computation
        query_for_attention = projected_query.reshape(
            -1, seq_len_q, self.head_dimension
        )
        key_for_attention = projected_key.reshape(-1, seq_len_k, self.head_dimension)
        value_for_attention = projected_value.reshape(
            -1, seq_len_k, self.head_dimension
        )

        # Run attention
        attention_output, attention_weights = scaled_dot_product_attention(
            query_for_attention, key_for_attention, value_for_attention, mask=mask
        )

        # Reshape attention weights back: (batch * heads, seq_q, seq_k) -> (batch, heads, seq_q, seq_k)
        self._attention_weights_cache = attention_weights.reshape(
            batch_size, self.num_heads, seq_len_q, seq_len_k
        )

        # Step 4: Reshape attention output and concatenate heads
        # Shape: (batch * heads, seq_q, head_dim) -> (batch, heads, seq_q, head_dim)
        attention_output = attention_output.reshape(
            batch_size, self.num_heads, seq_len_q, self.head_dimension
        )

        # Transpose and reshape to concatenate heads
        # (batch, heads, seq_q, head_dim) -> (batch, seq_q, heads, head_dim) -> (batch, seq_q, d_model)
        attention_output = attention_output.transpose(0, 2, 1, 3)
        concatenated_output = attention_output.reshape(
            batch_size, seq_len_q, self.embedding_dimension
        )

        self._attention_output_cache = concatenated_output

        # Step 5: Final output projection
        output = self.output_projection.forward(concatenated_output)

        return output

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through multi-head attention.

        Computes gradients for all parameters and returns gradient w.r.t. input.

        Args:
            upstream_gradient: Gradient from next layer, shape (batch, seq_q, d_model)

        Returns:
            input_gradient: Gradient w.r.t. query input (for self-attention)
                           For cross-attention, gradients for K and V are stored separately
        """
        batch_size = upstream_gradient.shape[0]
        seq_len_q = upstream_gradient.shape[1]
        seq_len_k = self._key_cache.shape[1]

        # Backward through output projection
        d_concatenated = self.output_projection.backward(upstream_gradient)

        # Reshape for multi-head: (batch, seq_q, d_model) -> (batch, seq_q, heads, head_dim)
        d_concatenated = d_concatenated.reshape(
            batch_size, seq_len_q, self.num_heads, self.head_dimension
        )
        # -> (batch, heads, seq_q, head_dim)
        d_attention_output = d_concatenated.transpose(0, 2, 1, 3)

        # Reshape for attention backward
        d_attention_output_flat = d_attention_output.reshape(
            -1, seq_len_q, self.head_dimension
        )

        # Get cached projected values and reshape
        projected_query = self._projected_query_cache.reshape(
            batch_size, seq_len_q, self.num_heads, self.head_dimension
        )
        projected_query = projected_query.transpose(0, 2, 1, 3).reshape(
            -1, seq_len_q, self.head_dimension
        )

        projected_key = self._projected_key_cache.reshape(
            batch_size, seq_len_k, self.num_heads, self.head_dimension
        )
        projected_key = projected_key.transpose(0, 2, 1, 3).reshape(
            -1, seq_len_k, self.head_dimension
        )

        projected_value = self._projected_value_cache.reshape(
            batch_size, seq_len_k, self.num_heads, self.head_dimension
        )
        projected_value = projected_value.transpose(0, 2, 1, 3).reshape(
            -1, seq_len_k, self.head_dimension
        )

        attention_weights = self._attention_weights_cache.reshape(
            -1, seq_len_q, seq_len_k
        )

        # Backward through attention
        d_proj_query, d_proj_key, d_proj_value = attention_backward(
            d_attention_output_flat,
            projected_query,
            projected_key,
            projected_value,
            attention_weights,
            self._mask_cache,
        )

        # Reshape gradients back: (batch * heads, seq, head_dim) -> (batch, seq, d_model)
        d_proj_query = d_proj_query.reshape(
            batch_size, self.num_heads, seq_len_q, self.head_dimension
        )
        d_proj_query = d_proj_query.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embedding_dimension
        )

        d_proj_key = d_proj_key.reshape(
            batch_size, self.num_heads, seq_len_k, self.head_dimension
        )
        d_proj_key = d_proj_key.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_k, self.embedding_dimension
        )

        d_proj_value = d_proj_value.reshape(
            batch_size, self.num_heads, seq_len_k, self.head_dimension
        )
        d_proj_value = d_proj_value.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_k, self.embedding_dimension
        )

        # Backward through projection layers
        d_query = self.query_projection.backward(d_proj_query)
        d_key = self.key_projection.backward(d_proj_key)
        d_value = self.value_projection.backward(d_proj_value)

        # For self-attention, all gradients go to the same input
        # Return gradient w.r.t. query (which is also key and value in self-attention)
        # Store key and value gradients for potential cross-attention use
        self._d_key = d_key
        self._d_value = d_value

        return d_query

    def get_parameters(self) -> dict:
        """Return all learnable parameters."""
        params = {}
        params.update(
            {f"query_{k}": v for k, v in self.query_projection.get_parameters().items()}
        )
        params.update(
            {f"key_{k}": v for k, v in self.key_projection.get_parameters().items()}
        )
        params.update(
            {f"value_{k}": v for k, v in self.value_projection.get_parameters().items()}
        )
        params.update(
            {
                f"output_{k}": v
                for k, v in self.output_projection.get_parameters().items()
            }
        )
        return params

    def set_parameters(self, params: dict) -> None:
        """
        Set parameters from a dictionary.

        Args:
            params: Dictionary mapping parameter names to arrays
        """
        # Set query projection parameters
        if "query_weight" in params:
            self.query_projection.weight = params["query_weight"]
        if "query_bias" in params:
            self.query_projection.bias = params["query_bias"]

        # Set key projection parameters
        if "key_weight" in params:
            self.key_projection.weight = params["key_weight"]
        if "key_bias" in params:
            self.key_projection.bias = params["key_bias"]

        # Set value projection parameters
        if "value_weight" in params:
            self.value_projection.weight = params["value_weight"]
        if "value_bias" in params:
            self.value_projection.bias = params["value_bias"]

        # Set output projection parameters
        if "output_weight" in params:
            self.output_projection.weight = params["output_weight"]
        if "output_bias" in params:
            self.output_projection.bias = params["output_bias"]

    def get_gradients(self) -> dict:
        """Return all parameter gradients."""
        grads = {}
        grads.update(
            {f"query_{k}": v for k, v in self.query_projection.get_gradients().items()}
        )
        grads.update(
            {f"key_{k}": v for k, v in self.key_projection.get_gradients().items()}
        )
        grads.update(
            {f"value_{k}": v for k, v in self.value_projection.get_gradients().items()}
        )
        grads.update(
            {
                f"output_{k}": v
                for k, v in self.output_projection.get_gradients().items()
            }
        )
        return grads


def create_causal_mask(sequence_length: int) -> np.ndarray:
    """
    Create a causal (autoregressive) attention mask.

    In causal attention, each position can only attend to itself and
    previous positions, not future positions. This is essential for
    autoregressive language modeling where we predict one token at a time.

    Args:
        sequence_length: Length of the sequence

    Returns:
        mask: Boolean mask of shape (sequence_length, sequence_length)
              True where attention is allowed, False where masked

    Example:
        For sequence_length=4:
        [[True, False, False, False],   # Position 0 can only see position 0
         [True, True,  False, False],   # Position 1 can see 0, 1
         [True, True,  True,  False],   # Position 2 can see 0, 1, 2
         [True, True,  True,  True]]    # Position 3 can see all
    """
    # np.tril creates lower triangular matrix (including diagonal)
    return np.tril(np.ones((sequence_length, sequence_length), dtype=bool))


# =============================================================================
# EDUCATIONAL DEMO
# Run with: python -m src.attention
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ATTENTION MECHANISM DEMO")
    print("=" * 70)
    print()
    print("Attention is the core innovation of transformers. It allows each token")
    print("to 'look at' other tokens and decide which ones are relevant.")
    print()
    print("Dependencies:")
    print("  - src.layers (Linear layers for Q, K, V projections)")
    print("  - src.activations (softmax for attention weights)")
    print()

    from src.layers import Linear  # Showing the dependency

    # -------------------------------------------------------------------------
    # THE INTUITION: What is attention?
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("1. THE INTUITION - What problem does attention solve?")
    print("-" * 70)
    print()
    print("Consider: 'The cat sat on the mat because it was tired.'")
    print()
    print("When processing 'it', the model needs to figure out what 'it' refers to.")
    print(
        "Attention computes: 'How relevant is each previous word to understanding it?'"
    )
    print()
    print("  'The'     -> low relevance")
    print("  'cat'     -> HIGH relevance (it = the cat)")
    print("  'sat'     -> medium relevance")
    print("  'on'      -> low relevance")
    print("  'the'     -> low relevance")
    print("  'mat'     -> low relevance")
    print("  'because' -> medium relevance")
    print()

    # -------------------------------------------------------------------------
    # QUERY, KEY, VALUE: The attention mechanism
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("2. QUERY, KEY, VALUE - The three vectors")
    print("-" * 70)
    print()
    print("Each token produces three vectors:")
    print("  - Query (Q): 'What am I looking for?'")
    print("  - Key (K):   'What do I contain?'")
    print("  - Value (V): 'What information do I provide?'")
    print()
    print("The attention formula:")
    print("  Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V")
    print()
    print("Step by step:")
    print("  1. Q @ K.T = similarity scores (how well does each Q match each K?)")
    print("  2. / sqrt(d_k) = scale down to prevent extreme values")
    print("  3. softmax = convert to probabilities (sum to 1)")
    print("  4. @ V = weighted sum of values")
    print()

    # Simple example with small dimensions
    np.random.seed(42)
    batch_size = 1
    seq_len = 4
    d_k = 8  # Small dimension for clarity

    # Create sample Q, K, V (normally these come from linear projections)
    Q = np.random.randn(batch_size, seq_len, d_k) * 0.5
    K = np.random.randn(batch_size, seq_len, d_k) * 0.5
    V = np.random.randn(batch_size, seq_len, d_k) * 0.5

    print(f"Shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print("  (batch_size=1, sequence_length=4, dimension=8)")
    print()

    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("Attention weights (which tokens each position attends to):")
    print("  Each row sums to 1.0 (it's a probability distribution)")
    print()
    print("         Token0  Token1  Token2  Token3")
    for i in range(seq_len):
        weights_str = "  ".join(f"{w:.3f}" for w in attention_weights[0, i])
        print(f"  Pos {i}:  {weights_str}")
    print()

    # -------------------------------------------------------------------------
    # CAUSAL MASKING: Can't look at the future
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("3. CAUSAL MASKING - Preventing cheating")
    print("-" * 70)
    print()
    print("In language modeling, we predict the NEXT token.")
    print("Position 2 should not see tokens at positions 3, 4, 5...")
    print("(That would be cheating!)")
    print()

    mask = create_causal_mask(seq_len)
    print("Causal mask (True = can attend, False = blocked):")
    print()
    print("         Token0  Token1  Token2  Token3")
    for i in range(seq_len):
        mask_str = "  ".join("  Y  " if m else "  -  " for m in mask[i])
        print(f"  Pos {i}:  {mask_str}")
    print()
    print("Notice the triangular pattern: each position sees only itself and before.")
    print()

    # Apply causal mask
    output_causal, attention_weights_causal = scaled_dot_product_attention(
        Q, K, V, mask=mask
    )

    print("Attention weights WITH causal mask:")
    print()
    print("         Token0  Token1  Token2  Token3")
    for i in range(seq_len):
        weights_str = "  ".join(f"{w:.3f}" for w in attention_weights_causal[0, i])
        print(f"  Pos {i}:  {weights_str}")
    print()
    print("Notice: Position 0 has 1.000 for Token0 (can only see itself)")
    print("        Position 3 distributes attention across all tokens")
    print()

    # -------------------------------------------------------------------------
    # MULTI-HEAD ATTENTION: Multiple perspectives
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("4. MULTI-HEAD ATTENTION - Multiple perspectives")
    print("-" * 70)
    print()
    print("Instead of one attention, we run multiple in parallel ('heads').")
    print("Each head can learn to focus on different types of relationships:")
    print("  - Head 1 might learn syntax (subject-verb agreement)")
    print("  - Head 2 might learn semantics (word meaning relationships)")
    print("  - Head 3 might learn coreference (what pronouns refer to)")
    print()

    embedding_dim = 32
    num_heads = 4
    mha = MultiHeadAttention(embedding_dimension=embedding_dim, num_heads=num_heads)

    print("Multi-Head Attention:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {mha.head_dimension} (= {embedding_dim} / {num_heads})")
    print()

    # Forward pass - for self-attention, query=key=value=x
    x = np.random.randn(batch_size, seq_len, embedding_dim)
    causal_mask = create_causal_mask(seq_len)
    output = mha.forward(query=x, key=x, value=x, mask=causal_mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()
    print("The output has the same shape as input - attention transforms")
    print("each token's representation based on context from other tokens.")
    print()

    # -------------------------------------------------------------------------
    # VISUALIZING ATTENTION PATTERNS
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("5. INSIDE THE MULTI-HEAD ATTENTION")
    print("-" * 70)
    print()
    print("The projections Q, K, V are learned linear transformations:")
    print(f"  query_projection: Linear({embedding_dim} -> {embedding_dim})")
    print(f"  key_projection:   Linear({embedding_dim} -> {embedding_dim})")
    print(f"  value_projection: Linear({embedding_dim} -> {embedding_dim})")
    print(f"  output_projection: Linear({embedding_dim} -> {embedding_dim})")
    print()

    total_params = sum(p.size for p in mha.get_parameters().values())
    print(f"Total parameters in Multi-Head Attention: {total_params:,}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("- Attention lets tokens 'look at' other tokens to gather context")
    print("- Query/Key/Value: Q asks, K answers, V provides information")
    print("- Causal mask: Prevents looking at future tokens (for generation)")
    print("- Multi-head: Multiple attention patterns learned in parallel")
    print()
    print("Attention is what makes transformers powerful - it allows modeling")
    print("long-range dependencies that RNNs struggle with.")
    print()
    print("Next step: Run 'python -m src.transformer' to see how attention")
    print("           combines with feed-forward networks in transformer blocks.")
