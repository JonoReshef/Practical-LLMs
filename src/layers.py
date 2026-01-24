"""
Neural Network Layers for Transformer Models

This module implements fundamental neural network layers used in transformer
architectures, with both forward and backward passes for gradient computation.

All implementations are in pure NumPy for educational purposes.

Classes:
    Linear: Fully connected layer (y = xW^T + b)
    LayerNorm: Layer normalization for stable training
    Embedding: Token ID to dense vector lookup table
    PositionalEncoding: Sinusoidal position embeddings

Reference:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "Layer Normalization" (Ba et al., 2016)
"""

import numpy as np


class Linear:
    """
    Fully Connected (Linear) Layer.

    Computes the affine transformation: y = x @ W^T + b

    This is the fundamental building block of neural networks. In transformers,
    it's used for:
    - Query, Key, Value projections in attention
    - Feed-forward network layers
    - Output projection to vocabulary

    Attributes:
        weights: Weight matrix of shape (output_features, input_features)
        bias: Bias vector of shape (output_features,) or None
        weight_gradient: Gradient of loss w.r.t. weights
        bias_gradient: Gradient of loss w.r.t. bias

    Weight Initialization:
        Xavier/Glorot initialization: W ~ N(0, sqrt(2 / (fan_in + fan_out)))
        This helps maintain variance of activations across layers.
    """

    def __init__(
        self, input_features: int, output_features: int, use_bias: bool = True
    ):
        """
        Initialize Linear layer with Xavier initialization.

        Args:
            input_features: Size of input dimension (fan_in)
            output_features: Size of output dimension (fan_out)
            use_bias: Whether to include a bias term
        """
        self.input_features = input_features
        self.output_features = output_features
        self.use_bias = use_bias

        # Xavier/Glorot initialization
        # Variance = 2 / (fan_in + fan_out)
        weight_std = np.sqrt(2.0 / (input_features + output_features))
        self.weights = np.random.randn(output_features, input_features) * weight_std

        if use_bias:
            self.bias = np.zeros(output_features)
        else:
            self.bias = None

        # Placeholders for gradients
        self.weight_gradient = None
        self.bias_gradient = None

        # Cache for backward pass
        self._input_cache = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W^T + b

        Args:
            input_tensor: Input of shape (..., input_features)
                         Can be 2D (batch, features) or 3D (batch, seq, features)

        Returns:
            output_tensor: Output of shape (..., output_features)

        Mathematical Operation:
            For each sample x in the batch:
            y = x @ W^T + b

            Where:
            - x has shape (input_features,)
            - W has shape (output_features, input_features)
            - W^T has shape (input_features, output_features)
            - y has shape (output_features,)
        """
        # Cache input for backward pass
        self._input_cache = input_tensor

        # Matrix multiplication: (..., in) @ (in, out) -> (..., out)
        output_tensor = input_tensor @ self.weights.T

        # Add bias if present
        if self.use_bias:
            output_tensor = output_tensor + self.bias

        return output_tensor

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients for weights, bias, and input.

        Args:
            upstream_gradient: Gradient from the next layer, shape (..., output_features)

        Returns:
            input_gradient: Gradient w.r.t. input, shape (..., input_features)

        Mathematical Derivation:
            Forward: y = x @ W^T + b

            d_loss/d_W = upstream^T @ x  (summed over batch)
            d_loss/d_b = sum(upstream)   (summed over batch)
            d_loss/d_x = upstream @ W
        """
        input_tensor = self._input_cache
        original_shape = input_tensor.shape

        # Reshape to 2D for gradient computation
        if input_tensor.ndim > 2:
            batch_dims = input_tensor.shape[:-1]
            input_2d = input_tensor.reshape(-1, self.input_features)
            upstream_2d = upstream_gradient.reshape(-1, self.output_features)
        else:
            input_2d = input_tensor
            upstream_2d = upstream_gradient

        # Weight gradient: upstream^T @ input
        # Shape: (output_features, batch) @ (batch, input_features) -> (output_features, input_features)
        self.weight_gradient = upstream_2d.T @ input_2d

        # Bias gradient: sum over batch
        if self.use_bias:
            self.bias_gradient = np.sum(upstream_2d, axis=0)

        # Input gradient: upstream @ W
        # Shape: (batch, output_features) @ (output_features, input_features) -> (batch, input_features)
        input_gradient = upstream_2d @ self.weights

        # Reshape back to original batch dimensions
        if len(original_shape) > 2:
            input_gradient = input_gradient.reshape(original_shape)

        return input_gradient

    def get_parameters(self) -> dict:
        """Return dictionary of learnable parameters."""
        params = {"weight": self.weights}  # Use 'weight' for consistency
        if self.use_bias:
            params["bias"] = self.bias
        return params

    def get_gradients(self) -> dict:
        """Return dictionary of parameter gradients."""
        grads = {"weight": self.weight_gradient}  # Use 'weight' for consistency
        if self.use_bias:
            grads["bias"] = self.bias_gradient
        return grads

    @property
    def weight(self) -> np.ndarray:
        """Alias for weights for compatibility."""
        return self.weights

    @weight.setter
    def weight(self, value: np.ndarray) -> None:
        """Alias for weights for compatibility."""
        self.weights = value


class LayerNorm:
    """
    Layer Normalization.

    Normalizes activations across the feature dimension, which helps stabilize
    training by reducing internal covariate shift.

    Formula:
        y = gamma * (x - mean) / sqrt(var + eps) + beta

    Where:
        - mean and var are computed across the last dimension (features)
        - gamma and beta are learnable parameters
        - eps is a small constant for numerical stability

    Unlike BatchNorm, LayerNorm:
        - Normalizes across features, not across batch
        - Works the same during training and inference
        - Is the standard choice for transformers

    Reference: "Layer Normalization" (Ba et al., 2016)
    """

    def __init__(self, normalized_shape: int, epsilon: float = 1e-5):
        """
        Initialize LayerNorm.

        Args:
            normalized_shape: Size of the last dimension to normalize over
            epsilon: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        # Learnable parameters
        # Gamma (scale): initialized to 1
        self.gamma = np.ones(normalized_shape)
        # Beta (shift): initialized to 0
        self.beta = np.zeros(normalized_shape)

        # Gradients
        self.gamma_gradient = None
        self.beta_gradient = None

        # Cache for backward pass
        self._input_cache = None
        self._normalized_cache = None
        self._std_cache = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass: normalize input and apply learnable affine transformation.

        Args:
            input_tensor: Input of shape (..., normalized_shape)

        Returns:
            output_tensor: Normalized output of same shape

        Steps:
            1. Compute mean across last axis
            2. Compute variance across last axis
            3. Normalize: (x - mean) / sqrt(var + eps)
            4. Scale and shift: gamma * normalized + beta
        """
        # Cache input for backward pass
        self._input_cache = input_tensor

        # Step 1: Compute mean across the feature dimension (last axis)
        mean = np.mean(input_tensor, axis=-1, keepdims=True)

        # Step 2: Compute variance across the feature dimension
        variance = np.var(input_tensor, axis=-1, keepdims=True)

        # Step 3: Normalize
        # std = sqrt(var + eps) for numerical stability
        std = np.sqrt(variance + self.epsilon)
        self._std_cache = std

        normalized = (input_tensor - mean) / std
        self._normalized_cache = normalized

        # Step 4: Apply learnable affine transformation
        # gamma scales, beta shifts
        output_tensor = self.gamma * normalized + self.beta

        return output_tensor

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients for gamma, beta, and input.

        The gradient computation for LayerNorm is complex because mean and
        variance depend on all input elements.

        Args:
            upstream_gradient: Gradient from the next layer

        Returns:
            input_gradient: Gradient w.r.t. input
        """
        normalized = self._normalized_cache
        std = self._std_cache
        input_tensor = self._input_cache

        # Number of features being normalized
        n_features = self.normalized_shape

        # Gradient w.r.t. gamma: sum of (upstream * normalized) over batch
        self.gamma_gradient = np.sum(
            upstream_gradient * normalized,
            axis=tuple(range(upstream_gradient.ndim - 1)),
        )

        # Gradient w.r.t. beta: sum of upstream over batch
        self.beta_gradient = np.sum(
            upstream_gradient, axis=tuple(range(upstream_gradient.ndim - 1))
        )

        # Gradient w.r.t. normalized
        d_normalized = upstream_gradient * self.gamma

        # Gradient w.r.t. input (complex due to mean and variance dependencies)
        # Using the formula from the LayerNorm paper
        d_var = np.sum(
            d_normalized
            * (input_tensor - np.mean(input_tensor, axis=-1, keepdims=True))
            * (-0.5)
            * np.power(std, -3),
            axis=-1,
            keepdims=True,
        )

        d_mean = np.sum(
            d_normalized * (-1.0 / std), axis=-1, keepdims=True
        ) + d_var * np.mean(
            -2.0 * (input_tensor - np.mean(input_tensor, axis=-1, keepdims=True)),
            axis=-1,
            keepdims=True,
        )

        input_gradient = (
            (d_normalized / std)
            + (
                d_var
                * 2.0
                * (input_tensor - np.mean(input_tensor, axis=-1, keepdims=True))
                / n_features
            )
            + (d_mean / n_features)
        )

        return input_gradient

    def get_parameters(self) -> dict:
        """Return dictionary of learnable parameters."""
        return {"gamma": self.gamma, "beta": self.beta}

    def get_gradients(self) -> dict:
        """Return dictionary of parameter gradients."""
        return {"gamma": self.gamma_gradient, "beta": self.beta_gradient}


class Embedding:
    """
    Embedding Layer (Lookup Table).

    Converts discrete token IDs into dense vector representations by looking up
    rows in a learned embedding matrix.

    In transformers, embeddings are used for:
        - Input token embeddings (words/subwords -> vectors)
        - Output projection (often tied with input embeddings)

    Reference: "Attention Is All You Need" Section 3.4
    """

    def __init__(self, vocabulary_size: int, embedding_dimension: int):
        """
        Initialize Embedding layer.

        Args:
            vocabulary_size: Number of unique tokens in vocabulary
            embedding_dimension: Size of embedding vectors
        """
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension

        # Initialize embedding table with normal distribution
        # Scale by 1/sqrt(d) is common practice
        scale = 1.0 / np.sqrt(embedding_dimension)
        self.embedding_table = (
            np.random.randn(vocabulary_size, embedding_dimension) * scale
        )

        # Gradient placeholder
        self.embedding_gradient = None

        # Cache for backward pass
        self._token_ids_cache = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass: look up embeddings for token IDs.

        Args:
            token_ids: Integer array of shape (batch_size, sequence_length) or (sequence_length,)
                      Values should be in range [0, vocabulary_size)

        Returns:
            embeddings: Float array of shape (..., embedding_dimension)

        Operation:
            Simply index into the embedding table using token IDs.
            embeddings[i] = embedding_table[token_ids[i]]
        """
        # Cache token IDs for backward pass
        self._token_ids_cache = token_ids

        # Simple table lookup using advanced indexing
        embeddings = self.embedding_table[token_ids]

        return embeddings

    def backward(self, upstream_gradient: np.ndarray) -> None:
        """
        Backward pass: compute gradient for embedding table.

        Args:
            upstream_gradient: Gradient from next layer, shape matches forward output

        Note:
            - There's no input gradient to return (token IDs are discrete)
            - Gradients accumulate for tokens that appear multiple times
        """
        token_ids = self._token_ids_cache

        # Initialize gradient to zeros
        self.embedding_gradient = np.zeros_like(self.embedding_table)

        # Flatten for easier indexing
        flat_token_ids = token_ids.flatten()
        flat_gradient = upstream_gradient.reshape(-1, self.embedding_dimension)

        # Accumulate gradients for each token
        # np.add.at handles repeated indices correctly (accumulates)
        np.add.at(self.embedding_gradient, flat_token_ids, flat_gradient)

    def get_parameters(self) -> dict:
        """Return dictionary of learnable parameters."""
        return {"embedding_table": self.embedding_table}

    def get_gradients(self) -> dict:
        """Return dictionary of parameter gradients."""
        return {"embedding_table": self.embedding_gradient}

    @property
    def weight(self) -> np.ndarray:
        """Alias for embedding_table for compatibility."""
        return self.embedding_table

    @weight.setter
    def weight(self, value: np.ndarray) -> None:
        """Alias for embedding_table for compatibility."""
        self.embedding_table = value


class PositionalEncoding:
    """
    Sinusoidal Positional Encoding.

    Adds position information to embeddings using fixed sinusoidal patterns.
    This allows the model to understand the order of tokens in a sequence.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
        - Each position has a unique encoding
        - Encoding is bounded in [-1, 1]
        - Model can learn to attend to relative positions because
          PE(pos+k) can be represented as a linear function of PE(pos)

    Reference: "Attention Is All You Need" Section 3.5
    """

    def __init__(self, max_sequence_length: int, embedding_dimension: int):
        """
        Initialize and precompute positional encodings.

        Args:
            max_sequence_length: Maximum sequence length to support
            embedding_dimension: Must match the embedding dimension of the model
        """
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension

        # Precompute positional encodings for all positions
        self.encoding_table = self._create_encoding_table()

    def _create_encoding_table(self) -> np.ndarray:
        """
        Create the full positional encoding table.

        Returns:
            encoding_table: Array of shape (max_sequence_length, embedding_dimension)
        """
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = np.arange(self.max_sequence_length)[
            :, np.newaxis
        ]  # Shape: (max_seq, 1)

        # Dimension indices for the formula: [0, 2, 4, ..., d-2] for sin
        # and [1, 3, 5, ..., d-1] for cos
        dimension_indices = np.arange(self.embedding_dimension)[
            np.newaxis, :
        ]  # Shape: (1, d)

        # Compute the angle rates
        # angle_rate = 1 / 10000^(2i/d_model) = 10000^(-2i/d_model)
        # We use 2*(i//2) to get [0,0,2,2,4,4,...] pattern
        angle_rates = 1 / np.power(
            10000.0, (2 * (dimension_indices // 2)) / self.embedding_dimension
        )

        # Compute angles: position * angle_rate
        angles = positions * angle_rates  # Shape: (max_seq, d)

        # Apply sin to even indices, cos to odd indices
        encoding_table = np.zeros_like(angles)
        encoding_table[:, 0::2] = np.sin(angles[:, 0::2])  # Even dimensions: sin
        encoding_table[:, 1::2] = np.cos(angles[:, 1::2])  # Odd dimensions: cos

        return encoding_table

    def get_encoding(self, sequence_length: int) -> np.ndarray:
        """
        Get positional encoding for a specific sequence length.

        Args:
            sequence_length: Length of the sequence (must be <= max_sequence_length)

        Returns:
            encoding: Array of shape (sequence_length, embedding_dimension)
        """
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum {self.max_sequence_length}"
            )

        return self.encoding_table[:sequence_length]

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.

        Args:
            embeddings: Input embeddings of shape (batch_size, sequence_length, embedding_dimension)
                       or (sequence_length, embedding_dimension)

        Returns:
            Output with positional encoding added, same shape as input
        """
        if embeddings.ndim == 2:
            sequence_length = embeddings.shape[0]
        else:
            sequence_length = embeddings.shape[1]

        position_encoding = self.get_encoding(sequence_length)

        return embeddings + position_encoding


# =============================================================================
# EDUCATIONAL DEMO
# Run with: python -m src.layers
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL NETWORK LAYERS DEMO")
    print("=" * 70)
    print()
    print("This module provides the building blocks for neural networks:")
    print("  - Linear: Learnable matrix multiplication (the core operation)")
    print("  - LayerNorm: Stabilizes training by normalizing activations")
    print("  - Embedding: Converts token IDs to vectors")
    print("  - PositionalEncoding: Adds position information to embeddings")
    print()
    print("Dependencies: src.activations (uses activation functions)")
    print()

    # -------------------------------------------------------------------------
    # LINEAR LAYER: The fundamental building block
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("1. LINEAR LAYER - Learnable transformation")
    print("-" * 70)
    print()
    print("A linear layer computes: output = input @ weights.T + bias")
    print("The weights and bias are learned during training.")
    print()

    # Create a linear layer: 4 input features -> 3 output features
    linear = Linear(input_features=4, output_features=3)
    print(f"Linear layer: {linear.input_features} -> {linear.output_features}")
    print(f"  Weight shape: {linear.weights.shape} (output_features x input_features)")
    print(f"  Bias shape: {linear.bias.shape}")
    print(f"  Total parameters: {linear.weights.size + linear.bias.size}")
    print()

    # Forward pass
    x = np.array([[1.0, 2.0, 3.0, 4.0]])  # Batch of 1, 4 features
    output = linear.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()

    # Backward pass demonstration
    upstream_grad = np.ones_like(output)  # Gradient from next layer
    input_grad = linear.backward(upstream_grad)
    grads = linear.get_gradients()
    print("After backward pass, we have gradients for learning:")
    print(f"  Weight gradient shape: {grads['weight'].shape}")
    print(f"  Bias gradient shape: {grads['bias'].shape}")
    print(f"  Input gradient shape: {input_grad.shape} (to pass to previous layer)")
    print()

    # -------------------------------------------------------------------------
    # LAYER NORMALIZATION: Training stability
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("2. LAYER NORMALIZATION - Stabilizing activations")
    print("-" * 70)
    print()
    print("LayerNorm normalizes values to have mean=0, std=1, then scales/shifts.")
    print("This prevents values from exploding or vanishing during training.")
    print()

    layer_norm = LayerNorm(normalized_shape=4)

    # Simulate activations with varying scales
    x = np.array([[100.0, 200.0, 300.0, 400.0], [0.001, 0.002, 0.003, 0.004]])
    print("Input (notice very different scales between rows):")
    print(f"  Row 1: {x[0]}")
    print(f"  Row 2: {x[1]}")
    print()

    normalized = layer_norm.forward(x)
    print("After LayerNorm (both rows now have similar scale):")
    print(f"  Row 1: {normalized[0]}")
    print(f"  Row 2: {normalized[1]}")
    print()
    print(f"  Row 1 mean: {normalized[0].mean():.6f}, std: {normalized[0].std():.6f}")
    print(f"  Row 2 mean: {normalized[1].mean():.6f}, std: {normalized[1].std():.6f}")
    print()

    # -------------------------------------------------------------------------
    # EMBEDDING: Token IDs to vectors
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("3. EMBEDDING - Converting tokens to vectors")
    print("-" * 70)
    print()
    print("The embedding layer is a lookup table: each token ID maps to a vector.")
    print("These vectors are learned during training to capture token meaning.")
    print()

    # Create embedding: vocabulary of 100 tokens, 8-dimensional embeddings
    vocab_size = 100
    embedding_dim = 8
    embedding = Embedding(vocabulary_size=vocab_size, embedding_dimension=embedding_dim)
    print(f"Embedding table shape: {embedding.embedding_table.shape}")
    print(f"  - {vocab_size} tokens in vocabulary")
    print(f"  - Each token represented by {embedding_dim} numbers")
    print()

    # Look up embeddings for some token IDs
    token_ids = np.array([[5, 10, 15]])  # 3 tokens
    embeddings = embedding.forward(token_ids)
    print(f"Token IDs: {token_ids[0]}")
    print(f"Output shape: {embeddings.shape} (batch=1, seq_len=3, embed_dim=8)")
    print()
    print("Embedding for token 5 (first 4 values):", embeddings[0, 0, :4].round(3))
    print("Embedding for token 10 (first 4 values):", embeddings[0, 1, :4].round(3))
    print()
    print("Key insight: Similar tokens will have similar embeddings after training.")
    print()

    # -------------------------------------------------------------------------
    # POSITIONAL ENCODING: Where is each token?
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("4. POSITIONAL ENCODING - Adding position information")
    print("-" * 70)
    print()
    print("Token embeddings don't know their position in the sequence.")
    print("'Dog bites man' and 'Man bites dog' would look the same!")
    print("Positional encoding adds unique patterns for each position.")
    print()

    pos_encoder = PositionalEncoding(
        embedding_dimension=embedding_dim, max_sequence_length=50
    )

    # Get encodings for first 5 positions
    pos_encodings = pos_encoder.get_encoding(5)
    print(f"Positional encoding shape: {pos_encodings.shape} (5 positions, 8 dims)")
    print()
    print("Each position has a unique pattern (first 4 dimensions shown):")
    for pos in range(5):
        print(f"  Position {pos}: {pos_encodings[pos, :4].round(3)}")
    print()

    # Show how it's added to embeddings
    print("Adding positional encoding to token embeddings:")
    combined = pos_encoder.forward(embeddings)
    print(f"  Original embedding[0,0]: {embeddings[0, 0, :4].round(3)}")
    print(f"  + Position encoding[0]:  {pos_encodings[0, :4].round(3)}")
    print(f"  = Combined[0,0]:         {combined[0, 0, :4].round(3)}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("- Linear: The learnable transformation (weights and biases)")
    print("- LayerNorm: Keeps values stable during training")
    print("- Embedding: Turns token IDs into meaningful vectors")
    print("- PositionalEncoding: Tells the model where each token is")
    print()
    print("In a transformer, data flows:")
    print("  Token IDs -> Embedding -> + PositionalEncoding -> Linear layers...")
    print()
    print("Next step: Run 'python -m src.tokenizer' to see how text becomes token IDs.")
