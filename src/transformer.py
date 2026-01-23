"""
Transformer Architecture Components

This module implements the core transformer building blocks: the feed-forward
network and the transformer block (which combines attention, FFN, and residuals).

The transformer block is the fundamental repeating unit that gets stacked
to create the full model. Each block allows the model to:
1. Attend to relevant context (via attention)
2. Process information (via FFN)
3. Maintain information flow (via residual connections)

Reference: "Attention Is All You Need" (Vaswani et al., 2017) Section 3.1, 3.3
           "Language Models are Unsupervised Multitask Learners" (GPT-2 paper)

Classes:
    FeedForwardNetwork: Position-wise feed-forward network
    TransformerBlock: Single transformer decoder block
    TransformerStack: Stack of transformer blocks
"""

from typing import List, Optional

import numpy as np

from src.activations import gelu, gelu_backward
from src.attention import MultiHeadAttention, create_causal_mask
from src.layers import LayerNorm, Linear


class FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network.

    This is applied independently to each position in the sequence.
    It consists of two linear transformations with a GELU activation:

        FFN(x) = Linear_2(GELU(Linear_1(x)))

    The hidden dimension is typically 4x the model dimension, allowing
    the network to expand the representation, process it, and compress
    it back down.

    Mathematical Formula:
        FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
        (Original paper uses ReLU, modern models use GELU)

    Why FFN after attention?
        Attention captures relationships between positions, but is linear
        in the values. The FFN adds non-linearity and allows position-specific
        transformations.

    Reference: "Attention Is All You Need" Section 3.3

    Attributes:
        embedding_dimension: Input/output dimension (d_model)
        hidden_dimension: Inner dimension (d_ff), typically 4 * d_model
        linear_1: First linear transformation (expansion)
        linear_2: Second linear transformation (compression)
    """

    def __init__(
        self, embedding_dimension: int, hidden_dimension: Optional[int] = None
    ):
        """
        Initialize Feed-Forward Network.

        Args:
            embedding_dimension: Input and output dimension
            hidden_dimension: Inner hidden dimension (default: 4 * embedding_dimension)
        """
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension or (4 * embedding_dimension)

        # First linear layer: expand from d_model to d_ff
        self.linear_1 = Linear(
            input_features=embedding_dimension, output_features=self.hidden_dimension
        )

        # Second linear layer: compress from d_ff back to d_model
        self.linear_2 = Linear(
            input_features=self.hidden_dimension, output_features=embedding_dimension
        )

        # Cache for backward pass
        self._input_cache = None
        self._hidden_cache = None  # Output of linear_1 (before activation)
        self._activated_cache = None  # Output after GELU

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feed-forward network.

        Args:
            input_tensor: Input of shape (batch_size, seq_len, embedding_dim)

        Returns:
            Output of shape (batch_size, seq_len, embedding_dim)

        Computation:
            1. Linear expansion: d_model -> d_ff
            2. GELU activation
            3. Linear compression: d_ff -> d_model
        """
        self._input_cache = input_tensor

        # Step 1: Expand to hidden dimension
        hidden = self.linear_1.forward(input_tensor)
        self._hidden_cache = hidden

        # Step 2: Apply GELU activation
        activated = gelu(hidden)
        self._activated_cache = activated

        # Step 3: Compress back to embedding dimension
        output = self.linear_2.forward(activated)

        return output

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through the feed-forward network.

        Args:
            upstream_gradient: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # Backward through linear_2
        d_activated = self.linear_2.backward(upstream_gradient)

        # Backward through GELU
        d_hidden = gelu_backward(d_activated, self._hidden_cache)

        # Backward through linear_1
        d_input = self.linear_1.backward(d_hidden)

        return d_input

    def get_parameters(self) -> dict:
        """Return all learnable parameters."""
        params = {}
        params.update(
            {f"ffn_linear1_{k}": v for k, v in self.linear_1.get_parameters().items()}
        )
        params.update(
            {f"ffn_linear2_{k}": v for k, v in self.linear_2.get_parameters().items()}
        )
        return params

    def set_parameters(self, params: dict) -> None:
        """
        Set parameters from a dictionary.

        Args:
            params: Dictionary mapping parameter names to arrays
        """
        # Set linear_1 parameters
        if "ffn_linear1_weight" in params:
            self.linear_1.weight = params["ffn_linear1_weight"]
        if "ffn_linear1_bias" in params:
            self.linear_1.bias = params["ffn_linear1_bias"]

        # Set linear_2 parameters
        if "ffn_linear2_weight" in params:
            self.linear_2.weight = params["ffn_linear2_weight"]
        if "ffn_linear2_bias" in params:
            self.linear_2.bias = params["ffn_linear2_bias"]

    def get_gradients(self) -> dict:
        """Return all parameter gradients."""
        grads = {}
        grads.update(
            {f"ffn_linear1_{k}": v for k, v in self.linear_1.get_gradients().items()}
        )
        grads.update(
            {f"ffn_linear2_{k}": v for k, v in self.linear_2.get_gradients().items()}
        )
        return grads


class TransformerBlock:
    """
    Single Transformer Decoder Block.

    This is the fundamental repeating unit of a GPT-style transformer.
    It uses the Pre-LayerNorm architecture (norm before attention/FFN)
    which is more stable for training deep networks.

    Architecture (Pre-LN):
        x -> LayerNorm -> MultiHeadAttention -> + (residual)
                                                |
        x -> LayerNorm -> FeedForward --------> + (residual) -> output

    Components:
        1. Pre-attention LayerNorm
        2. Causal Multi-Head Self-Attention
        3. Residual connection (add input back)
        4. Pre-FFN LayerNorm
        5. Position-wise Feed-Forward Network
        6. Residual connection

    Why Pre-LN (Layer Norm before attention)?
        The original Transformer used Post-LN (norm after residual).
        Pre-LN is more stable for training very deep networks because
        the gradient flows directly through the residual path.

    Reference:
        - "Attention Is All You Need" Section 3.1
        - "On Layer Normalization in the Transformer Architecture"
    """

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        ffn_hidden_dimension: Optional[int] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize a Transformer Block.

        Args:
            embedding_dimension: Model dimension (d_model)
            num_heads: Number of attention heads
            ffn_hidden_dimension: FFN hidden dimension (default: 4 * d_model)
            dropout_rate: Dropout rate (not used in NumPy implementation)
        """
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads

        # Layer norms (Pre-LN architecture)
        self.attention_layer_norm = LayerNorm(embedding_dimension)
        self.ffn_layer_norm = LayerNorm(embedding_dimension)

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )

        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            embedding_dimension=embedding_dimension,
            hidden_dimension=ffn_hidden_dimension,
        )

        # Cache for backward pass
        self._input_cache = None
        self._normed_for_attention_cache = None
        self._attention_output_cache = None
        self._post_attention_cache = None
        self._normed_for_ffn_cache = None

    def forward(
        self,
        input_tensor: np.ndarray,
        use_causal_mask: bool = True,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass through the transformer block.

        Args:
            input_tensor: Input of shape (batch_size, seq_len, embedding_dim)
            use_causal_mask: Whether to apply causal (autoregressive) masking
            attention_mask: Optional additional attention mask

        Returns:
            Output of shape (batch_size, seq_len, embedding_dim)
        """
        self._input_cache = input_tensor
        sequence_length = input_tensor.shape[1]

        # Create causal mask if needed
        if use_causal_mask:
            causal_mask = create_causal_mask(sequence_length)
            if attention_mask is not None:
                # Combine masks
                mask = causal_mask & attention_mask
            else:
                mask = causal_mask
        else:
            mask = attention_mask

        # ============ Attention Sub-block ============
        # Pre-LN: normalize before attention
        normed_for_attention = self.attention_layer_norm.forward(input_tensor)
        self._normed_for_attention_cache = normed_for_attention

        # Self-attention (Q=K=V=normed input)
        attention_output = self.self_attention.forward(
            query=normed_for_attention,
            key=normed_for_attention,
            value=normed_for_attention,
            mask=mask,
        )
        self._attention_output_cache = attention_output

        # Residual connection: add input to attention output
        post_attention = input_tensor + attention_output
        self._post_attention_cache = post_attention

        # ============ Feed-Forward Sub-block ============
        # Pre-LN: normalize before FFN
        normed_for_ffn = self.ffn_layer_norm.forward(post_attention)
        self._normed_for_ffn_cache = normed_for_ffn

        # Feed-forward network
        ffn_output = self.feed_forward.forward(normed_for_ffn)

        # Residual connection: add post-attention to FFN output
        output = post_attention + ffn_output

        return output

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through the transformer block.

        Args:
            upstream_gradient: Gradient from next layer/block

        Returns:
            Gradient with respect to input
        """
        # ============ FFN Sub-block Backward ============
        # Gradient splits at residual connection
        d_ffn_output = upstream_gradient
        d_post_attention_from_residual = upstream_gradient  # Residual path

        # Backward through FFN
        d_normed_for_ffn = self.feed_forward.backward(d_ffn_output)

        # Backward through FFN layer norm
        d_post_attention_from_norm = self.ffn_layer_norm.backward(d_normed_for_ffn)

        # Combine gradients at post-attention
        d_post_attention = d_post_attention_from_residual + d_post_attention_from_norm

        # ============ Attention Sub-block Backward ============
        # Gradient splits at residual connection
        d_attention_output = d_post_attention
        d_input_from_residual = d_post_attention  # Residual path

        # Backward through self-attention
        d_normed_for_attention = self.self_attention.backward(d_attention_output)

        # Backward through attention layer norm
        d_input_from_norm = self.attention_layer_norm.backward(d_normed_for_attention)

        # Combine gradients at input
        d_input = d_input_from_residual + d_input_from_norm

        return d_input

    def get_parameters(self) -> dict:
        """Return all learnable parameters."""
        params = {}
        params.update(
            {
                f"attn_ln_{k}": v
                for k, v in self.attention_layer_norm.get_parameters().items()
            }
        )
        params.update(
            {f"ffn_ln_{k}": v for k, v in self.ffn_layer_norm.get_parameters().items()}
        )
        params.update(
            {f"attn_{k}": v for k, v in self.self_attention.get_parameters().items()}
        )
        params.update(self.feed_forward.get_parameters())
        return params

    def set_parameters(self, params: dict) -> None:
        """
        Set parameters from a dictionary.

        Args:
            params: Dictionary mapping parameter names to arrays
        """
        # Set attention layer norm parameters
        if "attn_ln_gamma" in params:
            self.attention_layer_norm.gamma = params["attn_ln_gamma"]
        if "attn_ln_beta" in params:
            self.attention_layer_norm.beta = params["attn_ln_beta"]

        # Set FFN layer norm parameters
        if "ffn_ln_gamma" in params:
            self.ffn_layer_norm.gamma = params["ffn_ln_gamma"]
        if "ffn_ln_beta" in params:
            self.ffn_layer_norm.beta = params["ffn_ln_beta"]

        # Set attention parameters
        attn_params = {
            k[5:]: v
            for k, v in params.items()
            if k.startswith("attn_") and not k.startswith("attn_ln")
        }
        if attn_params:
            self.self_attention.set_parameters(attn_params)

        # Set feed-forward parameters
        self.feed_forward.set_parameters(params)

    def get_gradients(self) -> dict:
        """Return all parameter gradients."""
        grads = {}
        grads.update(
            {
                f"attn_ln_{k}": v
                for k, v in self.attention_layer_norm.get_gradients().items()
            }
        )
        grads.update(
            {f"ffn_ln_{k}": v for k, v in self.ffn_layer_norm.get_gradients().items()}
        )
        grads.update(
            {f"attn_{k}": v for k, v in self.self_attention.get_gradients().items()}
        )
        grads.update(self.feed_forward.get_gradients())
        return grads


class TransformerStack:
    """
    Stack of Transformer Blocks.

    This creates the main body of a GPT-style model by stacking multiple
    transformer blocks. Each block refines the representation through
    attention and feed-forward processing.

    Typical configurations:
        - GPT-2 Small: 12 blocks
        - GPT-2 Medium: 24 blocks
        - GPT-2 Large: 36 blocks
        - GPT-2 XL: 48 blocks
        - Our educational model: 4 blocks (for fast CPU training)

    Attributes:
        blocks: List of TransformerBlock instances
        num_layers: Number of transformer blocks
    """

    def __init__(
        self,
        num_layers: int,
        embedding_dimension: int,
        num_heads: int,
        ffn_hidden_dimension: Optional[int] = None,
    ):
        """
        Initialize a stack of transformer blocks.

        Args:
            num_layers: Number of transformer blocks to stack
            embedding_dimension: Model dimension (d_model)
            num_heads: Number of attention heads
            ffn_hidden_dimension: FFN hidden dimension (default: 4 * d_model)
        """
        self.num_layers = num_layers
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads

        # Create stack of transformer blocks
        self.blocks: List[TransformerBlock] = []
        for i in range(num_layers):
            block = TransformerBlock(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                ffn_hidden_dimension=ffn_hidden_dimension,
            )
            self.blocks.append(block)

    def forward(
        self,
        input_tensor: np.ndarray,
        use_causal_mask: bool = True,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass through all transformer blocks.

        Args:
            input_tensor: Input of shape (batch_size, seq_len, embedding_dim)
            use_causal_mask: Whether to apply causal masking
            attention_mask: Optional additional attention mask

        Returns:
            Output of shape (batch_size, seq_len, embedding_dim)
        """
        hidden_states = input_tensor

        # Pass through each block sequentially
        for block in self.blocks:
            hidden_states = block.forward(
                hidden_states,
                use_causal_mask=use_causal_mask,
                attention_mask=attention_mask,
            )

        return hidden_states

    def backward(self, upstream_gradient: np.ndarray) -> tuple:
        """
        Backward pass through all transformer blocks.

        Args:
            upstream_gradient: Gradient from the layer after the stack

        Returns:
            Tuple of (gradient with respect to input, dict of parameter gradients)
        """
        gradient = upstream_gradient

        # Pass gradient backward through blocks in reverse order
        for block in reversed(self.blocks):
            gradient = block.backward(gradient)

        # Collect all gradients
        all_gradients = self.get_gradients()

        return gradient, all_gradients

    def get_parameters(self) -> dict:
        """Return all learnable parameters from all blocks."""
        params = {}
        for i, block in enumerate(self.blocks):
            block_params = block.get_parameters()
            params.update({f"block_{i}_{k}": v for k, v in block_params.items()})
        return params

    def set_parameters(self, params: dict) -> None:
        """
        Set parameters from a dictionary.

        Args:
            params: Dictionary mapping parameter names to arrays
        """
        for i, block in enumerate(self.blocks):
            prefix = f"block_{i}_"
            block_params = {
                k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)
            }
            if block_params:
                block.set_parameters(block_params)

    def get_gradients(self) -> dict:
        """Return all parameter gradients from all blocks."""
        grads = {}
        for i, block in enumerate(self.blocks):
            block_grads = block.get_gradients()
            grads.update({f"block_{i}_{k}": v for k, v in block_grads.items()})
        return grads
