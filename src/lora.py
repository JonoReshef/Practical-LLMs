"""
LoRA (Low-Rank Adaptation) Implementation

LoRA is a parameter-efficient fine-tuning technique that freezes the pre-trained
model weights and injects trainable low-rank matrices into transformer layers.

Key Concept:
    Instead of fine-tuning W directly, LoRA learns a low-rank decomposition:
    W' = W + BA

    Where:
    - W is the frozen pre-trained weight matrix (d_out x d_in)
    - B is a trainable matrix (d_out x r) initialized to zeros
    - A is a trainable matrix (r x d_in) initialized with small random values
    - r << min(d_in, d_out) is the "rank" (typically 4, 8, or 16)

Benefits:
    - Dramatically reduces trainable parameters (e.g., GPT-3: 175B -> 4.7M)
    - No inference latency added (LoRA weights can be merged into base)
    - Works well for domain adaptation and instruction tuning
    - Multiple LoRA adapters can be swapped efficiently

Reference:
    "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
    https://arxiv.org/abs/2106.09685

Classes:
    LoRALinear: Linear layer with LoRA adaptation

Functions:
    apply_lora_to_model: Add LoRA adapters to a GPT model
    merge_lora_weights: Merge LoRA weights into base weights
    get_lora_parameters: Get only the LoRA parameters from a model
    get_lora_gradients: Get gradients for LoRA parameters
"""

from typing import Dict, List, Optional

import numpy as np

from src.model import GPTModel


class LoRALinear:
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    This implements a linear layer where the base weights are frozen and
    only low-rank adaptation matrices are trained.

    Forward computation:
        output = x @ W^T + x @ (BA)^T * scaling + bias
               = x @ W^T + x @ A^T @ B^T * (alpha/rank) + bias

    Where:
    - W: Frozen base weights (d_out, d_in)
    - A: Trainable down-projection (rank, d_in) - initialized randomly
    - B: Trainable up-projection (d_out, rank) - initialized to zeros
    - scaling = alpha / rank: Scaling factor for LoRA contribution

    Why initialize B to zeros?
        This ensures that at the start of training, the LoRA contribution
        is zero, so the model behaves exactly like the pre-trained model.
        Gradients will make B non-zero during training.

    Why use scaling (alpha/rank)?
        When changing rank, you don't want to retune the learning rate.
        The scaling factor keeps the magnitude consistent across ranks.

    Attributes:
        base_weights: Frozen pre-trained weight matrix
        base_bias: Frozen bias (or None)
        lora_A: Down-projection matrix (rank, input_features)
        lora_B: Up-projection matrix (output_features, rank)
        rank: LoRA rank (bottleneck dimension)
        alpha: LoRA scaling factor numerator
        scaling: Computed as alpha / rank
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        base_weights: Optional[np.ndarray] = None,
        base_bias: Optional[np.ndarray] = None,
    ):
        """
        Initialize LoRA linear layer.

        Args:
            input_features: Input dimension
            output_features: Output dimension
            rank: LoRA rank (bottleneck dimension). Lower = fewer params, higher = more capacity
            alpha: Scaling factor numerator. Common choices: same as rank, or 2x rank
            base_weights: Pre-trained weights to freeze. If None, initialize randomly.
            base_bias: Pre-trained bias to freeze. If None, no bias.
        """
        self.input_features = input_features
        self.output_features = output_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Scaling factor for LoRA contribution

        # Base (frozen) weights
        if base_weights is not None:
            self.base_weights = base_weights.copy()  # Copy to ensure no shared memory
        else:
            # Initialize randomly (for testing purposes)
            weight_std = np.sqrt(2.0 / (input_features + output_features))
            self.base_weights = (
                np.random.randn(output_features, input_features) * weight_std
            )

        self.base_bias = base_bias.copy() if base_bias is not None else None

        # LoRA matrices (trainable)
        # A: Down-projection, initialized with small random values
        # Using Kaiming initialization for A
        a_std = np.sqrt(2.0 / input_features)
        self.lora_A = np.random.randn(rank, input_features) * a_std

        # B: Up-projection, initialized to zeros
        # This ensures LoRA contribution is zero at start
        self.lora_B = np.zeros((output_features, rank))

        # Gradient placeholders
        self.lora_A_gradient = None
        self.lora_B_gradient = None

        # Cache for backward pass
        self._input_cache = None
        self._lora_intermediate_cache = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass with LoRA adaptation.

        Args:
            input_tensor: Input of shape (..., input_features)

        Returns:
            Output of shape (..., output_features)

        Computation:
            output = (input @ W^T) + (input @ A^T @ B^T) * scaling + bias
        """
        self._input_cache = input_tensor

        # Base model contribution (frozen)
        # input @ W^T
        base_output = np.matmul(input_tensor, self.base_weights.T)

        # LoRA contribution (trainable)
        # input @ A^T -> (batch, seq, rank)
        lora_intermediate = np.matmul(input_tensor, self.lora_A.T)
        self._lora_intermediate_cache = lora_intermediate

        # intermediate @ B^T -> (batch, seq, output_features)
        lora_output = np.matmul(lora_intermediate, self.lora_B.T)

        # Apply scaling and combine
        output = base_output + lora_output * self.scaling

        # Add bias if present
        if self.base_bias is not None:
            output = output + self.base_bias

        return output

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through LoRA layer.

        Only computes gradients for LoRA matrices (A and B).
        The base weights are frozen and don't receive gradients.

        Args:
            upstream_gradient: Gradient from next layer, shape (..., output_features)

        Returns:
            Gradient w.r.t. input, shape (..., input_features)
        """
        input_tensor = self._input_cache
        lora_intermediate = self._lora_intermediate_cache

        # Reshape for gradient computation
        original_shape = input_tensor.shape
        if input_tensor.ndim > 2:
            batch_dims = input_tensor.shape[:-1]
            input_2d = input_tensor.reshape(-1, self.input_features)
            upstream_2d = upstream_gradient.reshape(-1, self.output_features)
            intermediate_2d = lora_intermediate.reshape(-1, self.rank)
        else:
            input_2d = input_tensor
            upstream_2d = upstream_gradient
            intermediate_2d = lora_intermediate

        # Scale upstream gradient for LoRA path
        scaled_upstream = upstream_2d * self.scaling

        # Gradient w.r.t. B: upstream^T @ intermediate
        # Shape: (output_features, batch*seq) @ (batch*seq, rank) -> (output_features, rank)
        self.lora_B_gradient = scaled_upstream.T @ intermediate_2d

        # Gradient w.r.t. intermediate (for A gradient)
        # Shape: (batch*seq, output_features) @ (output_features, rank) -> (batch*seq, rank)
        d_intermediate = scaled_upstream @ self.lora_B

        # Gradient w.r.t. A: d_intermediate^T @ input
        # Shape: (rank, batch*seq) @ (batch*seq, input_features) -> (rank, input_features)
        self.lora_A_gradient = d_intermediate.T @ input_2d

        # Input gradient (through both base and LoRA paths)
        # Base: upstream @ W (frozen, but still need for chain rule)
        d_input_base = upstream_2d @ self.base_weights

        # LoRA: d_intermediate @ A
        d_input_lora = d_intermediate @ self.lora_A

        d_input = d_input_base + d_input_lora

        # Reshape back
        if len(original_shape) > 2:
            d_input = d_input.reshape(original_shape)

        return d_input

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters for compatibility with Linear interface."""
        return {
            "weight": self.base_weights,  # Return base weights for compatibility
            "bias": self.base_bias,
        }

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get gradients for compatibility with Linear interface."""
        # Return LoRA gradients under standard names for the training loop
        return {
            "weight": np.zeros_like(self.base_weights),  # Base is frozen
            "bias": np.zeros_like(self.base_bias)
            if self.base_bias is not None
            else None,
        }

    def get_lora_parameters(self) -> Dict[str, np.ndarray]:
        """Get only the LoRA parameters (trainable)."""
        return {"lora_A": self.lora_A, "lora_B": self.lora_B}

    def get_lora_gradients(self) -> Dict[str, np.ndarray]:
        """Get gradients for LoRA parameters."""
        return {"lora_A": self.lora_A_gradient, "lora_B": self.lora_B_gradient}

    def get_all_parameters(self) -> Dict[str, np.ndarray]:
        """Get all parameters (including frozen base)."""
        params = {
            "base_weights": self.base_weights,
            "lora_A": self.lora_A,
            "lora_B": self.lora_B,
        }
        if self.base_bias is not None:
            params["base_bias"] = self.base_bias
        return params


def merge_lora_weights(
    base_weights: np.ndarray, lora_A: np.ndarray, lora_B: np.ndarray, scaling: float
) -> np.ndarray:
    """
    Merge LoRA weights into base weights for inference.

    After training, we can merge the LoRA weights into the base weights
    for efficient inference without any overhead.

    W' = W + B @ A * scaling

    Args:
        base_weights: Original frozen weights, shape (d_out, d_in)
        lora_A: LoRA down-projection, shape (rank, d_in)
        lora_B: LoRA up-projection, shape (d_out, rank)
        scaling: LoRA scaling factor (alpha / rank)

    Returns:
        Merged weights, shape (d_out, d_in)
    """
    # B @ A: (d_out, rank) @ (rank, d_in) -> (d_out, d_in)
    lora_weights = np.matmul(lora_B, lora_A)

    # Merge: W + B @ A * scaling
    merged_weights = base_weights + lora_weights * scaling

    return merged_weights


def apply_lora_to_model(
    model: GPTModel, rank: int = 4, alpha: float = 8.0, target_modules: List[str] = None
) -> GPTModel:
    """
    Apply LoRA adapters to specified modules in a GPT model.

    This modifies the model in-place to use LoRA layers instead of
    regular linear layers for the specified modules.

    Common target modules for GPT:
    - 'query': Query projection in attention (most important)
    - 'key': Key projection in attention
    - 'value': Value projection in attention (important)
    - 'output': Output projection in attention
    - 'ffn': Feed-forward network layers

    Args:
        model: GPT model to modify
        rank: LoRA rank for all adapters
        alpha: LoRA alpha (scaling numerator) for all adapters
        target_modules: List of module names to apply LoRA to.
                       Default: ['query', 'value'] (recommended by paper)

    Returns:
        Modified model with LoRA adapters (same object, modified in-place)
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    # Track which LoRA layers we've added
    model._lora_layers = {}

    # Apply LoRA to each transformer block
    for block_idx, block in enumerate(model.transformer_stack.blocks):
        attention = block.self_attention

        if "query" in target_modules:
            # Replace query projection with LoRA version
            lora_query = LoRALinear(
                input_features=attention.embedding_dimension,
                output_features=attention.embedding_dimension,
                rank=rank,
                alpha=alpha,
                base_weights=attention.query_projection.weights,
                base_bias=attention.query_projection.bias,
            )
            attention._original_query_projection = attention.query_projection
            attention.query_projection = lora_query
            model._lora_layers[f"block_{block_idx}_query"] = lora_query

        if "key" in target_modules:
            lora_key = LoRALinear(
                input_features=attention.embedding_dimension,
                output_features=attention.embedding_dimension,
                rank=rank,
                alpha=alpha,
                base_weights=attention.key_projection.weights,
                base_bias=attention.key_projection.bias,
            )
            attention._original_key_projection = attention.key_projection
            attention.key_projection = lora_key
            model._lora_layers[f"block_{block_idx}_key"] = lora_key

        if "value" in target_modules:
            lora_value = LoRALinear(
                input_features=attention.embedding_dimension,
                output_features=attention.embedding_dimension,
                rank=rank,
                alpha=alpha,
                base_weights=attention.value_projection.weights,
                base_bias=attention.value_projection.bias,
            )
            attention._original_value_projection = attention.value_projection
            attention.value_projection = lora_value
            model._lora_layers[f"block_{block_idx}_value"] = lora_value

        if "output" in target_modules:
            lora_output = LoRALinear(
                input_features=attention.embedding_dimension,
                output_features=attention.embedding_dimension,
                rank=rank,
                alpha=alpha,
                base_weights=attention.output_projection.weights,
                base_bias=attention.output_projection.bias,
            )
            attention._original_output_projection = attention.output_projection
            attention.output_projection = lora_output
            model._lora_layers[f"block_{block_idx}_output"] = lora_output

    return model


def get_lora_parameters(model: GPTModel) -> Dict[str, np.ndarray]:
    """
    Get only the LoRA parameters from a model.

    This is useful for training, where we only want to update LoRA weights.

    Args:
        model: Model with LoRA adapters applied

    Returns:
        Dictionary mapping parameter names to LoRA parameter arrays
    """
    if not hasattr(model, "_lora_layers"):
        return {}

    params = {}
    for layer_name, lora_layer in model._lora_layers.items():
        lora_params = lora_layer.get_lora_parameters()
        for param_name, param in lora_params.items():
            params[f"{layer_name}_{param_name}"] = param

    return params


def get_lora_gradients(model: GPTModel) -> Dict[str, np.ndarray]:
    """
    Get gradients for LoRA parameters from a model.

    Args:
        model: Model with LoRA adapters after backward pass

    Returns:
        Dictionary mapping parameter names to gradient arrays
    """
    if not hasattr(model, "_lora_layers"):
        return {}

    grads = {}
    for layer_name, lora_layer in model._lora_layers.items():
        lora_grads = lora_layer.get_lora_gradients()
        for grad_name, grad in lora_grads.items():
            if grad is not None:
                grads[f"{layer_name}_{grad_name}"] = grad

    return grads


def set_lora_parameters(model: GPTModel, params: Dict[str, np.ndarray]) -> None:
    """
    Set LoRA parameters from a dictionary.

    Args:
        model: Model with LoRA adapters
        params: Dictionary of parameter name -> array
    """
    if not hasattr(model, "_lora_layers"):
        return

    for layer_name, lora_layer in model._lora_layers.items():
        a_key = f"{layer_name}_lora_A"
        b_key = f"{layer_name}_lora_B"

        if a_key in params:
            lora_layer.lora_A = params[a_key]
        if b_key in params:
            lora_layer.lora_B = params[b_key]


def count_lora_parameters(model: GPTModel) -> int:
    """Count the number of trainable LoRA parameters."""
    params = get_lora_parameters(model)
    return sum(p.size for p in params.values())


def count_total_parameters(model: GPTModel) -> int:
    """Count total parameters in the model."""
    return sum(p.size for p in model.get_parameters().values())
