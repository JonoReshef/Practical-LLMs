# Transformer Block: The Fundamental Building Unit

A Transformer block is the repeating unit that gives transformers their name. It combines attention, feed-forward processing, residual connections, and layer normalization into a single modular component that can be stacked to build deep networks.

---

## Table of Contents

1. [Block Architecture](#block-architecture)
2. [Residual Connections](#residual-connections)
3. [Layer Normalization](#layer-normalization)
4. [Pre-LN vs Post-LN](#pre-ln-vs-post-ln)
5. [Step-by-Step Numeric Example](#step-by-step-numeric-example)
6. [Why These Components Work Together](#why-these-components-work-together)
7. [Code Implementation](#code-implementation)
8. [Visualization](#visualization)
9. [References](#references)

---

## Block Architecture

A single Transformer block consists of:

1. **Layer Normalization** (Pre-LN) - Normalize before attention
2. **Multi-Head Self-Attention** - Tokens communicate
3. **Residual Connection** - Add original input back
4. **Layer Normalization** (Pre-LN) - Normalize before FFN
5. **Feed-Forward Network** - Per-position processing
6. **Residual Connection** - Add back again

### Mathematical Formulation

$$x_1 = x + \text{Attention}(\text{LayerNorm}(x))$$
$$x_2 = x_1 + \text{FFN}(\text{LayerNorm}(x_1))$$

### Block Diagram

```
Input x
    │
    ├──────────────────────┐
    │                      │
    ▼                      │
LayerNorm                  │
    │                      │
    ▼                      │
Multi-Head Attention       │
(with causal mask)         │
    │                      │
    ▼                      │
+  ◄───────────────────────┘  (Residual: add input back)
    │
    ├──────────────────────┐
    │                      │
    ▼                      │
LayerNorm                  │
    │                      │
    ▼                      │
Feed-Forward Network       │
    │                      │
    ▼                      │
+  ◄───────────────────────┘  (Residual: add back)
    │
    ▼
Output
```

---

## Residual Connections

**Residual connections** (or skip connections) allow the input to "skip over" a layer and be added directly to the output.

### The Concept

```python
# Without residual connection:
output = layer(input)

# With residual connection:
output = layer(input) + input
```

### Why Residual Connections?

1. **Gradient Flow**: Gradients can flow directly through the skip connection, avoiding vanishing gradients
2. **Easier to Learn Identity**: If a layer should do nothing, it just learns zero
3. **Enables Deep Networks**: Without residuals, networks degrade with depth

### Gradient Flow Visualization

```
Forward Pass:              Backward Pass (with residuals):

x ──────────┬─────────►    ◄───────────┬─────────── dL/dx
            │              gradient    │
            ▼              flows       │
         Layer             through     ▼
            │              BOTH       Layer
            ▼              paths       │
x + Layer(x)               dL/dx + dL/dLayer
            │                          │
            ▼                          ▼
         Output                    Input gradient
                                   always non-zero!
```

### Mathematical Insight

For a residual block $f$:
$$y = x + f(x)$$

The gradient is:
$$\frac{\partial y}{\partial x} = 1 + \frac{\partial f}{\partial x}$$

Even if $\frac{\partial f}{\partial x} \approx 0$, the gradient is still $\approx 1$!

---

## Layer Normalization

**Layer Normalization** normalizes each sample independently across the feature dimension.

### The Formula

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:

- $\mu$ = mean across features: $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$
- $\sigma^2$ = variance across features: $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$
- $\gamma$ = learnable scale parameter
- $\beta$ = learnable shift parameter
- $\epsilon$ = small constant for numerical stability (typically 1e-5)

### Numeric Example

```python
# Input: single position with 4 features
x = [2.0, 4.0, 6.0, 8.0]

# Step 1: Compute mean
mu = (2 + 4 + 6 + 8) / 4 = 5.0

# Step 2: Compute variance
var = ((2-5)² + (4-5)² + (6-5)² + (8-5)²) / 4
    = (9 + 1 + 1 + 9) / 4
    = 5.0

# Step 3: Normalize
std = sqrt(5.0 + 1e-5) ≈ 2.236
x_normalized = (x - mu) / std
             = [(2-5)/2.236, (4-5)/2.236, (6-5)/2.236, (8-5)/2.236]
             = [-1.34, -0.45, 0.45, 1.34]

# Step 4: Scale and shift (assuming γ=1, β=0)
output = x_normalized
       = [-1.34, -0.45, 0.45, 1.34]

# Verify: mean ≈ 0, std ≈ 1 ✓
```

### Why Layer Normalization?

| Without Normalization          | With Normalization               |
| ------------------------------ | -------------------------------- |
| Activations can grow unbounded | Values stay in predictable range |
| Internal covariate shift       | Stable statistics                |
| Harder to train deep networks  | Easier optimization              |
| Learning rate sensitive        | More robust                      |

---

## Pre-LN vs Post-LN

### Post-LN (Original Transformer)

```
x ──► Attention ──► + ──► LayerNorm ──► FFN ──► + ──► LayerNorm ──► output
        │          ▲                     │      ▲
        └──────────┘(residual)           └──────┘(residual)
```

### Pre-LN (GPT-2 and later)

```
x ──► LayerNorm ──► Attention ──► + ──► LayerNorm ──► FFN ──► + ──► output
  │                              ▲   │                       ▲
  └──────────────────────────────┘   └───────────────────────┘
              (residual)                    (residual)
```

### Why Pre-LN?

| Aspect             | Post-LN                    | Pre-LN                        |
| ------------------ | -------------------------- | ----------------------------- |
| Gradient flow      | Can accumulate before norm | Direct path through residuals |
| Training stability | Requires careful warmup    | More stable                   |
| Final output       | Already normalized         | Needs final LayerNorm         |
| Modern preference  | Less common                | Standard in GPT-2/3/4         |

### Gradient Path Comparison

```
Post-LN:
Gradient must pass through LayerNorm before reaching residual
─► Problem: gradients can become very small

Pre-LN:
Gradient flows directly through residual connection
─► Solution: consistent gradient magnitude
```

---

## Step-by-Step Numeric Example

Let's trace through a complete Transformer block:

### Setup

```
Embedding dimension: 4
Number of heads: 2
FFN hidden dimension: 8
Sequence length: 3
```

### Input

```python
x = [
    [1.0, 2.0, 0.5, 1.5],   # Position 0
    [0.5, 1.0, 1.5, 2.0],   # Position 1
    [2.0, 0.5, 1.0, 1.5],   # Position 2
]
# Shape: (3, 4)
```

### Step 1: First LayerNorm

For position 0: `[1.0, 2.0, 0.5, 1.5]`

```python
mu = (1.0 + 2.0 + 0.5 + 1.5) / 4 = 1.25
var = ((1-1.25)² + (2-1.25)² + (0.5-1.25)² + (1.5-1.25)²) / 4
    = (0.0625 + 0.5625 + 0.5625 + 0.0625) / 4 = 0.3125
std = sqrt(0.3125) ≈ 0.559

norm_0 = [(1-1.25)/0.559, (2-1.25)/0.559, (0.5-1.25)/0.559, (1.5-1.25)/0.559]
       = [-0.45, 1.34, -1.34, 0.45]

# Similarly for other positions:
norm_1 = [-0.84, -0.17, 0.50, 1.51]  # Simplified
norm_2 = [1.17, -0.84, -0.50, 0.17]  # Simplified
```

### Step 2: Multi-Head Attention

(Using simplified attention for illustration)

```python
# After attention with 2 heads:
attn_output = [
    [-0.30, 0.85, -0.70, 0.55],   # Position 0 attends to past
    [-0.50, 0.30, 0.10, 0.90],   # Position 1
    [0.20, -0.40, 0.60, 0.70],   # Position 2
]
```

### Step 3: First Residual Connection

```python
# Add original input back
x_1 = x + attn_output

# Position 0:
x_1[0] = [1.0, 2.0, 0.5, 1.5] + [-0.30, 0.85, -0.70, 0.55]
       = [0.70, 2.85, -0.20, 2.05]

# Full result:
x_1 = [
    [0.70, 2.85, -0.20, 2.05],
    [0.00, 1.30, 1.60, 2.90],
    [2.20, 0.10, 1.60, 2.20],
]
```

### Step 4: Second LayerNorm

```python
# Normalize x_1 (showing position 0)
mu = (0.70 + 2.85 - 0.20 + 2.05) / 4 = 1.35
var = ... ≈ 1.35
std ≈ 1.16

norm_x1 = [
    [-0.56, 1.29, -1.34, 0.60],
    [-0.84, 0.10, 0.35, 1.39],
    [0.42, -1.18, 0.07, 0.69],
]
```

### Step 5: Feed-Forward Network

```python
# Expand to 8 dims, GELU, compress back to 4
# (Simplified values)
ffn_output = [
    [0.15, 0.40, -0.30, 0.20],
    [0.10, 0.25, 0.35, 0.45],
    [0.30, -0.15, 0.20, 0.25],
]
```

### Step 6: Second Residual Connection

```python
# Add x_1 back
output = x_1 + ffn_output

output = [
    [0.70+0.15, 2.85+0.40, -0.20-0.30, 2.05+0.20],
    [0.00+0.10, 1.30+0.25, 1.60+0.35, 2.90+0.45],
    [2.20+0.30, 0.10-0.15, 1.60+0.20, 2.20+0.25],
]

# Final output:
output = [
    [0.85, 3.25, -0.50, 2.25],
    [0.10, 1.55, 1.95, 3.35],
    [2.50, -0.05, 1.80, 2.45],
]
```

### Summary of Transformations

```
Position 0 through the block:

Input:          [1.00, 2.00, 0.50, 1.50]
                         │
After Norm 1:   [-0.45, 1.34, -1.34, 0.45]
                         │
After Attn:     [-0.30, 0.85, -0.70, 0.55]
                         │
+ Residual:     [0.70, 2.85, -0.20, 2.05]
                         │
After Norm 2:   [-0.56, 1.29, -1.34, 0.60]
                         │
After FFN:      [0.15, 0.40, -0.30, 0.20]
                         │
+ Residual:     [0.85, 3.25, -0.50, 2.25]
                         │
Output          ▼
```

---

## Why These Components Work Together

### The Role of Each Component

| Component     | Purpose                          | What Happens Without It         |
| ------------- | -------------------------------- | ------------------------------- |
| **Attention** | Token communication              | No context understanding        |
| **FFN**       | Per-position processing          | No non-linearity                |
| **Residual**  | Gradient flow, identity learning | Training fails for deep nets    |
| **LayerNorm** | Stable activations               | Exploding/vanishing activations |

### Synergy of Components

```
LayerNorm → Stabilizes inputs to attention/FFN
     │
Attention → Gathers relevant information from context
     │
Residual → Preserves original information + new context
     │
LayerNorm → Stabilizes for next processing step
     │
FFN → Processes gathered information non-linearly
     │
Residual → Final fusion: original + context + processed
```

---

## Code Implementation

From [src/transformer.py](src/transformer.py):

```python
class TransformerBlock:
    """
    Single Transformer Decoder Block.

    Uses Pre-LayerNorm architecture (GPT-2 style):
    - LayerNorm before attention
    - Residual connection after attention
    - LayerNorm before FFN
    - Residual connection after FFN
    """

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        ffn_hidden_dimension: int = None,
    ):
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads

        # Layer norms
        self.attention_layer_norm = LayerNorm(embedding_dimension)
        self.ffn_layer_norm = LayerNorm(embedding_dimension)

        # Self-attention
        self.self_attention = MultiHeadAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads
        )

        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            embedding_dimension=embedding_dimension,
            hidden_dimension=ffn_hidden_dimension,
        )

    def forward(
        self,
        input_tensor: np.ndarray,
        use_causal_mask: bool = True,
    ) -> np.ndarray:
        """
        Forward pass through the transformer block.

        Args:
            input_tensor: Shape (batch, seq_len, embedding_dim)
            use_causal_mask: Whether to use causal masking

        Returns:
            Output of same shape as input
        """
        # Cache for residual
        self._input_cache = input_tensor
        seq_len = input_tensor.shape[1]

        # Create causal mask
        mask = create_causal_mask(seq_len) if use_causal_mask else None

        # ============ Attention Sub-block ============
        # Step 1: Pre-attention layer norm
        normed = self.attention_layer_norm.forward(input_tensor)

        # Step 2: Self-attention
        attn_output = self.self_attention.forward(
            query=normed, key=normed, value=normed, mask=mask
        )

        # Step 3: Residual connection
        post_attention = input_tensor + attn_output

        # ============ FFN Sub-block ============
        # Step 4: Pre-FFN layer norm
        normed = self.ffn_layer_norm.forward(post_attention)

        # Step 5: Feed-forward network
        ffn_output = self.feed_forward.forward(normed)

        # Step 6: Residual connection
        output = post_attention + ffn_output

        return output
```

### Layer Normalization Implementation

```python
class LayerNorm:
    """
    Layer Normalization.

    Normalizes across the feature dimension:
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, normalized_shape: int, epsilon: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        # Learnable parameters, initialized to identity transform
        self.gamma = np.ones(normalized_shape)   # Scale
        self.beta = np.zeros(normalized_shape)   # Shift

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """Normalize input tensor."""
        self._input_cache = input_tensor

        # Compute statistics across last axis (features)
        mean = np.mean(input_tensor, axis=-1, keepdims=True)
        variance = np.var(input_tensor, axis=-1, keepdims=True)

        # Normalize
        std = np.sqrt(variance + self.epsilon)
        normalized = (input_tensor - mean) / std

        # Scale and shift
        output = self.gamma * normalized + self.beta

        return output
```

---

## Visualization

### Complete Block Flow

```
                    Transformer Block
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   Input x                                                  │
│   [batch, seq_len, embed_dim]                              │
│        │                                                   │
│        ├──────────────────────────────────┐                │
│        │                                  │                │
│        ▼                                  │                │
│   ┌─────────────┐                        │                │
│   │ LayerNorm   │                        │                │
│   └──────┬──────┘                        │                │
│          │                               │                │
│          ▼                               │                │
│   ┌─────────────────────────┐           │                │
│   │ Multi-Head Self-Attn    │           │                │
│   │ ┌─────────────────────┐ │           │                │
│   │ │ Head 1 │ Head 2 │...│ │           │                │
│   │ └─────────────────────┘ │           │                │
│   └──────────┬──────────────┘           │                │
│              │                          │                │
│              ▼                          │                │
│           [  +  ] ◄─────────────────────┘  Residual      │
│              │                                            │
│        ├──────────────────────────────────┐                │
│        │                                  │                │
│        ▼                                  │                │
│   ┌─────────────┐                        │                │
│   │ LayerNorm   │                        │                │
│   └──────┬──────┘                        │                │
│          │                               │                │
│          ▼                               │                │
│   ┌─────────────────────────┐           │                │
│   │ Feed-Forward Network    │           │                │
│   │  Linear → GELU → Linear │           │                │
│   └──────────┬──────────────┘           │                │
│              │                          │                │
│              ▼                          │                │
│           [  +  ] ◄─────────────────────┘  Residual      │
│              │                                            │
│              ▼                                            │
│   Output                                                   │
│   [batch, seq_len, embed_dim]                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Stacking Multiple Blocks

```
              Input Embeddings + Positional Encoding
                              │
                              ▼
                   ┌────────────────────┐
                   │ Transformer Block 1│
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Transformer Block 2│
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Transformer Block 3│
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Transformer Block 4│
                   └─────────┬──────────┘
                             │
                             ▼
                       Final LayerNorm
                             │
                             ▼
                    Output Projection
```

### Information Processing Per Block

```
Layer  What Changes                     Example (conceptual)
─────  ──────────────────────────────  ─────────────────────────────
Input: Raw token embeddings            "cat": [0.2, 0.4, 0.1, ...]
                                       "sat": [0.3, 0.1, 0.5, ...]

Block 1: Local patterns                "cat" learns it's near "sat"
└─► Attention finds immediate          "sat" learns it's a verb
    neighbors

Block 2: Syntactic relations           "cat" knows it's the subject
└─► Subject-verb connections           "sat" knows "cat" is doing it

Block 3: Semantic understanding        "cat" encodes "animal doing action"
└─► Meaning accumulates

Block 4: High-level representation     "cat" = full contextual meaning
└─► Ready for prediction               ready to predict next word
```

---

## Try It Yourself

Run the transformer demo:

```bash
python -m src.transformer
```

Or experiment in Python:

```python
from src.transformer import TransformerBlock
import numpy as np

# Create a transformer block
block = TransformerBlock(
    embedding_dimension=128,
    num_heads=4,
    ffn_hidden_dimension=512
)

# Sample input
x = np.random.randn(2, 10, 128)  # batch=2, seq=10, dim=128

# Forward pass
output = block.forward(x, use_causal_mask=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Same shape? {x.shape == output.shape}")

# Verify residual connection effect
# If we set attention and FFN weights near zero,
# output should be close to input (identity)
```

---

## References

1. **Original Transformer**: [Vaswani, A., et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)

2. **Pre-LN Analysis**: [Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

3. **Deep Residual Learning**: [He, K., et al. (2015). Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

4. **Layer Normalization**: [Ba, J., et al. (2016). Layer Normalization](https://arxiv.org/abs/1607.06450)

5. **This Repository**: See [src/transformer.py](src/transformer.py) for `TransformerBlock` and `TransformerStack` classes.

---

**Next Step**: Now we understand individual blocks. Continue to [07 - GPTModel.md](07%20-%20GPTModel.md) to see how everything assembles into a complete language model.
