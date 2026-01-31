# GPT Model: Assembling the Complete Architecture

The GPT (Generative Pre-trained Transformer) model combines all the components we've studied into a complete language model capable of understanding and generating text.

---

## Table of Contents

1. [Model Architecture Overview](#model-architecture-overview)
2. [Component Assembly](#component-assembly)
3. [Forward Pass Walkthrough](#forward-pass-walkthrough)
4. [Numeric Example](#numeric-example)
5. [Weight Sharing](#weight-sharing)
6. [Model Configuration](#model-configuration)
7. [Code Implementation](#code-implementation)
8. [Visualization](#visualization)
9. [References](#references)

---

## Model Architecture Overview

A GPT model is a **decoder-only transformer** designed for autoregressive language modeling.

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                        GPT Model                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Token IDs: [15, 234, 89, ...]                        │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                        │
│  │ Token Embedding │  Vocabulary → Vectors                   │
│  └────────┬────────┘                                        │
│           │                                                  │
│           │        ┌─────────────────────┐                  │
│           ├──────► │ Positional Encoding │                  │
│           │        └─────────┬───────────┘                  │
│           │                  │                              │
│           └────────────[  +  ]  Element-wise addition       │
│                          │                                   │
│                          ▼                                   │
│             ┌───────────────────────┐                       │
│             │ Transformer Block 1   │                       │
│             └───────────┬───────────┘                       │
│                         │                                    │
│             ┌───────────────────────┐                       │
│             │ Transformer Block 2   │                       │
│             └───────────┬───────────┘                       │
│                        ...                                   │
│             ┌───────────────────────┐                       │
│             │ Transformer Block N   │                       │
│             └───────────┬───────────┘                       │
│                         │                                    │
│                         ▼                                    │
│               ┌─────────────────┐                           │
│               │ Final LayerNorm │                           │
│               └────────┬────────┘                           │
│                        │                                     │
│                        ▼                                     │
│               ┌─────────────────┐                           │
│               │ Output Linear   │  Hidden → Vocabulary      │
│               └────────┬────────┘                           │
│                        │                                     │
│                        ▼                                     │
│  Output Logits: [vocab_size] per position                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Input Shape | Output Shape | Purpose |
|-----------|-------------|--------------|---------|
| Token Embedding | (batch, seq_len) | (batch, seq_len, embed_dim) | IDs → Vectors |
| Positional Encoding | (batch, seq_len, embed_dim) | (batch, seq_len, embed_dim) | Add position info |
| Transformer Blocks | (batch, seq_len, embed_dim) | (batch, seq_len, embed_dim) | Process context |
| Final LayerNorm | (batch, seq_len, embed_dim) | (batch, seq_len, embed_dim) | Normalize outputs |
| Output Linear | (batch, seq_len, embed_dim) | (batch, seq_len, vocab_size) | Predict next token |

---

## Component Assembly

### 1. Input Processing

```
Token IDs → Embedding → Add Positional → Ready for processing

Example:
  "The cat sat" → [42, 156, 891] → [[0.2, 0.1, ...], [0.5, 0.3, ...], [0.1, 0.4, ...]]
                                          + positional encoding
```

### 2. Transformer Stack

Multiple identical blocks process the input sequentially:

```python
# Pseudo-code
x = token_embedding + positional_encoding

for block in transformer_blocks:
    x = block(x)  # Each block: Attention + FFN with residuals

x = final_layer_norm(x)
```

### 3. Output Projection

The final hidden states are projected to vocabulary size to get logits:

```
Hidden state [embed_dim] → Linear → Logits [vocab_size]

Example (embed_dim=4, vocab_size=6):
[0.5, -0.2, 0.8, 0.1] → W[4×6] → [2.1, -0.5, 3.4, 0.2, -1.2, 1.8]
                                    ↓
                               [0.08, 0.01, 0.47, 0.02, 0.01, 0.41] (after softmax)
                                         ↓
                              Predicted next token: index 2 (highest prob)
```

---

## Forward Pass Walkthrough

Let's trace a complete forward pass through the model:

### Setup

```
Input: "The cat sat"
Token IDs: [42, 156, 891]
Vocab size: 2000
Embedding dim: 128
Num heads: 4
Num layers: 4
FFN hidden: 512
```

### Step 1: Token Embedding

```python
# Lookup embeddings for each token ID
embedding_table.shape = (2000, 128)

embedded = embedding_table[[42, 156, 891]]  # Fancy indexing
# Shape: (3, 128)
```

### Step 2: Add Positional Encoding

```python
# Positional encodings for positions 0, 1, 2
pos_encoding.shape = (max_seq_len, 128)

x = embedded + pos_encoding[:3]
# Shape: (3, 128)
```

### Step 3: Pass Through Transformer Blocks

```python
# Each block:
for block in [block_1, block_2, block_3, block_4]:
    # Pre-LN attention
    normed = block.attn_norm(x)
    attn_out = block.attention(normed)
    x = x + attn_out
    
    # Pre-LN FFN
    normed = block.ffn_norm(x)
    ffn_out = block.ffn(normed)
    x = x + ffn_out

# Shape remains: (3, 128)
```

### Step 4: Final Layer Norm

```python
x = final_layer_norm(x)
# Shape: (3, 128)
```

### Step 5: Output Projection

```python
# Project to vocabulary size
output_linear.weight.shape = (128, 2000)

logits = x @ output_linear.weight  # Matrix multiplication
# Shape: (3, 2000)

# Each position predicts the next token!
# Position 0 ("The") predicts what comes after "The"
# Position 1 ("cat") predicts what comes after "cat"
# Position 2 ("sat") predicts what comes after "sat"
```

---

## Numeric Example

Let's trace actual numbers through a tiny GPT:

### Mini-Model Configuration

```
Vocab size: 6
Embedding dim: 4
Num heads: 2
Num layers: 1
FFN hidden: 8
Sequence length: 3
```

### Input

```python
token_ids = [2, 4, 1]  # Three tokens
```

### Step 1: Token Embedding

```python
# Embedding table (learned):
E = [
    [ 0.1,  0.2, -0.1,  0.3],  # Token 0
    [ 0.5, -0.2,  0.4,  0.1],  # Token 1
    [-0.3,  0.4,  0.2, -0.2],  # Token 2
    [ 0.2,  0.1,  0.3,  0.4],  # Token 3
    [ 0.4, -0.1, -0.3,  0.2],  # Token 4
    [-0.1,  0.3,  0.1, -0.4],  # Token 5
]

# Look up tokens [2, 4, 1]:
x = [
    [-0.3,  0.4,  0.2, -0.2],  # Token 2
    [ 0.4, -0.1, -0.3,  0.2],  # Token 4
    [ 0.5, -0.2,  0.4,  0.1],  # Token 1
]
```

### Step 2: Add Positional Encoding

```python
# Positional encodings (sinusoidal):
PE = [
    [0.00, 1.00, 0.00, 1.00],  # Position 0
    [0.84, 0.54, 0.01, 1.00],  # Position 1
    [0.91, -0.42, 0.02, 1.00], # Position 2
]

# Add:
x = [
    [-0.3+0.00, 0.4+1.00, 0.2+0.00, -0.2+1.00],
    [0.4+0.84, -0.1+0.54, -0.3+0.01, 0.2+1.00],
    [0.5+0.91, -0.2-0.42, 0.4+0.02, 0.1+1.00],
]

x = [
    [-0.30, 1.40, 0.20, 0.80],
    [1.24, 0.44, -0.29, 1.20],
    [1.41, -0.62, 0.42, 1.10],
]
```

### Step 3: Transformer Block

```python
# Simplified - after attention, residual, FFN, residual:
x = [
    [-0.15, 1.52, 0.28, 0.95],
    [1.10, 0.38, -0.12, 1.45],
    [1.35, -0.48, 0.55, 1.22],
]
```

### Step 4: Final LayerNorm

```python
# Normalize each position:
x = [
    [-0.95, 1.08, -0.32, 0.19],
    [0.45, -0.12, -0.88, 1.55],
    [0.72, -1.32, 0.05, 0.55],
]
```

### Step 5: Output Projection

```python
# Output weight matrix (4×6):
W_out = [
    [0.2, -0.1, 0.3, 0.1, -0.2, 0.4],
    [0.1, 0.3, -0.2, 0.4, 0.1, -0.1],
    [-0.3, 0.2, 0.1, -0.2, 0.3, 0.2],
    [0.4, -0.2, 0.2, 0.3, -0.1, 0.1],
]

# Compute logits for position 0:
logits_0 = [-0.95, 1.08, -0.32, 0.19] @ W_out
         = [0.24, 0.18, -0.44, 0.02, 0.18, -0.45]

# Apply softmax to get probabilities:
probs_0 = softmax([0.24, 0.18, -0.44, 0.02, 0.18, -0.45])
        = [0.23, 0.22, 0.12, 0.18, 0.22, 0.12]

# Most likely next token: Token 0 (23%)
```

### Summary Table

| Step | Shape | Description |
|------|-------|-------------|
| Input token IDs | (3,) | [2, 4, 1] |
| Token embedding | (3, 4) | Look up vectors |
| + Positional | (3, 4) | Add position info |
| After transformer | (3, 4) | Contextualized |
| After LayerNorm | (3, 4) | Normalized |
| Output logits | (3, 6) | Token predictions |

---

## Weight Sharing

Many GPT models share weights between the token embedding and output projection layers.

### Why Weight Sharing?

1. **Parameter Efficiency**: Embedding table is large (vocab × embed_dim)
2. **Semantic Consistency**: Same space for input and output
3. **Improved Generalization**: Fewer parameters to overfit

### How It Works

```python
# Without weight sharing:
embedding_weight = np.random.randn(vocab_size, embed_dim)  # Separate
output_weight = np.random.randn(embed_dim, vocab_size)     # Separate

# With weight sharing:
embedding_weight = np.random.randn(vocab_size, embed_dim)  # Shared
output_weight = embedding_weight.T                           # Transposed!

# Embedding: (vocab_size, embed_dim)
# Output:    (embed_dim, vocab_size) = embedding.T
```

### Mathematical Interpretation

Input embedding: $e_i = W_e[i]$ (row $i$ of embedding matrix)

Output logits: $l = h \cdot W_e^T$

The logit for token $i$ is: $l_i = h \cdot e_i$ (dot product with embedding)

**Intuition**: We're measuring similarity between the hidden state and each token's embedding!

---

## Model Configuration

### GPT Configuration Class

```python
@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    
    vocab_size: int = 2000           # Vocabulary size
    embedding_dimension: int = 128    # Hidden dimension
    num_heads: int = 4                # Attention heads
    num_layers: int = 4               # Transformer blocks
    ffn_hidden_dimension: int = 512   # FFN hidden size
    max_sequence_length: int = 256    # Maximum context length
    use_weight_tying: bool = True     # Share embedding weights
```

### Model Sizes Comparison

| Model | Layers | Embed Dim | Heads | FFN Dim | Params |
|-------|--------|-----------|-------|---------|--------|
| This repo | 4 | 128 | 4 | 512 | ~1M |
| GPT-1 | 12 | 768 | 12 | 3072 | 117M |
| GPT-2 Small | 12 | 768 | 12 | 3072 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 4096 | 355M |
| GPT-2 Large | 36 | 1280 | 20 | 5120 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 6400 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |

### Parameter Count Breakdown

For this repository's configuration:

```
Token Embedding:    vocab × embed = 2000 × 128 = 256,000

Per Transformer Block:
  - Attention Q,K,V: 3 × (embed × embed) = 3 × 16,384 = 49,152
  - Attention Output: embed × embed = 16,384
  - FFN Layer 1: embed × ffn = 128 × 512 = 65,536
  - FFN Layer 2: ffn × embed = 512 × 128 = 65,536
  - LayerNorms: 2 × (2 × embed) = 2 × 256 = 512
  ─────────────────────────────────────────────
  Subtotal: ~197,120 per block

4 Transformer Blocks: 4 × 197,120 = 788,480

Final LayerNorm: 2 × 128 = 256

Output Projection: (tied with embedding, so 0 extra)

─────────────────────────────────────────────
Total: ~1,044,736 parameters (~1M)
```

---

## Code Implementation

From [src/model.py](src/model.py):

```python
class GPTModel:
    """
    Complete GPT Model implementation.
    
    Combines:
    - Token embeddings
    - Positional encodings
    - Stack of transformer blocks
    - Final layer norm
    - Output projection
    """
    
    def __init__(self, config: GPTConfig):
        self.config = config
        
        # Token embedding
        self.token_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_dimension=config.embedding_dimension
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dimension=config.embedding_dimension,
            max_sequence_length=config.max_sequence_length
        )
        
        # Transformer stack
        self.transformer_stack = TransformerStack(
            embedding_dimension=config.embedding_dimension,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_hidden_dimension=config.ffn_hidden_dimension
        )
        
        # Final layer norm (for Pre-LN architecture)
        self.final_layer_norm = LayerNorm(config.embedding_dimension)
        
        # Output projection (optionally tied with embedding)
        if config.use_weight_tying:
            # Share weights with embedding
            self.output_projection = LinearWithEmbeddingTie(
                self.token_embedding
            )
        else:
            self.output_projection = Linear(
                input_features=config.embedding_dimension,
                output_features=config.vocab_size
            )
    
    def forward(
        self,
        input_ids: np.ndarray,
        use_causal_mask: bool = True,
    ) -> np.ndarray:
        """
        Forward pass through the GPT model.
        
        Args:
            input_ids: Token indices, shape (batch, seq_len)
            use_causal_mask: Whether to use causal attention
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Step 1: Token embedding
        x = self.token_embedding.forward(input_ids)
        
        # Step 2: Add positional encoding
        seq_len = input_ids.shape[1]
        x = self.positional_encoding.forward(x, seq_len)
        
        # Step 3: Pass through transformer blocks
        x = self.transformer_stack.forward(x, use_causal_mask)
        
        # Step 4: Final layer norm
        x = self.final_layer_norm.forward(x)
        
        # Step 5: Project to vocabulary
        logits = self.output_projection.forward(x)
        
        return logits
```

### Transformer Stack

```python
class TransformerStack:
    """Stack of identical Transformer blocks."""
    
    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_dimension: int = None,
    ):
        self.layers = [
            TransformerBlock(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                ffn_hidden_dimension=ffn_hidden_dimension,
            )
            for _ in range(num_layers)
        ]
    
    def forward(
        self,
        input_tensor: np.ndarray,
        use_causal_mask: bool = True,
    ) -> np.ndarray:
        x = input_tensor
        for block in self.layers:
            x = block.forward(x, use_causal_mask)
        return x
```

---

## Visualization

### Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            GPT MODEL                                     │
│                                                                          │
│   Input: "The cat sat on"                                               │
│   Token IDs: [42, 156, 891, 73]                                         │
│                    │                                                     │
│                    ▼                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              TOKEN EMBEDDING                                     │   │
│   │   Lookup table: vocab_size × embedding_dim                       │   │
│   │   [42, 156, 891, 73] → [[e₄₂], [e₁₅₆], [e₈₉₁], [e₇₃]]           │   │
│   └─────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 │   ┌────────────────────────────┐      │
│                                 │   │  POSITIONAL ENCODING       │      │
│                                 │   │  Sinusoidal: sin/cos waves │      │
│                                 │   │  [PE₀, PE₁, PE₂, PE₃]      │      │
│                                 │   └──────────────┬─────────────┘      │
│                                 │                  │                     │
│                                 └───────[  +  ]────┘                     │
│                                           │                              │
│                                           ▼                              │
│   ╔═══════════════════════════════════════════════════════════════════╗ │
│   ║                    TRANSFORMER STACK                               ║ │
│   ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│   ║  │ Block 1: LayerNorm → Attention → + → LayerNorm → FFN → +   │  ║ │
│   ║  └─────────────────────────────────────────────────────────────┘  ║ │
│   ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│   ║  │ Block 2: LayerNorm → Attention → + → LayerNorm → FFN → +   │  ║ │
│   ║  └─────────────────────────────────────────────────────────────┘  ║ │
│   ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│   ║  │ Block 3: LayerNorm → Attention → + → LayerNorm → FFN → +   │  ║ │
│   ║  └─────────────────────────────────────────────────────────────┘  ║ │
│   ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│   ║  │ Block 4: LayerNorm → Attention → + → LayerNorm → FFN → +   │  ║ │
│   ║  └─────────────────────────────────────────────────────────────┘  ║ │
│   ╚══════════════════════════════════════════════════════════════════╝  │
│                                           │                              │
│                                           ▼                              │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     FINAL LAYER NORM                             │   │
│   │              Stabilize before output projection                  │   │
│   └─────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    OUTPUT PROJECTION                             │   │
│   │     embedding_dim → vocab_size (weight tied with embedding)     │   │
│   │     Produces logits for each position                           │   │
│   └─────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│   Output Logits: (4, vocab_size)                                        │
│   Position 0 predicts: what follows "The"                               │
│   Position 1 predicts: what follows "The cat"                           │
│   Position 2 predicts: what follows "The cat sat"                       │
│   Position 3 predicts: what follows "The cat sat on" ← Used for gen     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Through Model

```
Input IDs:  [42, 156, 891, 73]
            Shape: (batch=1, seq=4)
                    │
                    ▼
Embedded:   [[0.1, 0.2, ...], [0.3, 0.1, ...], [0.5, 0.4, ...], [0.2, 0.3, ...]]
            Shape: (1, 4, 128)
                    │
                    ▼
+ Positional: [[0.1+PE₀], [0.3+PE₁], [0.5+PE₂], [0.2+PE₃]]
            Shape: (1, 4, 128)
                    │
                    ▼
Block 1:    Attention allows tokens to "see" each other (causally)
            Each token gathers context from tokens before it
            Shape: (1, 4, 128)
                    │
                    ▼
Block 2:    Deeper patterns emerge
            Shape: (1, 4, 128)
                    │
                    ▼
Block 3:    Higher-level understanding
            Shape: (1, 4, 128)
                    │
                    ▼
Block 4:    Final contextual representations
            Shape: (1, 4, 128)
                    │
                    ▼
LayerNorm:  Normalize for stable output
            Shape: (1, 4, 128)
                    │
                    ▼
Output:     Project to vocabulary
            Shape: (1, 4, 2000)
                    │
                    ▼
Logits:     Raw scores for each token at each position
            logits[0, 3, :] = predictions for next word after "The cat sat on"
```

---

## Try It Yourself

Run the model demo:

```bash
python run_demo.py
```

Or experiment in Python:

```python
from src.model import GPTModel, GPTConfig
import numpy as np

# Create model with config
config = GPTConfig(
    vocab_size=2000,
    embedding_dimension=128,
    num_heads=4,
    num_layers=4,
    ffn_hidden_dimension=512,
    max_sequence_length=256
)

model = GPTModel(config)

# Sample input
input_ids = np.array([[42, 156, 891, 73]])  # batch=1, seq=4

# Forward pass
logits = model.forward(input_ids)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}")

# Get predictions for last position
last_logits = logits[0, -1, :]  # Shape: (vocab_size,)
predicted_token = np.argmax(last_logits)
print(f"Predicted next token ID: {predicted_token}")
```

---

## References

1. **GPT-1**: [Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised)

2. **GPT-2**: [Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models)

3. **GPT-3**: [Brown, T., et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

4. **Weight Tying**: [Press, O., & Wolf, L. (2017). Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)

5. **This Repository**: See [src/model.py](src/model.py) for `GPTConfig` and `GPTModel` classes, [src/transformer.py](src/transformer.py) for `TransformerStack`.

---

**Next Step**: The model is complete! Continue to [Training.md](Training.md) to learn how we optimize these parameters with gradient descent.
