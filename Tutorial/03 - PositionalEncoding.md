# Positional Encoding: Teaching the Model Word Order

Transformers process all tokens in parallel, which makes them fast but creates a problem: how does the model know that "dog bites man" is different from "man bites dog"? The answer is **positional encoding**.

---

## Table of Contents

1. [The Problem: Position Blindness](#the-problem-position-blindness)
2. [Sinusoidal Positional Encoding](#sinusoidal-positional-encoding)
3. [The Mathematical Formula](#the-mathematical-formula)
4. [Step-by-Step Numeric Example](#step-by-step-numeric-example)
5. [Why Sines and Cosines?](#why-sines-and-cosines)
6. [Code Implementation](#code-implementation)
7. [Visualization](#visualization)
8. [Alternative Approaches](#alternative-approaches)
9. [References](#references)

---

## The Problem: Position Blindness

Consider these two sentences:

```
Sentence 1: "Dog bites man"
Sentence 2: "Man bites dog"
```

After tokenization and embedding (without positional encoding):

```
Sentence 1: [E("dog"), E("bites"), E("man")]
Sentence 2: [E("man"), E("bites"), E("dog")]
```

Where `E(x)` is the embedding for token x.

The **problem**: These sequences contain the exact same embeddings, just in different order. Without position information, a model using only attention would see these as the same!

### Why This Happens

Unlike RNNs that process tokens sequentially (and thus naturally encode position), self-attention computes relationships between all tokens simultaneously:

```
Self-Attention sees:
  - "dog" relates to "bites" ✓
  - "bites" relates to "man" ✓
  - "dog" relates to "man" ✓

But NOT:
  - "dog" is at position 0
  - "bites" is at position 1
  - "man" is at position 2
```

---

## Sinusoidal Positional Encoding

The solution from "Attention Is All You Need" is to **add** a position-dependent signal to each embedding:

```
final_embedding = token_embedding + positional_encoding
```

This creates unique patterns for each position using sine and cosine functions at different frequencies.

### Visual Intuition

Think of it like telling time with two clock hands:

- **Hour hand** (slow frequency): Changes slowly across positions
- **Minute hand** (fast frequency): Changes quickly across positions

Together, they uniquely identify each position.

---

## The Mathematical Formula

For a position `pos` in the sequence and dimension `i` of the embedding:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:

- $pos$ = position in sequence (0, 1, 2, ...)
- $i$ = dimension index (0, 1, 2, ..., $d_{model}/2 - 1$)
- $d_{model}$ = embedding dimension
- 10000 is a scaling constant

### Breaking Down the Formula

1. **Even dimensions (2i)**: Use sine
2. **Odd dimensions (2i+1)**: Use cosine
3. **Wavelength varies**: $10000^{2i/d_{model}}$ creates wavelengths from $2\pi$ to $10000 \cdot 2\pi$

---

## Step-by-Step Numeric Example

Let's compute positional encodings for:

- Sequence length: 4 positions
- Embedding dimension: 8

### Step 1: Calculate Frequency Divisors

For each dimension pair $(2i, 2i+1)$:

| i   | 2i (sin) | 2i+1 (cos) | Divisor = $10000^{2i/8}$ |
| --- | -------- | ---------- | ------------------------ |
| 0   | 0        | 1          | $10000^0 = 1$            |
| 1   | 2        | 3          | $10000^{0.25} = 10$      |
| 2   | 4        | 5          | $10000^{0.5} = 100$      |
| 3   | 6        | 7          | $10000^{0.75} = 1000$    |

### Step 2: Calculate Angles

Angle = $pos / divisor$

| Position | dim 0,1 (÷1) | dim 2,3 (÷10) | dim 4,5 (÷100) | dim 6,7 (÷1000) |
| -------- | ------------ | ------------- | -------------- | --------------- |
| pos=0    | 0.0          | 0.0           | 0.0            | 0.0             |
| pos=1    | 1.0          | 0.1           | 0.01           | 0.001           |
| pos=2    | 2.0          | 0.2           | 0.02           | 0.002           |
| pos=3    | 3.0          | 0.3           | 0.03           | 0.003           |

### Step 3: Apply Sin/Cos

| Position | sin(θ₀) | cos(θ₀) | sin(θ₁) | cos(θ₁) | sin(θ₂) | cos(θ₂) | sin(θ₃) | cos(θ₃) |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 0        | 0.000   | 1.000   | 0.000   | 1.000   | 0.000   | 1.000   | 0.000   | 1.000   |
| 1        | 0.841   | 0.540   | 0.100   | 0.995   | 0.010   | 1.000   | 0.001   | 1.000   |
| 2        | 0.909   | -0.416  | 0.199   | 0.980   | 0.020   | 1.000   | 0.002   | 1.000   |
| 3        | 0.141   | -0.990  | 0.296   | 0.955   | 0.030   | 1.000   | 0.003   | 1.000   |

### Step 4: Final Positional Encoding Matrix

```
PE = [
  # pos 0: [0.000,  1.000,  0.000,  1.000,  0.000,  1.000,  0.000,  1.000]
  # pos 1: [0.841,  0.540,  0.100,  0.995,  0.010,  1.000,  0.001,  1.000]
  # pos 2: [0.909, -0.416,  0.199,  0.980,  0.020,  1.000,  0.002,  1.000]
  # pos 3: [0.141, -0.990,  0.296,  0.955,  0.030,  1.000,  0.003,  1.000]
]
```

### Step 5: Add to Token Embeddings

```python
# Token embeddings for "dog bites man" (made up values)
token_embeddings = [
    [0.5, 0.3, 0.8, 0.1, 0.4, 0.6, 0.2, 0.9],  # "dog"
    [0.2, 0.7, 0.4, 0.9, 0.1, 0.3, 0.8, 0.5],  # "bites"
    [0.9, 0.1, 0.6, 0.3, 0.7, 0.5, 0.4, 0.2],  # "man"
]

# Add positional encoding
final_embeddings = [
    # "dog" at pos 0
    [0.5+0.000, 0.3+1.000, 0.8+0.000, 0.1+1.000, ...],
    # = [0.500, 1.300, 0.800, 1.100, ...]

    # "bites" at pos 1
    [0.2+0.841, 0.7+0.540, 0.4+0.100, 0.9+0.995, ...],
    # = [1.041, 1.240, 0.500, 1.895, ...]

    # "man" at pos 2
    [0.9+0.909, 0.1-0.416, 0.6+0.199, 0.3+0.980, ...],
    # = [1.809, -0.316, 0.799, 1.280, ...]
]
```

Now "dog" at position 0 has different values than "dog" would have at position 2!

---

## Why Sines and Cosines?

### 1. Unique Patterns for Each Position

The combination of many frequencies creates a unique "fingerprint" for each position:

```
Position 0: [0.00, 1.00, 0.00, 1.00, 0.00, 1.00, ...]  All cosines = 1
Position 1: [0.84, 0.54, 0.10, 0.99, 0.01, 1.00, ...]  Distinct pattern
Position 2: [0.91, -0.42, 0.20, 0.98, 0.02, 1.00, ...] Distinct pattern
```

### 2. Relative Position Through Linear Transformation

A key property of sinusoidal encoding is that for any offset $k$:

$$PE_{pos+k}$$ can be represented as a linear function of $$PE_{pos}$$

Using trigonometric identities:
$$\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$$
$$\cos(pos + k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)$$

This means the model can learn to "look back" or "look ahead" by $k$ positions through a simple matrix multiplication. The attention mechanism can exploit this to focus on relative positions.

### 3. Bounded Values

Unlike learned positions or simple counters:

- Values stay in range $[-1, 1]$
- No explosion for long sequences
- Smooth gradients for training

### 4. Generalization Beyond Training Length

Sinusoidal encoding works for sequences longer than seen during training because the mathematical function is defined for any position:

```python
# Trained on sequences up to length 512
# Can still encode position 1000:
PE[1000] = compute_sinusoidal(1000)  # Just plug in the formula!
```

---

## Code Implementation

From [src/layers.py](src/layers.py), here's the `PositionalEncoding` class:

```python
class PositionalEncoding:
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need".

    Creates deterministic position-dependent patterns using sine and cosine
    functions at different frequencies. These are added to token embeddings
    to give the model information about token positions.

    Attributes:
        max_sequence_length: Maximum sequence length supported
        embedding_dimension: Dimension of embeddings (must match model)
        encoding_table: Precomputed (max_len, embed_dim) encoding matrix
    """

    def __init__(self, max_sequence_length: int, embedding_dimension: int):
        """
        Initialize by precomputing encodings for all positions.

        Args:
            max_sequence_length: Max positions to precompute
            embedding_dimension: Size of embedding vectors
        """
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension

        # Precompute the full encoding table
        self.encoding_table = self._compute_encoding()

    def _compute_encoding(self) -> np.ndarray:
        """
        Compute sinusoidal positional encodings.

        Returns:
            Array of shape (max_sequence_length, embedding_dimension)
        """
        # Create position indices: [0, 1, 2, ..., max_len-1]
        positions = np.arange(self.max_sequence_length)[:, np.newaxis]

        # Create dimension indices for the frequency calculation
        # dim_indices = [0, 1, 2, ..., embed_dim/2 - 1]
        dim_indices = np.arange(0, self.embedding_dimension, 2)

        # Compute the frequency divisors: 10000^(2i/d_model)
        # Using log for numerical stability: 10000^x = exp(x * log(10000))
        divisors = np.exp(dim_indices * (-np.log(10000.0) / self.embedding_dimension))

        # Compute angles for all positions and dimensions
        angles = positions * divisors  # Shape: (max_len, embed_dim/2)

        # Initialize encoding table
        encoding = np.zeros((self.max_sequence_length, self.embedding_dimension))

        # Apply sin to even indices, cos to odd indices
        encoding[:, 0::2] = np.sin(angles)  # Even dimensions
        encoding[:, 1::2] = np.cos(angles)  # Odd dimensions

        return encoding

    def get_encoding(self, sequence_length: int) -> np.ndarray:
        """
        Get positional encoding for a sequence of given length.

        Args:
            sequence_length: Number of positions needed

        Returns:
            Array of shape (sequence_length, embedding_dimension)
        """
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum "
                f"{self.max_sequence_length}"
            )
        return self.encoding_table[:sequence_length]
```

### Usage Example

```python
import numpy as np
from src.layers import Embedding, PositionalEncoding

# Create components
vocab_size = 1000
embed_dim = 128
max_seq_len = 512

embedding = Embedding(vocab_size, embed_dim)
pos_encoding = PositionalEncoding(max_seq_len, embed_dim)

# Input tokens
token_ids = np.array([[42, 156, 89]])  # Shape: (1, 3)

# Get token embeddings
token_embeds = embedding.forward(token_ids)  # Shape: (1, 3, 128)

# Get positional encodings
pos_encodes = pos_encoding.get_encoding(3)  # Shape: (3, 128)

# Add them together (broadcasting adds to each batch)
final_embeds = token_embeds + pos_encodes  # Shape: (1, 3, 128)

print(f"Token embedding range: [{token_embeds.min():.3f}, {token_embeds.max():.3f}]")
print(f"Position encoding range: [{pos_encodes.min():.3f}, {pos_encodes.max():.3f}]")
```

---

## Visualization

### Sinusoidal Patterns Across Positions

```
Dimension 0 (high frequency):    Dimension 126 (low frequency):
sin wave, period ≈ 2π            sin wave, period ≈ 10000×2π

Position ──────────────────►     Position ──────────────────►
     ▲                                ▲
   1 │    ╭─╮    ╭─╮    ╭─         1 │──────────────────────────
     │   ╱   ╲  ╱   ╲  ╱              │
   0 │──╱─────╲╱─────╲╱──          0 │
     │ ╱       ╲       ╲              │
  -1 │╱    ╰───╯    ╰───           -1 │
     └───────────────────►           └────────────────────────►
       0  1  2  3  4  5              0   100   200   300   400

Fast oscillation for              Slow change for long-range
nearby positions                  position differences
```

### Heatmap of Position Encodings

The following visualization from [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) shows positional encodings for 20 positions (rows) with 512 dimensions (columns):

![Positional Encoding Heatmap](https://jalammar.github.io/images/t/attention-is-all-you-need-positional-encoding.png)

_Image credit: Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)_

**How to read this heatmap:**

- **Each row** = one position in the sequence (position 0 at top, position 19 at bottom)
- **Each column** = one dimension of the encoding vector (dim 0 on left, dim 511 on right)
- **Colors** = values between -1 (one color) and +1 (opposite color)

**Key observations:**

- **Left side (low dimensions)**: Rapid color changes as you move down rows - these are the **fast frequencies** that cycle every few positions
- **Right side (high dimensions)**: Colors stay nearly constant across all rows - these are the **slow frequencies** that barely change over 20 positions
- The pattern creates a unique "fingerprint" for each position

**Why the difference?**

- Dimension 0 uses divisor = 1, so the angle increases by 1.0 per position (completes a full sine cycle in ~6 positions)
- Dimension 510 uses divisor ≈ 10000, so the angle increases by only 0.0001 per position (would need ~60,000 positions to complete one cycle)

### Position Similarity Matrix

```
              Position 0  1  2  3  4  5  6  7
Position 0       1.00 .76 .41 .04 -.28 -.52 -.67 -.73
Position 1        .76 1.00 .76 .41 .04 -.28 -.52 -.67
Position 2        .41 .76 1.00 .76 .41 .04 -.28 -.52
Position 3        .04 .41 .76 1.00 .76 .41 .04 -.28
Position 4       -.28 .04 .41 .76 1.00 .76 .41 .04
Position 5       -.52 -.28 .04 .41 .76 1.00 .76 .41
Position 6       -.67 -.52 -.28 .04 .41 .76 1.00 .76
Position 7       -.73 -.67 -.52 -.28 .04 .41 .76 1.00

Observation: Nearby positions have higher similarity,
            distant positions have lower similarity.
```

---

## Alternative Approaches

### 1. Learned Positional Embeddings

Instead of fixed sinusoidal encoding, learn position embeddings:

```python
# Learnable position embedding table (like BERT, GPT-2)
position_embedding = np.random.randn(max_seq_len, embed_dim) * 0.02

# Lookup by position
pos_embed = position_embedding[position]
```

**Pros**: Can learn task-specific patterns
**Cons**: Can't extrapolate beyond max_seq_len

### 2. Rotary Position Embedding (RoPE)

Used in LLaMA, GPT-Neo-X, and modern models:

```python
# Rotate queries and keys based on position
def apply_rope(x, position):
    # Apply rotation matrix based on position
    cos = compute_cos(position)
    sin = compute_sin(position)
    return x * cos + rotate(x) * sin
```

**Pros**: Naturally encodes relative positions, infinite extrapolation
**Cons**: More complex to implement

### 3. ALiBi (Attention with Linear Biases)

Used in BLOOM and other models:

```python
# Add position-based bias directly to attention scores
attention_bias = -m * |pos_query - pos_key|  # m is head-specific slope
attention_scores += attention_bias
```

**Pros**: No changes to embeddings, natural length generalization
**Cons**: Only affects attention, not embedding representation

### Comparison

| Method     | Length Extrapolation | Computational Cost | Used In              |
| ---------- | -------------------- | ------------------ | -------------------- |
| Sinusoidal | Good                 | Low (precomputed)  | Original Transformer |
| Learned    | Poor                 | Low                | BERT, GPT-2          |
| RoPE       | Excellent            | Medium             | LLaMA, GPT-J         |
| ALiBi      | Excellent            | Low                | BLOOM                |

---

## Try It Yourself

Run the layers demo to see positional encoding:

```bash
python -m src.layers
```

Or experiment in Python:

```python
from src.layers import PositionalEncoding
import numpy as np

# Create positional encoding
pe = PositionalEncoding(max_sequence_length=512, embedding_dimension=128)

# Get encodings
encodings = pe.get_encoding(10)  # For positions 0-9
print(f"Shape: {encodings.shape}")  # (10, 128)

# Check properties
print(f"Position 0: {encodings[0, :4]}...")
print(f"Position 1: {encodings[1, :4]}...")

# Compute similarity between positions
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Similarity pos0-pos1: {cosine_sim(encodings[0], encodings[1]):.3f}")
print(f"Similarity pos0-pos5: {cosine_sim(encodings[0], encodings[5]):.3f}")
print(f"Similarity pos0-pos9: {cosine_sim(encodings[0], encodings[9]):.3f}")
# Note: closer positions are more similar
```

---

## References

1. **Original Paper**: [Vaswani, A., et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.5

2. **Position Encoding Deep Dive**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar

3. **RoPE Paper**: [Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

4. **ALiBi Paper**: [Press, O., et al. (2022). Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409)

5. **This Repository**: See [src/layers.py](src/layers.py) for the `PositionalEncoding` class implementation.

---

**Next Step**: Now that embeddings have position information, the model needs to understand how tokens relate to each other. Continue to [Attention.md](Attention.md) to learn about the attention mechanism.
