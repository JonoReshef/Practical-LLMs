# Fine-Tuning: Adapting Models to Specific Tasks

Fine-tuning adapts a pre-trained model to new tasks or domains. We'll explore both full fine-tuning and LoRA (Low-Rank Adaptation), a parameter-efficient technique that trains only a small fraction of parameters.

![Transfer Learning Pipeline](https://www.researchgate.net/publication/385629006/figure/fig2/AS:11431281289117434@1731026104322/llustration-of-our-transfer-learning-pipeline-that-adapts-ImageNet-pre-trained.png)
_Fine-tuning leverages pre-trained knowledge and adapts the model to specific downstream tasks._

---

## Table of Contents

1. [Why Fine-Tune?](#why-fine-tune)
2. [Full Fine-Tuning](#full-fine-tuning)
3. [The Problem with Full Fine-Tuning](#the-problem-with-full-fine-tuning)
4. [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
5. [The Math Behind LoRA](#the-math-behind-lora)
6. [Numeric Example](#numeric-example)
7. [LoRA Configuration](#lora-configuration)
8. [Code Implementation](#code-implementation)
9. [Comparing Approaches](#comparing-approaches)
10. [References](#references)

---

## Why Fine-Tune?

Pre-trained models learn general language patterns, but we often want specialized behavior:

| Goal                  | Fine-Tuning Approach                                 |
| --------------------- | ---------------------------------------------------- |
| Domain adaptation     | Train on domain-specific text (medical, legal, code) |
| Task specialization   | Train on Q&A, summarization, translation pairs       |
| Style transfer        | Train on text with specific writing style            |
| Instruction following | Train on instruction-response pairs                  |

### The Transfer Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PRE-TRAINING (expensive)                             │
│                                                                          │
│  Massive text corpus → General language model                           │
│  - Wikipedia, books, web pages                                          │
│  - Billions of tokens                                                   │
│  - Weeks/months of GPU time                                             │
│  - Learn: grammar, facts, reasoning patterns                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FINE-TUNING (cheap)                                  │
│                                                                          │
│  Small task-specific data → Specialized model                           │
│  - Thousands to millions of examples                                    │
│  - Hours to days of GPU time                                            │
│  - Learn: task-specific patterns, domain vocabulary                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Full Fine-Tuning

In **full fine-tuning**, all model parameters are updated during training.

### Process

```python
# Load pre-trained model
model = load_pretrained("gpt-model.pt")

# Train on new data (all parameters updated)
for batch in task_data:
    logits = model.forward(batch.input)
    loss = compute_loss(logits, batch.target)
    gradients = model.backward(loss)

    # Update ALL parameters
    for param in model.parameters:
        param -= learning_rate * gradients[param]
```

### Visualization

```
PRE-TRAINED WEIGHTS              FINE-TUNED WEIGHTS
┌─────────────────┐              ┌─────────────────┐
│ W_embed         │    ────►     │ W_embed'        │  (changed)
│ W_attn_q        │    ────►     │ W_attn_q'       │  (changed)
│ W_attn_k        │    ────►     │ W_attn_k'       │  (changed)
│ W_attn_v        │    ────►     │ W_attn_v'       │  (changed)
│ W_ffn_1         │    ────►     │ W_ffn_1'        │  (changed)
│ W_ffn_2         │    ────►     │ W_ffn_2'        │  (changed)
│ ...             │    ────►     │ ...             │  (all changed)
└─────────────────┘              └─────────────────┘

All ~1M parameters updated
```

---

## The Problem with Full Fine-Tuning

### 1. Catastrophic Forgetting

When fine-tuning on a narrow task, the model can "forget" general capabilities:

```
Before: "What is the capital of France?" → "Paris"
After:  "What is the capital of France?" → "Q: What is... A: I don't know"
                                           (overfitted to Q&A format)
```

### 2. Storage Costs

Each fine-tuned model requires storing ALL parameters:

```
Base model:         1,000,000 parameters (100%)
Fine-tune for Task A: 1,000,000 parameters (100%) - new copy
Fine-tune for Task B: 1,000,000 parameters (100%) - new copy
Fine-tune for Task C: 1,000,000 parameters (100%) - new copy

Total storage: 4× the base model
```

### 3. Training Efficiency

Updating all parameters means:

- Large gradient tensors
- More memory for optimizer states (Adam stores 2× parameters)
- Slower training iterations

---

## LoRA: Low-Rank Adaptation

**LoRA** addresses these problems by freezing the base model and only training small "adapter" matrices.

### The Core Idea

Instead of changing weights $W$ to $W'$, learn a **low-rank update**:

$$W' = W + \Delta W = W + BA$$

Where:

- $W$: Original frozen weights $(d_{out} \times d_{in})$
- $B$: Trainable matrix $(d_{out} \times r)$
- $A$: Trainable matrix $(r \times d_{in})$
- $r$: Rank (typically 4, 8, or 16), much smaller than $d_{in}$ or $d_{out}$

### Visual Representation

```
Original weight matrix W:           LoRA decomposition:
┌─────────────────────────┐        ┌───┐   ┌─────────────────────────┐
│                         │        │   │   │                         │
│                         │        │   │ × │                         │
│     (d_out × d_in)      │   =    │ B │   │           A             │
│       e.g., 128×128     │        │   │   │                         │
│       = 16,384 params   │        │   │   │       (r × d_in)        │
│                         │        └───┘   │       e.g., 8×128       │
│                         │    (d_out×r)   │       = 1,024 params    │
│                         │    e.g., 128×8 └─────────────────────────┘
└─────────────────────────┘    = 1,024 params

W: 16,384 params (FROZEN)
B×A: 1,024 + 1,024 = 2,048 params (TRAINABLE)
Reduction: 87.5% fewer trainable parameters!
```

### Why Low-Rank?

Research shows that the weight changes during fine-tuning have **low intrinsic rank**. The model doesn't need to change arbitrarily—it needs focused updates in certain directions.

```
Full rank update:              Low-rank update:
Can change in ANY direction    Changes in FEW directions
┌─────────────────┐            ┌─────────────────┐
│← → ↑ ↓ ↗ ↘ ↙ ↖│            │  ↗       ↗     │
│               │            │    ↘   ↘       │
│ (all possible)│            │ (just a few)   │
└─────────────────┘            └─────────────────┘
```

---

## The Math Behind LoRA

### Forward Pass

For an input $x$:

$$y = xW^T + x(BA)^T \cdot \frac{\alpha}{r}$$
$$y = xW^T + xA^TB^T \cdot \frac{\alpha}{r}$$

Where $\frac{\alpha}{r}$ is a scaling factor.

### Why the Scaling Factor?

When you change rank $r$, the magnitude of $BA$ changes. The scaling factor $\frac{\alpha}{r}$ keeps the contribution consistent:

```
With rank r=4, alpha=8:  scaling = 8/4 = 2.0
With rank r=8, alpha=8:  scaling = 8/8 = 1.0
With rank r=16, alpha=8: scaling = 8/16 = 0.5
```

This lets you tune $\alpha$ once and easily experiment with different ranks.

### Initialization

**Critical insight**: We want the model to start exactly like the pre-trained model.

- Initialize $A$ with small random values (for learning signal)
- Initialize $B$ to zeros

At initialization:
$$BA = 0 \cdot A = 0$$

So the output is exactly $xW^T$, identical to the original model!

### Backward Pass (Gradients)

Given upstream gradient $\frac{\partial L}{\partial y}$:

$$\frac{\partial L}{\partial B} = \left(\frac{\partial L}{\partial y}\right)^T \cdot (xA^T) \cdot \frac{\alpha}{r}$$

$$\frac{\partial L}{\partial A} = \left(B^T \cdot \frac{\partial L}{\partial y}^T\right) \cdot x \cdot \frac{\alpha}{r}$$

Note: No gradient for $W$ (it's frozen!)

---

## Numeric Example

Let's trace LoRA through a small example:

### Setup

```
Input dimension:  4
Output dimension: 4
LoRA rank:       2
Alpha:           4
Scaling:         4/2 = 2.0
```

### Initialization

```python
# Base weights (frozen)
W = [
    [0.5, 0.2, -0.1, 0.3],
    [0.1, 0.4, 0.2, -0.2],
    [-0.3, 0.1, 0.5, 0.1],
    [0.2, -0.1, 0.3, 0.4],
]  # Shape: (4, 4)

# LoRA A (small random)
A = [
    [0.1, -0.1, 0.2, 0.15],
    [0.05, 0.12, -0.08, 0.1],
]  # Shape: (2, 4)

# LoRA B (zeros)
B = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
]  # Shape: (4, 2)
```

### Initial Forward Pass (B=0)

```python
x = [1.0, 0.5, -0.5, 0.2]  # Input vector

# Base output
y_base = x @ W.T
       = [1.0, 0.5, -0.5, 0.2] @ [[0.5, 0.1, -0.3, 0.2], ...]
       = [0.31, 0.16, -0.10, 0.38]

# LoRA output (B=0, so this is zero!)
lora_intermediate = x @ A.T  # Shape: (2,)
                  = [0.12, 0.11]  # But doesn't matter...

lora_out = lora_intermediate @ B.T  # Shape: (4,)
         = [0, 0, 0, 0]  # Because B is zeros!

# Final output
y = y_base + lora_out * 2.0  # scaling
  = [0.31, 0.16, -0.10, 0.38] + [0, 0, 0, 0]
  = [0.31, 0.16, -0.10, 0.38]

# Output unchanged from pre-trained model! ✓
```

### After Some Training (B non-zero)

```python
# After training, B has learned values
B = [
    [0.3, -0.1],
    [0.2, 0.4],
    [-0.1, 0.2],
    [0.15, 0.1],
]  # No longer zero

# Forward pass
lora_intermediate = x @ A.T = [0.12, 0.11]

lora_out = lora_intermediate @ B.T
         = [0.12, 0.11] @ [[0.3, 0.2, -0.1, 0.15], [-0.1, 0.4, 0.2, 0.1]]
         = [0.025, 0.068, 0.010, 0.029]

# Final output
y = y_base + lora_out * 2.0
  = [0.31, 0.16, -0.10, 0.38] + [0.050, 0.136, 0.020, 0.058]
  = [0.36, 0.30, -0.08, 0.44]

# Model behavior has adapted!
```

---

## LoRA Configuration

### Choosing Rank

| Rank  | Parameters | Capacity              | Use Case              |
| ----- | ---------- | --------------------- | --------------------- |
| r=1   | Minimal    | Very limited          | Simple style transfer |
| r=4   | ~0.5%      | Good for most tasks   | Default choice        |
| r=8   | ~1%        | Higher capacity       | Complex tasks         |
| r=16  | ~2%        | High capacity         | Challenging domains   |
| r=64+ | ~8%+       | Near full fine-tuning | Rarely needed         |

### Which Layers to Adapt?

LoRA can be applied to different weight matrices:

```
Common choices:
┌─────────────────────────────────────────────────────────────┐
│ Attention layers (most impactful):                          │
│   ✓ W_q (query projection)                                  │
│   ✓ W_k (key projection)                                    │
│   ✓ W_v (value projection)                                  │
│   ✓ W_o (output projection)                                 │
├─────────────────────────────────────────────────────────────┤
│ Feed-forward layers (optional):                             │
│   ○ W_ffn1 (up projection)                                  │
│   ○ W_ffn2 (down projection)                                │
├─────────────────────────────────────────────────────────────┤
│ Embedding layers (rarely):                                  │
│   ○ Token embedding                                         │
│   ○ Output projection                                       │
└─────────────────────────────────────────────────────────────┘

✓ = commonly adapted
○ = optionally adapted
```

### Learning Rate

LoRA often benefits from higher learning rates than full fine-tuning:

```
Full fine-tuning: lr = 1e-5 to 5e-5
LoRA:             lr = 1e-4 to 1e-3 (10-100× higher!)
```

---

## Code Implementation

### LoRA Linear Layer

From [src/lora.py](src/lora.py):

```python
class LoRALinear:
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    Forward computation:
        output = x @ W^T + x @ A^T @ B^T * (alpha/rank) + bias

    Where:
    - W: Frozen base weights (d_out, d_in)
    - A: Trainable down-projection (rank, d_in)
    - B: Trainable up-projection (d_out, rank) - initialized to zeros
    - scaling = alpha / rank
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
        self.input_features = input_features
        self.output_features = output_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base (frozen) weights
        if base_weights is not None:
            self.base_weights = base_weights.copy()
        else:
            weight_std = np.sqrt(2.0 / (input_features + output_features))
            self.base_weights = np.random.randn(output_features, input_features) * weight_std

        self.base_bias = base_bias.copy() if base_bias is not None else None

        # LoRA matrices (trainable)
        # A: initialized with small random values
        a_std = np.sqrt(2.0 / input_features)
        self.lora_A = np.random.randn(rank, input_features) * a_std

        # B: initialized to zeros (critical!)
        self.lora_B = np.zeros((output_features, rank))

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """Forward pass with LoRA adaptation."""
        self._input_cache = input_tensor

        # Base model contribution (frozen)
        base_output = np.matmul(input_tensor, self.base_weights.T)

        # LoRA contribution (trainable)
        # input @ A^T -> (batch, seq, rank)
        lora_intermediate = np.matmul(input_tensor, self.lora_A.T)
        self._lora_intermediate_cache = lora_intermediate

        # intermediate @ B^T -> (batch, seq, output_features)
        lora_output = np.matmul(lora_intermediate, self.lora_B.T)

        # Combine with scaling
        output = base_output + lora_output * self.scaling

        if self.base_bias is not None:
            output = output + self.base_bias

        return output

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass - only compute gradients for LoRA matrices.
        Base weights are frozen!
        """
        # ... gradient computation for A and B only
```

### Applying LoRA to a Model

```python
def apply_lora_to_model(
    model: GPTModel,
    rank: int = 4,
    alpha: float = 8.0,
    target_modules: List[str] = ["query", "key", "value"],
) -> GPTModel:
    """
    Apply LoRA adapters to a pre-trained GPT model.

    This modifies the model in-place, replacing specified Linear layers
    with LoRALinear equivalents that have frozen base weights and
    trainable LoRA parameters.

    Args:
        model: Pre-trained GPT model
        rank: LoRA rank (bottleneck dimension)
        alpha: LoRA scaling factor
        target_modules: Which attention projections to adapt

    Returns:
        Model with LoRA adapters (same object, modified)
    """
    for layer in model.transformer_stack.layers:
        attn = layer.self_attention

        # Replace attention projections with LoRA versions
        if "query" in target_modules:
            attn.query_linear = LoRALinear(
                input_features=attn.embedding_dimension,
                output_features=attn.embedding_dimension,
                rank=rank,
                alpha=alpha,
                base_weights=attn.query_linear.weights,
                base_bias=attn.query_linear.bias,
            )
        # Similarly for key, value, output...

    return model
```

### Training with LoRA

From [train_finetune_lora.py](train_finetune_lora.py):

```python
def train_step_lora(
    model: GPTModel,
    optimizer: AdamW,
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> float:
    """
    LoRA fine-tuning step.

    Key difference: Only LoRA parameters are updated!
    """
    # Forward pass
    logits = model.forward(inputs)

    # Compute loss
    loss = cross_entropy_loss(logits, targets)

    # Backward pass
    grad_logits = cross_entropy_loss_backward(logits, targets)
    model.backward(grad_logits)

    # Get ONLY LoRA gradients (base weights have no gradients)
    lora_gradients = get_lora_gradients(model)

    # Clip
    lora_gradients = clip_gradient_norm(lora_gradients, max_grad_norm)

    # Update ONLY LoRA parameters
    lora_params = get_lora_parameters(model)
    optimizer.step(lora_gradients, lora_params, learning_rate)

    return loss
```

---

## Comparing Approaches

### Parameter Counts

For this repository's model (~1M parameters):

```
Full Fine-Tuning:
  Trainable: 1,044,736 (100%)
  Storage per adapter: 1,044,736 × 4 bytes = 4.2 MB

LoRA (rank=4):
  Trainable: ~8,192 (0.8%)
  Storage per adapter: 8,192 × 4 bytes = 33 KB

Savings: ~128× fewer parameters per adapter!
```

### Performance Comparison

| Metric                 | Full Fine-Tuning | LoRA      |
| ---------------------- | ---------------- | --------- |
| Parameters trained     | 100%             | 0.5-2%    |
| Memory during training | High             | Low       |
| Training speed         | Slower           | Faster    |
| Risk of forgetting     | Higher           | Lower     |
| Adapter storage        | Large            | Tiny      |
| Performance            | Excellent        | Very Good |

### When to Use Each

**Use Full Fine-Tuning when:**

- Unlimited compute and storage
- Task is very different from pretraining
- Maximum performance is critical
- Only one specialized model needed

**Use LoRA when:**

- Limited compute/memory
- Multiple tasks/domains
- Quick experimentation
- Preserving base capabilities matters
- Deploying many specialized models

---

## Merging LoRA Weights

For inference, LoRA weights can be **merged** into base weights:

$$W_{merged} = W + BA \cdot \frac{\alpha}{r}$$

After merging:

- No inference overhead
- Single weight matrix
- Can't easily swap adapters

```python
def merge_lora_weights(lora_layer: LoRALinear) -> np.ndarray:
    """Merge LoRA weights into base weights."""
    delta_W = lora_layer.lora_B @ lora_layer.lora_A
    merged = lora_layer.base_weights + delta_W * lora_layer.scaling
    return merged
```

---

## Try It Yourself

### Prerequisites

First, train a base model:

```bash
python train_pretrain.py
```

### Run LoRA Fine-Tuning

```bash
python train_finetune_lora.py
```

### Experiment with Different Ranks

```python
from src.lora import apply_lora_to_model
from src.model import GPTModel

# Load base model
model = GPTModel.load("checkpoints/model_best.npz")

# Try rank 4 (default)
model_r4 = apply_lora_to_model(model.copy(), rank=4)
print(f"Rank 4: {count_lora_parameters(model_r4):,} trainable")

# Try rank 8
model_r8 = apply_lora_to_model(model.copy(), rank=8)
print(f"Rank 8: {count_lora_parameters(model_r8):,} trainable")

# Try rank 16
model_r16 = apply_lora_to_model(model.copy(), rank=16)
print(f"Rank 16: {count_lora_parameters(model_r16):,} trainable")
```

---

## References

1. **LoRA Paper**: [Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

2. **QLoRA**: [Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

3. **LoRA Analysis**: [Biderman, S., et al. (2024). LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)

4. **Full Fine-Tuning vs Adapters**: [He, J., et al. (2021). Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)

5. **This Repository**: See [src/lora.py](src/lora.py) for LoRA implementation, [train_finetune_lora.py](train_finetune_lora.py) for training script.

---

**Congratulations!** You've completed the learning path for understanding GPT language models from the ground up. From tokenization to generation to fine-tuning, you now understand how modern language models work at a fundamental level.

**What's Next?**

- Experiment with the code in this repository
- Try training on different datasets
- Implement your own extensions (e.g., top-p sampling, rotary embeddings)
- Scale up to larger models on GPU with frameworks like PyTorch
