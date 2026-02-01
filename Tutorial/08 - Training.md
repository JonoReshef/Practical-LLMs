# Training: Optimizing the Language Model

Training a language model involves computing the difference between predictions and targets (loss), calculating how to adjust weights (gradients), and updating parameters to minimize the loss.

---

## Table of Contents

1. [Language Modeling Objective](#language-modeling-objective)
2. [Cross-Entropy Loss](#cross-entropy-loss)
3. [Backpropagation](#backpropagation)
4. [AdamW Optimizer](#adamw-optimizer)
5. [Learning Rate Scheduling](#learning-rate-scheduling)
6. [Gradient Clipping](#gradient-clipping)
7. [Complete Training Step](#complete-training-step)
8. [Numeric Example](#numeric-example)
9. [Code Implementation](#code-implementation)
10. [References](#references)

---

## Language Modeling Objective

GPT uses **causal language modeling**: predict the next token given all previous tokens.

### Training Data Format

```
Input:     "The cat sat on the"
Target:    "cat sat on the mat"

Position 0: Given "The"         → Predict "cat"
Position 1: Given "The cat"     → Predict "sat"
Position 2: Given "The cat sat" → Predict "on"
...
```

### The Training Signal

```
Model output: logits[batch, seq_len, vocab_size]
              ↓
Target:       one-hot encoded true next tokens
              ↓
Loss:         How wrong were we? (cross-entropy)
              ↓
Gradient:     Which direction to adjust weights?
              ↓
Update:       Move weights to reduce loss
```

---

## Cross-Entropy Loss

**Cross-entropy loss** measures how different the predicted probability distribution is from the true distribution.

### The Formula

For a single prediction:
$$L = -\sum_{i=1}^{V} y_i \log(\hat{p}_i)$$

Where:

- $V$ = vocabulary size
- $y_i$ = 1 if $i$ is the correct token, 0 otherwise (one-hot)
- $\hat{p}_i$ = predicted probability for token $i$

Since $y$ is one-hot, this simplifies to:
$$L = -\log(\hat{p}_{\text{correct}})$$

### Intuition

| Predicted probability for correct token | Loss                               |
| --------------------------------------- | ---------------------------------- |
| 0.99                                    | $-\log(0.99) = 0.01$ (very good!)  |
| 0.50                                    | $-\log(0.50) = 0.69$ (uncertain)   |
| 0.10                                    | $-\log(0.10) = 2.30$ (poor)        |
| 0.01                                    | $-\log(0.01) = 4.61$ (very wrong!) |

### From Logits to Loss

```
Step 1: Raw logits from model
        logits = [2.1, -0.5, 3.4, 0.2, -1.2, 1.8]

Step 2: Apply softmax (converts to probabilities)
        exp_logits = [8.17, 0.61, 29.96, 1.22, 0.30, 6.05]
        sum_exp = 46.31
        probs = [0.18, 0.01, 0.65, 0.03, 0.01, 0.13]

Step 3: If correct token is index 2
        p_correct = 0.65
        loss = -log(0.65) = 0.43
```

### Numeric Stability: LogSoftmax

Computing softmax then log can cause numerical issues. Instead, use log-softmax:

$$\log(\text{softmax}(x))_i = x_i - \log\left(\sum_j e^{x_j}\right)$$

With the **log-sum-exp trick** for stability:
$$\log\left(\sum_j e^{x_j}\right) = m + \log\left(\sum_j e^{x_j - m}\right)$$

where $m = \max(x)$

```python
# Stable implementation
def log_softmax(logits):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logit  # Prevents overflow
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp
```

---

## Backpropagation

**Backpropagation** computes gradients by applying the chain rule backwards through the network.

### The Chain Rule

If $L$ is the loss and $w$ is a weight deep in the network:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial w}$$

### Gradient Flow Through GPT

```
                Forward                          Backward
                ───────►                         ◄───────

Loss L ◄─────── Output Linear ◄─────── LayerNorm ◄─────── Transformer Stack
         ∂L/∂logits          ∂L/∂h_norm         ∂L/∂h_trans

─►dL/d(logits)──►dL/d(h)────►dL/d(blocks)────►dL/d(embeddings)
```

### Cross-Entropy Gradient

The gradient of cross-entropy loss with respect to logits is elegantly simple:

$$\frac{\partial L}{\partial z_i} = \hat{p}_i - y_i$$

In other words: `gradient = predicted_probs - one_hot_target`

```python
# Example
predicted = [0.18, 0.01, 0.65, 0.03, 0.01, 0.13]  # softmax output
target    = [0,    0,    1,    0,    0,    0   ]  # one-hot
gradient  = [0.18, 0.01, -0.35, 0.03, 0.01, 0.13] # pred - target
```

**Intuition**:

- Correct token (index 2): gradient is negative → push logit up
- Wrong tokens: gradient is positive → push logits down

---

## AdamW Optimizer

**AdamW** (Adam with decoupled Weight decay) is the standard optimizer for transformers.

### Why Not Plain SGD?

| Issue             | SGD Problem         | Adam Solution                  |
| ----------------- | ------------------- | ------------------------------ |
| Learning rate     | Same for all params | Adaptive per-parameter         |
| Gradient noise    | High variance       | Momentum smoothing             |
| Saddle points     | Can get stuck       | Momentum carries through       |
| Scale sensitivity | Problematic         | Normalized by gradient history |

### Adam Components

1. **First Moment (m)**: Exponential moving average of gradients
   $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

2. **Second Moment (v)**: Exponential moving average of squared gradients
   $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

3. **Bias Correction**: Compensate for initialization at zero
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
   $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

4. **Update Rule**:
   $$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Weight Decay in AdamW

Original Adam applies weight decay inside the gradient:
$$g_t = \nabla L + \lambda \theta_{t-1}$$ (L2 regularization)

AdamW **decouples** weight decay:
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \cdot \lambda \cdot \theta_{t-1}$$

### Typical Hyperparameters

| Parameter     | Symbol     | Typical Value | Description            |
| ------------- | ---------- | ------------- | ---------------------- |
| Learning rate | $\alpha$   | 3e-4          | Step size              |
| Beta1         | $\beta_1$  | 0.9           | Momentum decay         |
| Beta2         | $\beta_2$  | 0.999         | Squared gradient decay |
| Epsilon       | $\epsilon$ | 1e-8          | Numerical stability    |
| Weight decay  | $\lambda$  | 0.01          | Regularization         |

---

## Learning Rate Scheduling

A fixed learning rate is rarely optimal. Modern training uses **schedules**.

### Warmup + Cosine Decay

```
LR
│
│    ┌──────────────╮
│   ╱                ╲
│  ╱                  ╲
│ ╱                    ╲
│╱                      ╲
├─────────────────────────────► Steps
  Warmup    Peak    Cosine Decay
```

### Warmup Phase

Start with small learning rate and linearly increase:

$$\text{lr}_t = \text{lr}_{\text{max}} \cdot \frac{t}{\text{warmup\_steps}}$$

**Why warmup?**

- Early gradients are noisy (random weights)
- Large updates early can destabilize training
- Optimizer statistics (m, v) need time to stabilize

### Cosine Decay Phase

After warmup, decay the learning rate following a cosine curve:

$$\text{lr}_t = \text{lr}_{\text{min}} + \frac{1}{2}(\text{lr}_{\text{max}} - \text{lr}_{\text{min}})\left(1 + \cos\left(\frac{t - t_w}{T - t_w}\pi\right)\right)$$

Where:

- $t_w$ = warmup steps
- $T$ = total steps
- $\text{lr}_{\text{min}}$ often set to 0 or 1e-5

### Numeric Example

```python
# Configuration
max_lr = 3e-4
warmup_steps = 100
total_steps = 1000

# During warmup (step 50):
lr = 3e-4 * (50 / 100) = 1.5e-4

# After warmup (step 500):
progress = (500 - 100) / (1000 - 100) = 0.44
lr = 3e-4 * 0.5 * (1 + cos(0.44 * π))
   = 3e-4 * 0.5 * (1 + 0.17)
   = 1.75e-4

# Near end (step 900):
progress = (900 - 100) / (1000 - 100) = 0.89
lr = 3e-4 * 0.5 * (1 + cos(0.89 * π))
   = 3e-4 * 0.5 * (1 + (-0.93))
   = 1.1e-5
```

---

## Gradient Clipping

**Gradient clipping** prevents exploding gradients by limiting their magnitude.

### The Problem

Deep networks can have gradient magnitudes vary wildly:

```
Normal:    gradient norm = 0.5   → reasonable update
Exploding: gradient norm = 1000  → catastrophic update!
```

### Max Norm Clipping

If the total gradient norm exceeds a threshold, scale all gradients down:

$$\text{if } \|g\|_2 > \text{max\_norm}: \quad g = g \cdot \frac{\text{max\_norm}}{\|g\|_2}$$

```python
def clip_gradient_norm(gradients, max_norm=1.0):
    # Compute total gradient norm
    total_norm = 0.0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    # Scale if necessary
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for key in gradients:
            gradients[key] *= scale

    return gradients
```

### Typical Value

Most transformer training uses `max_norm = 1.0`

---

## Complete Training Step

Here's the full picture of what happens in one training iteration:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SINGLE TRAINING STEP                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. FORWARD PASS                                                        │
│     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│     │ Input IDs   │────►│ GPT Model   │────►│   Logits    │            │
│     │ (batch, seq)│     │  forward()  │     │(batch,seq,V)│            │
│     └─────────────┘     └─────────────┘     └──────┬──────┘            │
│                                                     │                   │
│  2. COMPUTE LOSS                                    │                   │
│     ┌─────────────┐                                │                   │
│     │  Targets    │─────────────────────────────────┘                   │
│     │ (batch, seq)│              │                                      │
│     └─────────────┘              ▼                                      │
│                         ┌─────────────────┐                             │
│                         │ Cross-Entropy   │                             │
│                         │     Loss        │                             │
│                         └────────┬────────┘                             │
│                                  │ scalar loss value                    │
│                                  │                                      │
│  3. BACKWARD PASS                ▼                                      │
│                         ┌─────────────────┐                             │
│                         │ Loss Gradient   │                             │
│                         │ (dL/d_logits)   │                             │
│                         └────────┬────────┘                             │
│                                  │                                      │
│                                  ▼                                      │
│                         ┌─────────────────┐                             │
│                         │  model.backward │─────► gradients dict        │
│                         │  (chain rule)   │       {param_name: grad}    │
│                         └─────────────────┘                             │
│                                                          │              │
│  4. GRADIENT CLIPPING                                    ▼              │
│                                               ┌─────────────────┐       │
│                                               │ clip_grad_norm  │       │
│                                               │ (max_norm=1.0)  │       │
│                                               └────────┬────────┘       │
│                                                        │                │
│  5. PARAMETER UPDATE                                   ▼                │
│                                               ┌─────────────────┐       │
│                                               │ AdamW.step()    │       │
│                                               │ - Update m, v   │       │
│                                               │ - Bias correct  │       │
│                                               │ - Update params │       │
│                                               │ - Weight decay  │       │
│                                               └─────────────────┘       │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Numeric Example

Let's trace through a complete training step with actual numbers.

### Setup

```
Vocabulary size: 4
Embedding dimension: 2
Batch size: 1
Sequence length: 2
```

### Forward Pass

```python
# Input
input_ids = [[0, 2]]  # "The cat"
targets = [[2, 1]]    # "cat sat"

# After forward pass (simplified)
logits = [
    [[1.5, 0.2, 2.1, -0.5],   # Position 0 predictions
     [0.8, 2.3, 0.1, -0.2]]   # Position 1 predictions
]
```

### Compute Loss

```python
# Position 0: target is token 2
logits_0 = [1.5, 0.2, 2.1, -0.5]
probs_0 = softmax(logits_0) = [0.31, 0.08, 0.57, 0.04]
loss_0 = -log(0.57) = 0.56

# Position 1: target is token 1
logits_1 = [0.8, 2.3, 0.1, -0.2]
probs_1 = softmax(logits_1) = [0.16, 0.72, 0.08, 0.06]
loss_1 = -log(0.72) = 0.33

# Average loss
total_loss = (0.56 + 0.33) / 2 = 0.45
```

### Compute Gradients

```python
# Gradient at logits (pred - target)
grad_logits_0 = probs_0 - one_hot(2)
             = [0.31, 0.08, 0.57, 0.04] - [0, 0, 1, 0]
             = [0.31, 0.08, -0.43, 0.04]

grad_logits_1 = probs_1 - one_hot(1)
             = [0.16, 0.72, 0.08, 0.06] - [0, 1, 0, 0]
             = [0.16, -0.28, 0.08, 0.06]
```

### AdamW Update (for one parameter)

```python
# Suppose we have output weight w = 0.5
# Its gradient g = 0.2

# Optimizer state (initialized)
m = 0  # First moment
v = 0  # Second moment
t = 1  # Step

# Update moments
beta1, beta2 = 0.9, 0.999
m = 0.9 * 0 + 0.1 * 0.2 = 0.02
v = 0.999 * 0 + 0.001 * 0.2² = 0.00004

# Bias correction
m_hat = 0.02 / (1 - 0.9¹) = 0.02 / 0.1 = 0.2
v_hat = 0.00004 / (1 - 0.999¹) = 0.00004 / 0.001 = 0.04

# Update (without weight decay)
lr = 3e-4
epsilon = 1e-8
update = lr * m_hat / (sqrt(v_hat) + epsilon)
       = 3e-4 * 0.2 / (0.2 + 1e-8)
       = 3e-4

# With weight decay (lambda = 0.01)
weight_decay_term = lr * 0.01 * 0.5 = 1.5e-6

# Final update
w_new = 0.5 - 3e-4 - 1.5e-6 ≈ 0.4997
```

---

## Code Implementation

### Cross-Entropy Loss

From [src/model.py](src/model.py):

```python
def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: Model predictions, shape (batch, seq_len, vocab_size)
        targets: Target token IDs, shape (batch, seq_len)

    Returns:
        Scalar loss value (average over all positions)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for easier indexing
    logits_flat = logits.reshape(-1, vocab_size)  # (batch*seq, vocab)
    targets_flat = targets.reshape(-1)             # (batch*seq,)

    # Compute log probabilities (numerically stable)
    max_logits = np.max(logits_flat, axis=-1, keepdims=True)
    shifted_logits = logits_flat - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=-1))
    log_probs = shifted_logits[np.arange(len(targets_flat)), targets_flat] - log_sum_exp

    # Average negative log probability
    loss = -np.mean(log_probs)

    return loss


def cross_entropy_loss_backward(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute gradient of cross-entropy loss with respect to logits.

    The gradient is simply: softmax(logits) - one_hot(targets)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Compute softmax probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Subtract 1 from correct class probabilities
    # This is equivalent to: probs - one_hot(targets)
    grad = probs.copy()
    batch_indices = np.arange(batch_size)[:, None]
    seq_indices = np.arange(seq_len)[None, :]
    grad[batch_indices, seq_indices, targets] -= 1.0

    # Average over batch and sequence
    grad = grad / (batch_size * seq_len)

    return grad
```

### AdamW Optimizer

From [src/optimizer.py](src/optimizer.py):

```python
class AdamW:
    """
    AdamW optimizer with decoupled weight decay.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step

    def initialize(self, parameters: Dict[str, np.ndarray]):
        """Initialize optimizer state for each parameter."""
        for name, param in parameters.items():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def step(
        self,
        gradients: Dict[str, np.ndarray],
        parameters: Dict[str, np.ndarray],
        learning_rate: float = None,
    ):
        """
        Perform one optimization step.
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        self.t += 1

        for name, param in parameters.items():
            if name not in gradients:
                continue

            grad = gradients[name]

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Compute update
            update = lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Apply update
            param -= update

            # Apply decoupled weight decay
            if self.weight_decay > 0:
                param -= lr * self.weight_decay * param
```

### Learning Rate Schedule

From [src/optimizer.py](src/optimizer.py):

```python
def get_learning_rate_with_warmup(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step
        base_lr: Maximum/base learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (default 0)

    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay
```

### Complete Training Step

From [train_pretrain.py](train_pretrain.py):

```python
def train_step(
    model: GPTModel,
    optimizer: AdamW,
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> float:
    """Perform a single training step."""

    # 1. Forward pass
    logits = model.forward(inputs)

    # 2. Compute loss
    loss = cross_entropy_loss(logits, targets)

    # 3. Backward pass
    grad_logits = cross_entropy_loss_backward(logits, targets)
    gradients = model.backward(grad_logits)

    # 4. Clip gradients
    gradients = clip_gradient_norm(gradients, max_grad_norm)

    # 5. Update parameters
    optimizer.step(gradients, learning_rate=learning_rate)

    return loss
```

---

## Training Monitoring

### Key Metrics

| Metric          | Good Sign          | Bad Sign                 |
| --------------- | ------------------ | ------------------------ |
| Training Loss   | Decreasing         | Stuck or increasing      |
| Validation Loss | Decreasing         | Increasing (overfitting) |
| Gradient Norm   | Stable ~0.1-1.0    | Very large or NaN        |
| Learning Rate   | Following schedule | N/A                      |

### Loss Visualization

```
Loss
│
4 │*
3 │ *
2 │  **
1 │    ***
  │      ****
0 │          *******  ← converging
  └──────────────────────────► Epochs

Healthy training: exponential-like decay
```

---

## Try It Yourself

Run the pretraining script:

```bash
python train_pretrain.py
```

Example output:

```
GPT Language Model Pretraining
============================================================

Loading data...
Loaded 1,115,394 characters of text
Train: 1,003,854 chars, Val: 111,540 chars

Training tokenizer...
Vocabulary size: 2000

Creating model...
Model parameters: 1,044,736

Starting training...
------------------------------------------------------------
Epoch 1/3 | Step 50/485 | Loss: 7.2341 | LR: 1.50e-04
Epoch 1/3 | Step 100/485 | Loss: 5.8234 | LR: 3.00e-04
Epoch 1/3 | Step 150/485 | Loss: 4.9823 | LR: 2.95e-04
...
```

---

## References

1. **Adam Optimizer**: [Kingma, D.P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

2. **AdamW**: [Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

3. **Learning Rate Warmup**: [Goyal, P., et al. (2017). Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)

4. **Gradient Clipping**: [Pascanu, R., et al. (2012). On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)

5. **This Repository**: See [src/optimizer.py](src/optimizer.py) for `AdamW` and learning rate functions, [train_pretrain.py](train_pretrain.py) for complete training loop.

---

**Next Step**: Now that we can train our model, continue to [09 - TextGeneration.md](09%20-%20TextGeneration.md) to learn how to use it to generate text.
