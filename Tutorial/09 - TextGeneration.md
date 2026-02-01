# Text Generation: From Predictions to Natural Text

Text generation is where the trained model shows its capabilities. The model generates text one token at a time, using its predictions to extend a prompt into coherent continuation.

---

## Table of Contents

1. [Autoregressive Generation](#autoregressive-generation)
2. [Greedy Decoding](#greedy-decoding)
3. [Temperature Sampling](#temperature-sampling)
4. [Top-K Sampling](#top-k-sampling)
5. [Top-P (Nucleus) Sampling](#top-p-nucleus-sampling)
6. [Step-by-Step Example](#step-by-step-example)
7. [Sampling Strategies Compared](#sampling-strategies-compared)
8. [Code Implementation](#code-implementation)
9. [Generation Tips](#generation-tips)
10. [References](#references)

---

## Autoregressive Generation

GPT generates text **autoregressively**: each new token is predicted based on all previous tokens, then appended to the context.

### The Generation Loop

```
Prompt: "The cat"
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: Forward("The cat") → logits → sample → "sat"           │
│  Context: "The cat sat"                                          │
│                                                                   │
│  Step 2: Forward("The cat sat") → logits → sample → "on"        │
│  Context: "The cat sat on"                                       │
│                                                                   │
│  Step 3: Forward("The cat sat on") → logits → sample → "the"    │
│  Context: "The cat sat on the"                                   │
│                                                                   │
│  Step 4: Forward("The cat sat on the") → logits → sample → "mat"│
│  Context: "The cat sat on the mat"                               │
│                                                                   │
│  ... continue until max_tokens or EOS                            │
└──────────────────────────────────────────────────────────────────┘

Output: "The cat sat on the mat"
```

### Key Insight

The model outputs **probability distributions**, not single tokens. How we convert these distributions to tokens determines the output's character:

```
logits → softmax → probabilities → ??? → next token

The ??? is where sampling strategy matters!
```

---

## Greedy Decoding

**Greedy decoding** always selects the most probable token.

### Algorithm

```python
next_token = argmax(probabilities)
```

### Example

```
Probabilities: [0.05, 0.03, 0.62, 0.20, 0.10]
                                 ↑
Tokens:        ["a", "an", "the", "that", "this"]

Greedy choice: "the" (index 2, probability 0.62)
```

### Pros and Cons

| Pros                         | Cons                   |
| ---------------------------- | ---------------------- |
| Deterministic (reproducible) | Repetitive output      |
| Fast (no sampling)           | Boring, generic text   |
| Highest confidence path      | Can get stuck in loops |

### The Repetition Problem

Greedy decoding often produces repetitive text:

```
Prompt: "The quick brown"

Greedy output: "The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog..."
```

---

## Temperature Sampling

**Temperature** controls the randomness of sampling by scaling the logits before softmax.

### The Formula

$$P(token_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

Where:

- $z_i$ = logit for token $i$
- $T$ = temperature

### Temperature Effects

| Temperature | Effect                                | Use Case           |
| ----------- | ------------------------------------- | ------------------ |
| $T < 1$     | Sharper distribution (more confident) | Factual, focused   |
| $T = 1$     | Original distribution                 | Balanced           |
| $T > 1$     | Flatter distribution (more random)    | Creative, diverse  |
| $T → 0$     | Approaches greedy                     | Maximum focus      |
| $T → ∞$     | Uniform distribution                  | Maximum randomness |

### Visual Intuition

```
Original logits: [2.0, 1.5, 1.0, 0.5, 0.0]

T = 0.5 (cold):          T = 1.0 (normal):        T = 2.0 (hot):
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│█████████        │0.53  │████████         │0.39  │█████            │0.27
│████             │0.22  │█████            │0.26  │█████            │0.22
│██               │0.13  │████             │0.17  │████             │0.18
│█                │0.08  │██               │0.11  │███              │0.17
│                 │0.05  │█                │0.07  │███              │0.15
└─────────────────┘      └─────────────────┘      └─────────────────┘
Peak probability: 53%         39%                      27%
```

### Numeric Example

```python
# Original logits
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0])

# Temperature = 1.0 (unchanged)
probs_t1 = softmax(logits / 1.0)
# = [0.39, 0.26, 0.17, 0.11, 0.07]

# Temperature = 0.5 (sharper)
probs_t05 = softmax(logits / 0.5)
# = softmax([4.0, 3.0, 2.0, 1.0, 0.0])
# = [0.53, 0.22, 0.13, 0.08, 0.05]

# Temperature = 2.0 (flatter)
probs_t2 = softmax(logits / 2.0)
# = softmax([1.0, 0.75, 0.5, 0.25, 0.0])
# = [0.27, 0.22, 0.18, 0.17, 0.15]
```

---

## Top-K Sampling

**Top-K sampling** restricts sampling to only the $K$ most likely tokens.

### Algorithm

```python
# Sort tokens by probability
sorted_probs = argsort(probs)[::-1]  # Descending

# Keep only top K
top_k_probs = sorted_probs[:K]

# Zero out the rest
probs_filtered = zeros_like(probs)
probs_filtered[top_k_probs] = probs[top_k_probs]

# Re-normalize
probs_filtered = probs_filtered / sum(probs_filtered)

# Sample
next_token = sample(probs_filtered)
```

### Example (K=3)

```
Original probs:  [0.35, 0.25, 0.20, 0.12, 0.08]
                   ↑     ↑     ↑
Tokens:          [A,    B,    C,    D,    E]

Top-3 only:      [0.35, 0.25, 0.20, 0.00, 0.00]

Re-normalized:   [0.44, 0.31, 0.25, 0.00, 0.00]
                   ↑
Now sample from this
```

### Why Top-K?

Without top-K, even with temperature adjustment, there's a small chance of sampling very unlikely tokens:

```
Probabilities: [0.80, 0.10, 0.05, 0.04, 0.01]
                                        ↑
Could sample this garbage token!

With K=3: [0.84, 0.11, 0.05, 0.00, 0.00]
                              ↑
No chance of garbage!
```

### Trade-off

| K value | Result               |
| ------- | -------------------- |
| K=1     | Equivalent to greedy |
| K=5-10  | Good balance         |
| K=50+   | Almost no filtering  |

---

## Top-P (Nucleus) Sampling

**Top-P** (also called nucleus sampling) keeps the smallest set of tokens whose cumulative probability exceeds $P$.

### Algorithm

```python
# Sort by probability (descending)
sorted_probs = sort(probs)[::-1]

# Find cumulative sum
cumulative = cumsum(sorted_probs)

# Find cutoff (first position where cumsum > P)
cutoff = first_where(cumulative > P)

# Keep only tokens up to cutoff
nucleus = sorted_probs[:cutoff]

# Re-normalize and sample
```

### Example (P=0.9)

```
Original probs (sorted): [0.40, 0.30, 0.15, 0.10, 0.05]
Cumulative sum:          [0.40, 0.70, 0.85, 0.95, 1.00]
                                              ↑
                                     First > 0.9

Nucleus (top 4):         [0.40, 0.30, 0.15, 0.10]
Re-normalized:           [0.42, 0.32, 0.16, 0.10]
```

### Why Top-P Instead of Top-K?

Top-P adapts to the distribution shape:

```
Scenario 1: Confident prediction
Probs: [0.80, 0.10, 0.05, 0.03, 0.02]
P=0.9 keeps: [0.80, 0.10] (just 2 tokens!)

Scenario 2: Uncertain prediction
Probs: [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
P=0.9 keeps: [0.25, 0.20, 0.18, 0.15, 0.12] (5 tokens)
```

With fixed K, you'd either have too few options when confident, or too many when uncertain.

---

## Step-by-Step Example

Let's generate "sat on the mat" from prompt "The cat":

### Setup

```
Vocabulary (simplified): {"The": 0, "cat": 1, "sat": 2, "on": 3, "the": 4, "mat": 5, ...}
Temperature: 0.8
Top-K: 3
```

### Generation Step 1: Predict after "The cat"

```python
# Forward pass with input [0, 1] ("The cat")
logits = model.forward([[0, 1]])  # Shape: (1, 2, vocab_size)

# Get logits for last position (after "cat")
next_logits = logits[0, -1, :]  # Shape: (vocab_size,)
# = [1.2, 0.5, 3.1, 0.8, 0.3, -0.2, ...]

# Apply temperature (T=0.8)
scaled_logits = next_logits / 0.8
# = [1.5, 0.63, 3.88, 1.0, 0.38, -0.25, ...]

# Softmax to get probabilities
probs = softmax(scaled_logits)
# = [0.08, 0.03, 0.55, 0.05, 0.02, 0.01, ...]

# Top-K filtering (K=3)
# Top 3: "sat"(0.55), "The"(0.08), "on"(0.05)
probs_filtered = [0.08, 0.00, 0.55, 0.05, 0.00, 0.00, ...]
probs_renorm = [0.12, 0.00, 0.81, 0.07, 0.00, 0.00, ...]

# Sample
next_token = sample(probs_renorm)  # Likely "sat" (index 2)
```

### Generation Step 2: Predict after "The cat sat"

```python
# Forward pass with [0, 1, 2] ("The cat sat")
logits = model.forward([[0, 1, 2]])

# Get logits for position after "sat"
next_logits = logits[0, -1, :]
# = [0.2, 0.3, 0.1, 2.8, 0.5, -0.1, ...]

# Apply temperature, softmax, top-K...
# Result: "on" sampled (index 3)
```

### Continue...

```
Step 3: "The cat sat on" → predict "the" (4)
Step 4: "The cat sat on the" → predict "mat" (5)
```

### Final Output

```
Input:  "The cat"
Output: "The cat sat on the mat"
```

---

## Sampling Strategies Compared

### Side-by-Side

```
Prompt: "Once upon a time there was a"

Greedy (T=0):
"Once upon a time there was a king who lived in a castle.
The king was a good king who lived in a castle. The king..."
(Repetitive, gets stuck)

Temperature 0.5:
"Once upon a time there was a young princess who lived in
the kingdom of Everland. She loved to read books and play
with her cat Whiskers."
(Coherent but predictable)

Temperature 1.0 + Top-K=50:
"Once upon a time there was a peculiar baker named Mortimer
who could communicate with bread. Every morning, the loaves
would whisper secrets of the kingdom."
(Creative, diverse)

Temperature 2.0:
"Once upon a time there was a seventh quantum jellybeans
harmonize the cathedral's purple manifesto concerning
vegetables that telegraph..."
(Too random, loses coherence)
```

### Strategy Selection Guide

| Goal               | Recommended Settings      |
| ------------------ | ------------------------- |
| Code generation    | T=0.2, greedy or top-K=10 |
| Factual writing    | T=0.5-0.7, top-K=50       |
| Creative writing   | T=0.8-1.0, top-P=0.95     |
| Brainstorming      | T=1.0-1.2, top-P=0.9      |
| Maximum creativity | T=1.5, top-P=0.95         |

---

## Code Implementation

From [src/model.py](src/model.py):

```python
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
```

### Helper Function: Top-P Filtering

```python
def top_p_filtering(logits: np.ndarray, top_p: float = 0.9) -> np.ndarray:
    """
    Filter logits using nucleus (top-p) sampling.

    Args:
        logits: Raw logits, shape (vocab_size,)
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    # Sort indices by logit value (descending)
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]

    # Convert to probabilities
    probs = softmax(sorted_logits)

    # Cumulative probabilities
    cumulative_probs = np.cumsum(probs)

    # Find where cumulative probability exceeds top_p
    cutoff_index = np.searchsorted(cumulative_probs, top_p) + 1

    # Create mask for tokens to keep
    indices_to_remove = sorted_indices[cutoff_index:]

    # Set removed indices to -inf
    filtered_logits = logits.copy()
    filtered_logits[indices_to_remove] = float('-inf')

    return filtered_logits
```

---

## Generation Tips

### 1. Start with a Good Prompt

```
Bad prompt:  "Write"
Good prompt: "Write a poem about autumn leaves falling:"

Bad prompt:  "The"
Good prompt: "The old wizard carefully opened the ancient tome and read:"
```

### 2. Match Temperature to Task

```python
# Factual/Code: Low temperature
output = model.generate(prompt, temperature=0.3)

# Creative writing: Higher temperature
output = model.generate(prompt, temperature=0.9, top_k=50)
```

### 3. Use Repetition Penalties (Advanced)

To reduce repetition, penalize tokens that already appeared:

```python
def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    for token in set(generated_tokens):
        logits[token] /= penalty
    return logits
```

### 4. Set Appropriate Max Length

```python
# Too short: May cut off mid-sentence
model.generate(prompt, max_new_tokens=10)  # "The cat sat on the"

# Too long: May degrade quality
model.generate(prompt, max_new_tokens=1000)  # Quality drops after a while
```

---

## Try It Yourself

Run the demo script:

```bash
python run_demo.py
```

Or experiment directly:

```python
from src.model import GPTModel, GPTConfig
from src.tokenizer import BPETokenizer

# Load trained model and tokenizer
model = GPTModel.load("checkpoints/model_best.npz")
tokenizer = BPETokenizer.load("checkpoints/tokenizer.json")

# Encode prompt
prompt = "ROMEO:"
prompt_tokens = np.array([tokenizer.encode(prompt)])

# Generate with different settings
print("=== Greedy ===")
output = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.001)
print(tokenizer.decode(output[0].tolist()))

print("\n=== Temperature 0.7 ===")
output = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(output[0].tolist()))

print("\n=== Temperature 1.0, Top-K 50 ===")
output = model.generate(prompt_tokens, max_new_tokens=50, temperature=1.0, top_k=50)
print(tokenizer.decode(output[0].tolist()))
```

---

## References

1. **Top-K Sampling**: [Fan, A., et al. (2018). Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)

2. **Nucleus Sampling**: [Holtzman, A., et al. (2019). The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

3. **Temperature Scaling**: [Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

4. **Sampling Strategies Survey**: [Ippolito, D., et al. (2019). Comparison of Diverse Decoding Methods](https://arxiv.org/abs/1909.00459)

5. **This Repository**: See [src/model.py](src/model.py) for `GPTModel.generate` method, [run_demo.py](run_demo.py) for usage examples.

---

**Next Step**: Now you can generate text! Continue to [10 - FineTuning.md](10%20-%20FineTuning.md) to learn how to adapt the model to specific tasks with minimal training.
