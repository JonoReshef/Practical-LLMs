# How to Learn with This Repository

Welcome! This repository contains a complete language model built from scratch using only NumPy. If you've ever wondered "how does ChatGPT actually work under the hood?", you're in the right place.

This guide will walk you through the codebase in a logical learning order, explaining what each component does and why it matters.

---

## What You'll Learn

By studying this codebase, you'll understand:

1. How text gets converted into numbers (tokenization)
2. How neural networks process sequences (attention mechanism)
3. How transformers stack these components together
4. How models learn from data (backpropagation and optimization)
5. How to generate text one token at a time
6. How to efficiently adapt models to new tasks (LoRA)

---

## Quick Start: Run the Demos

Every module includes an educational demo you can run directly. This is the fastest way to see each component in action:

```bash
# Activate the environment
source .venv/bin/activate

# Run demos in learning order:
python -m src.activations    # 1. Activation functions (softmax, GELU)
python -m src.layers         # 2. Core layers (Linear, LayerNorm, Embedding)
python -m src.tokenizer      # 3. Text to numbers (BPE tokenization)
python -m src.attention      # 4. The heart of transformers
python -m src.transformer    # 5. Transformer blocks and stacks
python -m src.optimizer      # 6. How models learn (AdamW)
python -m src.model          # 7. Complete GPT model
python -m src.lora           # 8. Efficient fine-tuning
python -m src.utils          # 9. Data loading and checkpointing
```

Each demo is self-contained and prints educational explanations alongside the output. Run them, read the output, then dive into the source code!

---

## Prerequisites

You should be comfortable with:

- Python basics (classes, functions, loops)
- NumPy array operations
- Basic linear algebra (matrix multiplication, vectors)
- Calculus concepts (derivatives, chain rule) - helpful but not required

---

## The Big Picture: What is a Language Model?

A language model predicts the next word (or token) given the previous words. That's it.

```
Input:  "The cat sat on the"
Output: "mat" (with some probability)
```

When you chat with ChatGPT, it's doing this prediction millions of times, one token at a time, to generate a response.

The magic is in _how_ we teach a neural network to make these predictions, and _how_ we structure the network to understand context and meaning.

---

## Learning Path

Each step in this learning path has a detailed accompanying document with numeric examples, visualizations, and deep explanations. These are linked at the beginning of each section.

| Step | Topic | Code | Deep Dive |
|------|-------|------|-----------|
| 1 | Tokenization | `src/tokenizer.py` | [Tokenization.md](Tokenization.md) |
| 2 | Embeddings | `src/layers.py` | [Embeddings.md](Embeddings.md) |
| 3 | Positional Encoding | `src/layers.py` | [PositionalEncoding.md](PositionalEncoding.md) |
| 4 | Attention | `src/attention.py` | [Attention.md](Attention.md) |
| 5 | Feed-Forward Network | `src/transformer.py` | [FeedForwardNetwork.md](FeedForwardNetwork.md) |
| 6 | Transformer Block | `src/transformer.py` | [TransformerBlock.md](TransformerBlock.md) |
| 7 | Full GPT Model | `src/model.py` | [GPTModel.md](GPTModel.md) |
| 8 | Training | `src/optimizer.py` | [Training.md](Training.md) |
| 9 | Text Generation | `src/model.py` | [TextGeneration.md](TextGeneration.md) |
| 10 | Fine-Tuning | `src/lora.py` | [FineTuning.md](FineTuning.md) |

---

### Step 1: Tokenization (`src/tokenizer.py`)

**Deep Dive:** [Tokenization.md](Tokenization.md) - Complete BPE walkthrough with step-by-step numeric examples

**The Question:** How do we convert text into numbers that a neural network can process?

**The Answer:** We break text into smaller pieces called "tokens" and assign each a unique ID.

**Run the Demo:** `python -m src.tokenizer` - See BPE training, encoding, and decoding in action.

**What to Study:**

- Open `src/tokenizer.py` and read the module docstring
- Look at the `BPETokenizer` class

**Key Concepts:**

1. **Why not just use characters?**
   - "hello" = 5 characters, but the model would need to learn that these 5 characters together mean something
   - Better to have "hello" as one token, or split it intelligently: "hel" + "lo"

2. **Byte Pair Encoding (BPE):**
   - Start with individual characters
   - Find the most common pair of adjacent tokens
   - Merge that pair into a new token
   - Repeat until you reach your vocabulary size

   Example progression:

   ```
   "low lower lowest"
   → ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
   → ['lo','w',' ','lo','w','e','r',' ','lo','w','e','s','t']  # merged 'l'+'o'
   → ['low',' ','low','e','r',' ','low','e','s','t']           # merged 'lo'+'w'
   → ['low',' ','low','er',' ','low','est']                    # and so on...
   ```

3. **Special Tokens:**
   - `<PAD>` - Fills empty space when batching sequences of different lengths
   - `<UNK>` - Represents unknown/rare tokens
   - `<BOS>` - "Beginning of sequence" marker
   - `<EOS>` - "End of sequence" marker

**Try It:**

```python
from src.tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train("To be or not to be, that is the question.", vocabulary_size=50)

tokens = tokenizer.encode("To be or not")
print(tokens)  # Something like [5, 12, 7, 23, ...]

text = tokenizer.decode(tokens)
print(text)  # "To be or not"
```

---

### Step 2: Embeddings (`src/layers.py` - Embedding class)

**Deep Dive:** [Embeddings.md](Embeddings.md) - Lookup tables, semantic similarity, and vector arithmetic examples

**The Question:** We have token IDs (integers). How do we give these meaning?

**The Answer:** We learn a lookup table that maps each token ID to a vector of numbers.

**Run the Demo:** `python -m src.layers` - See embeddings, layer norm, and linear layers explained.

**What to Study:**

- Open `src/layers.py` and find the `Embedding` class
- Read the docstring and look at the `forward` method

**Key Concepts:**

1. **What is an embedding?**
   - Each token gets a vector (list of numbers) that represents its "meaning"
   - Token 42 might map to [0.2, -0.5, 0.8, 0.1, ...]
   - These vectors are learned during training

2. **Why vectors?**
   - Similar words end up with similar vectors
   - "king" - "man" + "woman" ≈ "queen" (the famous word2vec example)
   - The model can generalize: if it learns something about "cat", it helps with "kitten"

3. **Embedding dimension:**
   - Our model uses 128-dimensional embeddings
   - GPT-3 uses 12,288 dimensions
   - More dimensions = more capacity to represent nuance

**The Math:**

```
If vocabulary_size = 2000 and embedding_dim = 128:
    embedding_table has shape (2000, 128)

For token ID 42:
    embedding = embedding_table[42]  # Shape: (128,)
```

---

### Step 3: Positional Encoding (`src/layers.py` - PositionalEncoding class)

**Deep Dive:** [PositionalEncoding.md](PositionalEncoding.md) - Sinusoidal formulas with numeric example matrix

**The Question:** Token embeddings don't know their position. How does the model know word order?

**The Answer:** We add a position-dependent signal to each embedding.

**What to Study:**

- Find `PositionalEncoding` in `src/layers.py`
- Look at the sinusoidal pattern formula

**Key Concepts:**

1. **The Problem:**
   - "Dog bites man" and "Man bites dog" have the same tokens
   - Without position info, the model can't tell them apart

2. **The Solution (Sinusoidal Encoding):**
   - Each position gets a unique pattern of sine and cosine waves
   - Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
   - Position 1: [sin(1/10000^0), cos(1/10000^0), sin(1/10000^(2/d)), ...]
3. **Why sines and cosines?**
   - They create unique patterns for each position
   - The model can learn to compute relative positions (position 5 vs position 3)
   - They work for sequences longer than seen during training

**The Math:**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

where:
- pos = position in sequence (0, 1, 2, ...)
- i = dimension index
- d = embedding dimension
```

---

### Step 4: Attention (`src/attention.py`)

**Deep Dive:** [Attention.md](Attention.md) - Q/K/V intuition, scaled dot-product, causal masking, multi-head attention

**This is the heart of transformers.** Take your time here.

**The Question:** How does the model decide which other tokens to "pay attention to" when processing each token?

**The Answer:** Each token creates a Query ("what am I looking for?"), and all tokens create Keys ("what do I contain?") and Values ("what information do I have?"). We match Queries to Keys to decide how much to weight each Value.

**Run the Demo:** `python -m src.attention` - See Q/K/V intuition, attention weights, and causal masking.

**What to Study:**

- Start with `scaled_dot_product_attention` function
- Then study `MultiHeadAttention` class

**Key Concepts:**

1. **Query, Key, Value (Q, K, V):**
   - Think of it like a search engine:
     - Query: Your search terms
     - Key: The title/tags of each document
     - Value: The actual content of each document
   - High Query-Key match = that Value is important

2. **The Attention Formula:**

   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   ```

   Step by step:
   - `QK^T`: How well does each query match each key? (similarity scores)
   - `/ √d_k`: Scale down to prevent extreme values
   - `softmax`: Convert to probabilities (sum to 1)
   - `× V`: Weighted sum of values

3. **Causal (Autoregressive) Masking:**
   - When predicting token 5, we can only look at tokens 0-4
   - We "mask out" future positions by setting their attention scores to -infinity
   - After softmax, -infinity becomes 0 probability

4. **Multi-Head Attention:**
   - Instead of one attention, we do it 4 times in parallel (4 "heads")
   - Each head can learn different patterns:
     - Head 1 might focus on grammar
     - Head 2 might focus on recent context
     - Head 3 might focus on subject-verb relationships
   - We concatenate all heads and project back

**Visual Example:**

```
Sentence: "The cat sat on the mat"

When processing "sat":
  Query for "sat" asks: "Who did the action? What was the context?"

  Attention weights might be:
    "The" : 0.05
    "cat" : 0.60  ← High! "cat" is the subject
    "sat" : 0.20
    "on"  : 0.10
    "the" : 0.03
    "mat" : 0.02  ← Can't see this (masked, it's in the future)
```

---

### Step 5: Feed-Forward Network (`src/transformer.py` - FeedForwardNetwork)

**Deep Dive:** [FeedForwardNetwork.md](FeedForwardNetwork.md) - Expand-GELU-contract pattern with numeric walkthrough

**The Question:** Attention lets tokens talk to each other. What processes each token individually?

**The Answer:** A simple two-layer neural network applied to each position.

**Run the Demo:** `python -m src.transformer` - See FFN, transformer blocks, and how they stack.

**What to Study:**

- Find `FeedForwardNetwork` in `src/transformer.py`

**Key Concepts:**

1. **Structure:**

   ```
   Input (128 dim) → Linear (512 dim) → GELU → Linear (128 dim) → Output
   ```

2. **Why expand then contract?**
   - The middle layer (512 dim) is larger, giving more capacity
   - Think of it as: expand to process, then compress back
   - This is where a lot of "knowledge" is stored

3. **GELU Activation:**
   - Like ReLU but smoother
   - `GELU(x) ≈ x × Φ(x)` where Φ is the normal distribution CDF
   - Allows small negative values through (unlike ReLU which is strictly 0)
   - **Demo:** `python -m src.activations` to see softmax, GELU, and ReLU compared

---

### Step 6: Transformer Block (`src/transformer.py` - TransformerBlock)

**Deep Dive:** [TransformerBlock.md](TransformerBlock.md) - Pre-LN architecture, residual connections, complete block flow

**The Question:** How do we combine attention and feed-forward into one unit?

**The Answer:** Attention → Add & Norm → FFN → Add & Norm

**What to Study:**

- Find `TransformerBlock` in `src/transformer.py`
- Trace through the `forward` method

**Key Concepts:**

1. **Residual Connections ("Add"):**

   ```python
   output = layer(x) + x  # Add the input back!
   ```

   - Helps gradients flow during training
   - Allows the model to "skip" layers if they're not helpful
   - Makes training much more stable

2. **Layer Normalization ("Norm"):**
   - Normalizes values to have mean=0, std=1
   - Prevents values from exploding or vanishing
   - Applied before each sub-layer (this is "Pre-LN" architecture)

3. **Pre-LN vs Post-LN:**
   - Original transformer: LayerNorm after attention/FFN (Post-LN)
   - Modern practice: LayerNorm before attention/FFN (Pre-LN)
   - Pre-LN is more stable for training

**The Flow:**

```
x
│
├──────────────────┐
│                  │
▼                  │
LayerNorm          │
│                  │
▼                  │
Multi-Head Attn    │
│                  │
▼                  │
+◄─────────────────┘  (Residual connection)
│
├──────────────────┐
│                  │
▼                  │
LayerNorm          │
│                  │
▼                  │
Feed-Forward       │
│                  │
▼                  │
+◄─────────────────┘  (Residual connection)
│
▼
output
```

---

### Step 7: The Full Model (`src/model.py`)

**Deep Dive:** [GPTModel.md](GPTModel.md) - Complete architecture diagram, weight tying, parameter count breakdown

**The Question:** How does it all fit together?

**The Answer:** Stack multiple transformer blocks and add input/output processing.

**Run the Demo:** `python -m src.model` - See the complete GPT model, forward pass, and text generation.

**What to Study:**

- Read `GPTConfig` to see the hyperparameters
- Trace through `GPTModel.forward`

**Key Concepts:**

1. **The Full Pipeline:**

   ```
   Token IDs
       ↓
   Token Embedding (lookup table)
       ↓
   + Positional Encoding
       ↓
   Transformer Block 1
       ↓
   Transformer Block 2
       ↓
   Transformer Block 3
       ↓
   Transformer Block 4
       ↓
   Final LayerNorm
       ↓
   Output Projection (Linear to vocab size)
       ↓
   Logits (scores for each possible next token)
   ```

2. **Output Projection:**
   - Converts hidden states (128 dim) to vocabulary scores (2000 dim)
   - Each score represents "how likely is this token to be next?"

3. **Logits vs Probabilities:**
   - Logits: Raw scores (can be any number)
   - Apply softmax to get probabilities (0-1, sum to 1)

---

### Step 8: Training (`src/optimizer.py`, `train_pretrain.py`)

**Deep Dive:** [Training.md](Training.md) - Cross-entropy loss, AdamW optimizer, learning rate scheduling

**The Question:** How does the model learn?

**The Answer:** Show it text, have it predict the next token, tell it how wrong it was, adjust weights to be less wrong.

**Run the Demo:** `python -m src.optimizer` - See AdamW, gradient clipping, and learning rate schedules.

**What to Study:**

- `cross_entropy_loss` in `src/model.py`
- `AdamW` optimizer in `src/optimizer.py`
- The training loop in `train_pretrain.py`

**Key Concepts:**

1. **Cross-Entropy Loss:**
   - Measures how "surprised" the model is by the correct answer
   - If model says "mat" has 90% probability and it was correct: low loss
   - If model says "mat" has 1% probability and it was correct: high loss
   - `Loss = -log(probability of correct token)`

2. **Backpropagation:**
   - Compute loss
   - Compute gradient of loss with respect to every weight
   - Gradient tells us: "if I increase this weight, how does the loss change?"
   - Update weights in the direction that decreases loss

3. **AdamW Optimizer:**
   - Smarter than basic gradient descent
   - Keeps track of momentum (recent gradient direction)
   - Adapts learning rate per-parameter
   - Includes weight decay (regularization)

4. **Learning Rate Schedule:**
   - Start small (warmup) - model is random, don't make big changes
   - Increase to peak learning rate
   - Gradually decrease (cosine decay) - fine-tune carefully

5. **Data Utilities:**
   - `TextDataset` and `DataLoader` in `src/utils.py` handle batching
   - **Demo:** `python -m src.utils` to see data loading and checkpointing

**The Training Loop:**

```python
for batch in data:
    # Forward pass
    logits = model.forward(input_tokens)
    loss = cross_entropy_loss(logits, target_tokens)

    # Backward pass
    gradients = model.backward(loss_gradient)

    # Update weights
    optimizer.step(gradients)
```

---

### Step 9: Text Generation (`GPTModel.generate`)

**Deep Dive:** [TextGeneration.md](TextGeneration.md) - Autoregressive generation, temperature, top-k, top-p sampling

**The Question:** How do we go from a trained model to actual text output?

**The Answer:** Predict one token, add it to the input, repeat.

**What to Study:**

- `generate` method in `src/model.py`

**Key Concepts:**

1. **Autoregressive Generation:**

   ```
   Input: "The"
   Model predicts: "cat" → Input becomes "The cat"
   Model predicts: "sat" → Input becomes "The cat sat"
   Model predicts: "on"  → Input becomes "The cat sat on"
   ...and so on
   ```

2. **Temperature:**
   - Controls randomness
   - `temperature = 0`: Always pick the highest probability token (deterministic)
   - `temperature = 1`: Sample according to probabilities
   - `temperature > 1`: More random (flatter distribution)
   - `temperature < 1`: More focused (sharper distribution)

3. **Top-k Sampling:**
   - Only consider the k most likely tokens
   - Prevents the model from picking very unlikely tokens
   - `top_k = 50`: Only sample from top 50 candidates

4. **Why sampling matters:**
   - Greedy (always pick best) leads to repetitive, boring text
   - Too random leads to nonsense
   - Temperature + top-k finds the sweet spot

---

### Step 10: Fine-Tuning (`train_finetune_full.py`, `train_finetune_lora.py`)

**Deep Dive:** [FineTuning.md](FineTuning.md) - Full fine-tuning vs LoRA, low-rank matrix decomposition

**The Question:** We have a model trained on general text. How do we specialize it?

**The Answer:** Continue training on task-specific data.

**Run the Demo:** `python -m src.lora` - See how LoRA reduces trainable parameters by 100x.

**What to Study:**

- Compare `train_finetune_full.py` and `train_finetune_lora.py`
- Study `LoRALinear` in `src/lora.py`

**Key Concepts:**

1. **Full Fine-Tuning:**
   - Update ALL model parameters
   - Requires storing full optimizer state for all weights
   - Risk of "catastrophic forgetting" (losing general knowledge)

2. **LoRA (Low-Rank Adaptation):**
   - Freeze the original weights
   - Add small "adapter" matrices that get trained
   - Much fewer parameters: `W' = W + BA` where B and A are small

   ```
   Original: W has shape (128, 128) = 16,384 parameters
   LoRA: B has shape (128, 4), A has shape (4, 128) = 1,024 parameters
   That's 16x fewer parameters!
   ```

3. **Why LoRA works:**
   - The "delta" (change) to weights often lies in a low-rank subspace
   - We don't need to modify every weight, just the important directions
   - Multiple LoRA adapters can be swapped without reloading the base model

---

## Exercises

### Exercise 1: Tokenizer Exploration

```python
from src.tokenizer import BPETokenizer

# Train on a small corpus
tokenizer = BPETokenizer()
text = "to be or not to be that is the question"
tokenizer.train(text, vocabulary_size=30)

# Try encoding various texts
print(tokenizer.encode("to be"))
print(tokenizer.encode("question"))
print(tokenizer.encode("banana"))  # Not in training data!
```

Questions:

- What happens with out-of-vocabulary words?
- How does vocabulary size affect tokenization?

### Exercise 2: Attention Visualization

```python
from src.attention import scaled_dot_product_attention, create_causal_mask
import numpy as np

# Create simple Q, K, V
seq_len, d_k = 4, 8
Q = np.random.randn(1, seq_len, d_k)
K = np.random.randn(1, seq_len, d_k)
V = np.random.randn(1, seq_len, d_k)

# Without mask
output1, weights1 = scaled_dot_product_attention(Q, K, V)

# With causal mask
mask = create_causal_mask(seq_len)
output2, weights2 = scaled_dot_product_attention(Q, K, V, mask)

print("Without mask - attention weights:")
print(weights1[0])
print("\nWith causal mask - attention weights:")
print(weights2[0])
```

Questions:

- What pattern do you see in the masked attention weights?
- Why are some weights exactly 0?

### Exercise 3: Temperature Effects

```python
from src.model import GPTModel, GPTConfig
from src.tokenizer import BPETokenizer
import numpy as np

# Load or create a trained model
# ... (see run_demo.py for setup)

prompt = tokenizer.encode("The king")
prompt = np.array([prompt])

# Try different temperatures
for temp in [0.1, 0.5, 1.0, 2.0]:
    output = model.generate(prompt, max_new_tokens=20, temperature=temp)
    print(f"Temperature {temp}: {tokenizer.decode(output[0])}")
```

Questions:

- How does the output change with temperature?
- What temperature gives the most coherent output?

---

## Common Questions

**Q: Why NumPy instead of PyTorch/TensorFlow?**

A: Educational clarity. NumPy forces us to implement every operation explicitly. There's no magic - you see exactly what matrix multiplications happen, what shapes tensors have, and how gradients flow. Once you understand this, PyTorch/TF become much clearer.

**Q: Why is this model so small?**

A: So you can train it on a laptop! GPT-3 has 175 billion parameters. Our model has ~600,000. The principles are identical, just scaled down.

**Q: Why does generation sometimes produce nonsense?**

A: Several reasons:

- Small model with limited capacity
- Small training dataset (just Shakespeare)
- Limited training time
- Language models are probabilistic - they can always generate unlikely sequences

**Q: How is this different from ChatGPT?**

A: ChatGPT adds:

- Much larger scale (billions of parameters)
- Instruction fine-tuning (RLHF - Reinforcement Learning from Human Feedback)
- Dialogue formatting
- Safety filters
- But the core transformer architecture is the same!

---

## Next Steps

After understanding this codebase:

1. **Scale Up:** Try implementing in PyTorch and training a larger model
2. **Read Papers:** Now you can understand the original "Attention Is All You Need" paper
3. **Explore Variants:** BERT (bidirectional), T5 (encoder-decoder), LLaMA, etc.
4. **Fine-tuning Practice:** Try LoRA on a pre-trained model like LLaMA
5. **Dive Deeper:** Study Flash Attention, KV caching, quantization

---

## Glossary

| Term                 | Definition                                             |
| -------------------- | ------------------------------------------------------ |
| **Token**            | A unit of text (word, subword, or character)           |
| **Embedding**        | A learned vector representation of a token             |
| **Attention**        | Mechanism for tokens to "look at" other tokens         |
| **Query/Key/Value**  | The three vectors used in attention computation        |
| **Transformer**      | Architecture using attention + feed-forward layers     |
| **Causal Mask**      | Prevents attending to future positions                 |
| **Logits**           | Raw output scores before softmax                       |
| **Cross-Entropy**    | Loss function for classification/next-token prediction |
| **Backpropagation**  | Algorithm for computing gradients                      |
| **Gradient Descent** | Optimization by moving opposite to gradient            |
| **Learning Rate**    | Step size for weight updates                           |
| **Epoch**            | One pass through the entire training dataset           |
| **Batch**            | Subset of data processed together                      |
| **Fine-tuning**      | Adapting a pre-trained model to a specific task        |
| **LoRA**             | Low-Rank Adaptation - parameter-efficient fine-tuning  |

---

Happy learning! The best way to understand this code is to run it, modify it, break it, and fix it. Every error message is a learning opportunity.
