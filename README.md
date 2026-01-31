# LLM Explainer

An educational implementation of a GPT-style language model from scratch using pure NumPy. This project demonstrates how Large Language Models (LLMs) work at every level, from tokenization to training to generation.

## Purpose

This codebase is designed to teach the internals of transformer-based language models through human-readable code with extensive documentation. Every component includes:

- Detailed docstrings explaining the mathematical operations
- References to original research papers
- Clear variable naming that maps to paper notation

## Project Structure

```
LLM-explainer/
├── src/                      # Core implementation modules
│   ├── activations.py        # Activation functions (softmax, GELU, ReLU)
│   ├── layers.py             # Neural network layers (Linear, LayerNorm, Embedding)
│   ├── tokenizer.py          # BPE tokenizer implementation
│   ├── attention.py          # Attention mechanisms (scaled dot-product, multi-head)
│   ├── transformer.py        # Transformer blocks and stacks
│   ├── optimizer.py          # AdamW optimizer with learning rate scheduling
│   ├── model.py              # Complete GPT model assembly (NumPy)
│   ├── model_mlx.py          # GPU-accelerated model (Apple MLX)
│   ├── lora.py               # LoRA parameter-efficient fine-tuning
│   └── utils.py              # Dataset, DataLoader, checkpointing utilities
├── tests/                    # Unit tests for all modules
│   ├── test_activations.py
│   ├── test_layers.py
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   ├── test_transformer.py
│   ├── test_optimizer.py
│   ├── test_model.py
│   ├── test_model_mlx.py     # MLX tests (skip on non-Apple systems)
│   ├── test_lora.py
│   └── test_utils.py
├── train_pretrain.py         # Pre-training script
├── train_finetune_full.py    # Full fine-tuning script
├── train_finetune_lora.py    # LoRA fine-tuning script
├── run_demo.py               # End-to-end demonstration (all modes)
├── data/                     # Training data (Shakespeare text)
├── checkpoints/              # NumPy model weights and tokenizer
└── checkpoints_mlx/          # MLX model checkpoints (when using mac-accel)
```

## Architecture Overview

This implementation follows the GPT (decoder-only transformer) architecture:

```
Input Token IDs
       │
       ▼
┌─────────────────────────────────────┐
│  Token Embedding + Positional Encoding  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│      Transformer Block × N          │
│  ┌───────────────────────────────┐  │
│  │ LayerNorm                     │  │
│  │ Multi-Head Self-Attention     │  │
│  │ (causal mask)                 │  │
│  │ + Residual Connection         │  │
│  ├───────────────────────────────┤  │
│  │ LayerNorm                     │  │
│  │ Feed-Forward Network          │  │
│  │ + Residual Connection         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Final LayerNorm                    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Output Projection → Logits         │
└─────────────────────────────────────┘
       │
       ▼
   Token Probabilities (via softmax)
```

## Detailed Documentation

For a deep understanding of each component, we provide comprehensive documentation files with:
- Mathematical formulas and derivations
- Step-by-step numeric examples
- Visualizations and diagrams
- References to research papers

| Topic | Documentation | Code |
|-------|---------------|------|
| Tokenization (BPE) | [Tokenization.md](Tokenization.md) | `src/tokenizer.py` |
| Word Embeddings | [Embeddings.md](Embeddings.md) | `src/layers.py` |
| Positional Encoding | [PositionalEncoding.md](PositionalEncoding.md) | `src/layers.py` |
| Attention Mechanism | [Attention.md](Attention.md) | `src/attention.py` |
| Feed-Forward Network | [FeedForwardNetwork.md](FeedForwardNetwork.md) | `src/transformer.py` |
| Transformer Block | [TransformerBlock.md](TransformerBlock.md) | `src/transformer.py` |
| Full GPT Model | [GPTModel.md](GPTModel.md) | `src/model.py` |
| Training & Optimization | [Training.md](Training.md) | `src/optimizer.py` |
| Text Generation | [TextGeneration.md](TextGeneration.md) | `src/model.py` |
| Fine-Tuning & LoRA | [FineTuning.md](FineTuning.md) | `src/lora.py` |

Start with [How to learn with this.md](How%20to%20learn%20with%20this.md) for a guided learning path.

## Module Reference

### `src/activations.py`

Activation functions with forward and backward passes.

**Functions:**

- `softmax(x, axis=-1)` - Numerically stable softmax
- `softmax_backward(softmax_output, upstream_gradient)` - Jacobian-vector product
- `gelu(x)` - Gaussian Error Linear Unit (used in GPT-2/3)
- `gelu_backward(x, upstream_gradient)` - GELU gradient
- `relu(x)` - Rectified Linear Unit
- `relu_backward(x, upstream_gradient)` - ReLU gradient

**Paper References:**

- GELU: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)

---

### `src/layers.py`

Core neural network building blocks.

**Classes:**

#### `Linear`

Fully connected layer: `y = xW^T + b`

- `forward(x)` - Compute output
- `backward(upstream_gradient)` - Compute gradients for weights, bias, and input
- `get_parameters()` / `get_gradients()` - Access weights and gradients

#### `LayerNorm`

Layer normalization for transformer stability.

- Normalizes across feature dimension: `y = γ * (x - μ) / σ + β`
- `forward(x)` / `backward(upstream_gradient)`

#### `Embedding`

Token embedding lookup table.

- `forward(token_ids)` - Look up embeddings for token IDs
- `backward(upstream_gradient)` - Accumulate gradients for each token

#### `PositionalEncoding`

Sinusoidal position encoding (from "Attention Is All You Need").

- `get_encoding(sequence_length)` - Get position vectors
- Encodes position using sin/cos at different frequencies

---

### `src/tokenizer.py`

Byte Pair Encoding (BPE) tokenizer.

**Class: `BPETokenizer`**

**Methods:**

- `train(text, vocabulary_size)` - Learn BPE merges from corpus
- `encode(text)` - Convert text to token IDs
- `decode(token_ids)` - Convert token IDs back to text
- `batch_encode(texts, padding, max_length)` - Batch tokenization
- `save(filepath)` / `load(filepath)` - Persist tokenizer

**Special Tokens:**

- `PAD` (0) - Padding token
- `UNK` (1) - Unknown token
- `BOS` (2) - Beginning of sequence
- `EOS` (3) - End of sequence

**Paper References:**

- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)

---

### `src/attention.py`

Attention mechanism implementations.

**Functions:**

- `scaled_dot_product_attention(Q, K, V, mask)` - Core attention operation
  - `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
- `create_causal_mask(sequence_length)` - Prevent attending to future tokens

**Classes:**

#### `MultiHeadAttention`

Multi-head self-attention with learned projections.

- Splits input into multiple heads for parallel attention
- Projects Q, K, V through learned linear layers
- `forward(x, mask)` / `backward(upstream_gradient)`

**Attributes:**

- `query_projection`, `key_projection`, `value_projection` - Input projections
- `output_projection` - Final projection after concatenating heads

**Paper References:**

- "Attention Is All You Need" (Vaswani et al., 2017)

---

### `src/transformer.py`

Transformer building blocks.

**Classes:**

#### `FeedForwardNetwork`

Two-layer MLP with GELU activation.

- `FFN(x) = GELU(xW_1 + b_1)W_2 + b_2`
- Hidden dimension typically 4x embedding dimension

#### `TransformerBlock`

Single transformer layer with Pre-LN architecture.

- Pre-LN: LayerNorm before attention/FFN (more stable training)
- Contains: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

#### `TransformerStack`

Stack of N transformer blocks.

- `forward(x, mask)` - Process through all layers
- `backward(upstream_gradient)` - Backpropagate through all layers
- `get_parameters()` / `set_parameters()` - Access all block parameters

---

### `src/optimizer.py`

Optimization algorithms.

**Classes:**

#### `AdamW`

Adam optimizer with decoupled weight decay.

- Maintains first moment (m) and second moment (v) estimates
- `initialize(parameters)` - Set up optimizer state
- `step(gradients, learning_rate)` - Update parameters

**Functions:**

- `clip_gradient_norm(gradients, max_norm)` - Gradient clipping for stability
- `get_learning_rate_with_warmup(step, base_lr, warmup_steps, total_steps)` - Linear warmup + cosine decay schedule

**Paper References:**

- "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)

---

### `src/model.py`

Complete GPT model.

**Classes:**

#### `GPTConfig`

Dataclass holding model hyperparameters:

- `vocab_size` - Vocabulary size
- `embedding_dim` - Token embedding dimension
- `num_heads` - Number of attention heads
- `num_layers` - Number of transformer blocks
- `ffn_hidden_dim` - Feed-forward hidden dimension
- `max_sequence_length` - Maximum context length

#### `GPTModel`

Full GPT language model.

**Methods:**

- `forward(input_tokens)` - Compute logits for next token prediction
- `backward(grad_logits)` - Compute all parameter gradients
- `generate(input_tokens, max_new_tokens, temperature, top_k)` - Autoregressive generation
- `get_parameters()` / `set_parameters(params)` - Model state management

**Functions:**

- `cross_entropy_loss(logits, targets, ignore_index)` - Language modeling loss
- `cross_entropy_loss_backward(logits, targets, ignore_index)` - Loss gradient

---

### `src/lora.py`

Low-Rank Adaptation for parameter-efficient fine-tuning.

**Classes:**

#### `LoRALinear`

Linear layer with frozen base weights + trainable low-rank adapters.

- `W' = W + BA` where B∈R^(d×r), A∈R^(r×d), r << d
- `forward(x)` - Compute with LoRA contribution
- `backward(upstream_gradient)` - Only compute gradients for A and B

**Functions:**

- `apply_lora_to_model(model, rank, alpha, target_modules)` - Add LoRA to attention layers
- `get_lora_parameters(model)` / `get_lora_gradients(model)` - Access LoRA weights
- `merge_lora_weights(base, A, B, scaling)` - Merge LoRA into base for inference
- `count_lora_parameters(model)` - Count trainable LoRA parameters

**Paper References:**

- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

---

### `src/utils.py`

Training utilities.

**Classes:**

#### `TextDataset`

Language modeling dataset.

- Creates input-target pairs with configurable sequence length
- `__getitem__(idx)` returns `(input_ids, target_ids)` shifted by 1

#### `DataLoader`

Batched data iteration.

- `__iter__()` yields batches of (inputs, targets)
- Supports shuffling

**Functions:**

- `create_batches(data, batch_size, drop_last)` - Split data into batches
- `get_batch(data, batch_size, index)` - Get specific batch
- `save_checkpoint(model, filepath, step)` - Save model state
- `load_checkpoint(model, filepath)` - Restore model state
- `download_shakespeare(data_dir)` - Download training corpus
- `create_qa_pairs_from_shakespeare(text, num_pairs)` - Generate Q&A fine-tuning data

---

## Training Pipeline

### 1. Pre-training (`train_pretrain.py`)

Trains the model on raw text using next-token prediction.

```python
# Pseudocode
tokenizer = BPETokenizer()
tokenizer.train(text, vocabulary_size=2000)

model = GPTModel(config)
optimizer = AdamW(learning_rate=3e-4)

for batch in dataloader:
    logits = model.forward(inputs)
    loss = cross_entropy_loss(logits, targets)
    gradients = model.backward(cross_entropy_loss_backward(logits, targets))
    optimizer.step(gradients)
```

### 2. Full Fine-tuning (`train_finetune_full.py`)

Updates all model parameters on task-specific data.

- Loads pre-trained checkpoint
- Trains on Q&A formatted data
- Uses smaller learning rate (1e-4)

### 3. LoRA Fine-tuning (`train_finetune_lora.py`)

Parameter-efficient fine-tuning.

- Freezes base model weights
- Adds LoRA adapters to query/value projections
- Trains only ~1% of parameters
- Achieves similar performance to full fine-tuning

---

## Default Model Configuration

```python
GPTConfig(
    vocab_size=2000,           # BPE vocabulary
    embedding_dim=128,         # Token embedding dimension
    num_heads=4,               # Attention heads
    num_layers=4,              # Transformer blocks
    ffn_hidden_dim=512,        # FFN hidden size (4x embed)
    max_sequence_length=128    # Context window
)
# Total parameters: ~600K
```

---

## Key Concepts

### Autoregressive Generation

The model predicts one token at a time, feeding each prediction back as input:

```
Input:  "The cat"
Step 1: Model predicts "sat" → "The cat sat"
Step 2: Model predicts "on"  → "The cat sat on"
Step 3: Model predicts "the" → "The cat sat on the"
...
```

### Causal Masking

During training, attention is masked so each position can only attend to previous positions. This ensures the model learns to predict without seeing future tokens.

### Temperature Sampling

Controls randomness in generation:

- `temperature=0.0`: Greedy (always pick highest probability)
- `temperature=1.0`: Sample from full distribution
- `temperature>1.0`: More random/creative

### Top-k Sampling

Restricts sampling to the k most likely tokens, preventing rare token selection.

---

## Data Flow Example

```python
# Input: "Hello world" (as token IDs)
tokens = [5, 42, 17]  # Shape: (batch=1, seq=3)

# 1. Embedding lookup
embeddings = embedding_table[tokens]  # Shape: (1, 3, 128)

# 2. Add positional encoding
x = embeddings + positional_encoding[:3]  # Shape: (1, 3, 128)

# 3. Through transformer blocks
for block in transformer_blocks:
    x = block(x, causal_mask)  # Shape preserved: (1, 3, 128)

# 4. Final projection to vocabulary
logits = x @ output_projection  # Shape: (1, 3, 2000)

# 5. Softmax for probabilities
probs = softmax(logits)  # Shape: (1, 3, 2000)

# Each position predicts the next token:
# Position 0 predicts token at position 1
# Position 1 predicts token at position 2
# Position 2 predicts the next token (for generation)
```

---

## Gradient Flow (Backward Pass)

```
Loss (cross-entropy)
       │
       ▼ grad_logits
Output Projection
       │
       ▼ grad_hidden
Final LayerNorm
       │
       ▼
Transformer Blocks (reverse order)
  │ Each block:
  │ ├─ FFN backward
  │ ├─ LayerNorm backward
  │ ├─ Attention backward (Q, K, V projections)
  │ └─ LayerNorm backward
       │
       ▼
Embedding (accumulates gradients per token)
```

---

## Usage

### Quick Demo (2 minutes)

```bash
python run_demo.py quick
```

### Mac Accelerated Training (Apple Silicon)

For significantly faster training on M1/M2/M3/M4 Macs, use the MLX-accelerated mode:

```bash
# Install MLX (Apple Silicon only)
pip install mlx

# Run with default settings (small model, 10% data)
python run_demo.py mac-accel

# Experiment with larger models and more data
python run_demo.py mac-accel --model-size medium --data-size 0.5
python run_demo.py mac-accel --model-size large --data-size 1.0

# Use absolute data sizes
python run_demo.py mac-accel --model-size large --data-size 500k
```

**Model Size Presets:**
| Size | Parameters | Embedding | Heads | Layers | Best For |
|--------|------------|-----------|-------|--------|----------|
| tiny | ~100K | 64 | 2 | 2 | Quick iteration |
| small | ~500K | 128 | 4 | 4 | Default, balanced |
| medium | ~2M | 256 | 8 | 6 | Better quality |
| large | ~8M | 512 | 8 | 8 | Best quality |

**Data Size Options:**

- Fraction: `0.1` (10% of dataset), `0.5` (50%), `1.0` (full)
- Absolute: `50000` (50K characters)
- Suffix: `100k`, `500k`, `1m`

The purpose of this mode is to demonstrate:

1. How larger models improve text generation quality
2. How more training data improves model performance
3. The speed difference between NumPy (single CPU) and MLX (GPU)

Open Activity Monitor to see GPU utilization during training.

### Full Training Pipeline

```bash
# Pre-train on Shakespeare
python train_pretrain.py

# Fine-tune (choose one)
python train_finetune_full.py   # All parameters
python train_finetune_lora.py   # LoRA (1% parameters)

# Interactive generation
python run_demo.py generate
```

### Running Tests

```bash
pytest tests/ -v  # 139 tests (MLX tests skip on non-Apple systems)
```

---

## Dependencies

### Core (NumPy-based, all platforms)

- Python 3.10+
- NumPy (all computations)
- pytest (testing only)

### Optional (Mac Accelerated mode)

- mlx (Apple Silicon only - M1/M2/M3/M4 Macs)

No PyTorch, TensorFlow, or other ML frameworks required.

---

## File Relationships

```
tokenizer.py ─────────────────────────────────────────┐
                                                      │
activations.py ──────┬───────────────────────────────┤
                     │                                │
layers.py ───────────┼─── attention.py ───┬──────────┤
                     │                    │          │
                     └─── transformer.py ─┘          │
                              │                      │
                              └─────── model.py ─────┤
                                          │         │
                              lora.py ────┘         │
                                                    │
optimizer.py ───────────────────────────────────────┤
                                                    │
utils.py ───────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     train_pretrain.py  train_finetune_*.py  run_demo.py
```

---

## Paper References

1. **Attention Mechanism**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **GPT Architecture**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
3. **BPE Tokenization**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
4. **GELU Activation**: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)
5. **Layer Normalization**: "Layer Normalization" (Ba et al., 2016)
6. **AdamW Optimizer**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
7. **LoRA Fine-tuning**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
