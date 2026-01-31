# LLM Explainer

An educational implementation of a GPT-style language model from scratch using pure NumPy. This project demonstrates how Large Language Models (LLMs) work at every level, from tokenization to training to generation.

## Purpose

This codebase is designed to teach the internals of transformer-based language models through human-readable code with extensive documentation. Every component includes:

- Detailed docstrings explaining the mathematical operations
- References to original research papers
- Clear variable naming that maps to paper notation

## Setup

This works on a mac/linux machine.

```bash
uv sync
```

## Key docs

For a deep understanding of each component, we provide comprehensive documentation files with:

- Mathematical formulas and derivations
- Step-by-step numeric examples
- Visualizations and diagrams
- References to research papers

| Topic                   | Documentation                                                   | Code                 |
| ----------------------- | --------------------------------------------------------------- | -------------------- |
| Tokenization (BPE)      | [Tokenization.md](Tutorial/1%20-%20Tokenization.md)             | `src/tokenizer.py`   |
| Word Embeddings         | [Embeddings.md](Tutorial/2%20-%20Embeddings.md)                 | `src/layers.py`      |
| Positional Encoding     | [PositionalEncoding.md](Tutorial/3%20-%20PositionalEncoding.md) | `src/layers.py`      |
| Attention Mechanism     | [Attention.md](Tutorial/4%20-%20Attention.md)                   | `src/attention.py`   |
| Feed-Forward Network    | [FeedForwardNetwork.md](Tutorial/5%20-%20FeedForwardNetwork.md) | `src/transformer.py` |
| Transformer Block       | [TransformerBlock.md](Tutorial/6%20-%20TransformerBlock.md)     | `src/transformer.py` |
| Full GPT Model          | [GPTModel.md](Tutorial/7%20-%20GPTModel.md)                     | `src/model.py`       |
| Training & Optimization | [Training.md](Tutorial/8%20-%20Training.md)                     | `src/optimizer.py`   |
| Text Generation         | [TextGeneration.md](Tutorial/9%20-%20TextGeneration.md)         | `src/model.py`       |
| Fine-Tuning & LoRA      | [FineTuning.md](Tutorial/10%20-%20FineTuning.md)                | `src/lora.py`        |

Start with [How to learn with this.md](How%20to%20learn%20with%20this.md) for a guided learning path.

## Usage

### What can you run?

```bash
uv run run_demo.py --help
```

### Quick Demo (2 minutes)

```bash
uv run run_demo.py quick
```

### Full Demo (15 minutes)

```bash
uv run run_demo.py full
```

### Mac Accelerated Training (Apple Silicon)

**NOTE** this is experimental and does not currently have parity with the NumPy mode.

For significantly faster training on M1/M2/M3/M4 Macs, use the MLX-accelerated mode:

```bash
# Run with default settings (small model, 10% data)
uv run run_demo.py mac-accel

# Experiment with larger models and more data
uv run run_demo.py mac-accel --model-size medium --data-size 0.5
uv run run_demo.py mac-accel --model-size large --data-size 1.0

# Use absolute data sizes
uv run run_demo.py mac-accel --model-size large --data-size 500k
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
uv run train_pretrain.py

# Fine-tune (choose one)
uv run train_finetune_full.py   # All parameters
uv run train_finetune_lora.py   # LoRA (1% parameters)
# Interactive generation
uv run run_demo.py generate
```

### Running Tests

```bash
uv run pytest tests/ -v  # 139 tests (MLX tests skip on non-Apple systems)
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
