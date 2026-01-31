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

Start with [How to learn with this.md](How%20to%20learn%20with%20this.md) for a guided learning path and high level summaries of each section.

| Topic                   | Documentation                                                    | Code                 |
| ----------------------- | ---------------------------------------------------------------- | -------------------- |
| Tokenization (BPE)      | [Tokenization.md](Tutorial/01%20-%20Tokenization.md)             | `src/tokenizer.py`   |
| Word Embeddings         | [Embeddings.md](Tutorial/02%20-%20Embeddings.md)                 | `src/layers.py`      |
| Positional Encoding     | [PositionalEncoding.md](Tutorial/03%20-%20PositionalEncoding.md) | `src/layers.py`      |
| Attention Mechanism     | [Attention.md](Tutorial/04%20-%20Attention.md)                   | `src/attention.py`   |
| Feed-Forward Network    | [FeedForwardNetwork.md](Tutorial/05%20-%20FeedForwardNetwork.md) | `src/transformer.py` |
| Transformer Block       | [TransformerBlock.md](Tutorial/06%20-%20TransformerBlock.md)     | `src/transformer.py` |
| Full GPT Model          | [GPTModel.md](Tutorial/07%20-%20GPTModel.md)                     | `src/model.py`       |
| Training & Optimization | [Training.md](Tutorial/08%20-%20Training.md)                     | `src/optimizer.py`   |
| Text Generation         | [TextGeneration.md](Tutorial/09%20-%20TextGeneration.md)         | `src/model.py`       |
| Fine-Tuning & LoRA      | [FineTuning.md](Tutorial/10%20-%20FineTuning.md)                 | `src/lora.py`        |

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

## File Relationships

Ideally each section would be independent to isolate concepts, but in reality some concepts are dependent on others. Here's a diagram of the main dependencies:

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

## References

### Core Transformer Architecture

1. **Attention Mechanism**: [Vaswani, A., et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. **Pre-LN Transformer**: [Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
3. **Residual Connections**: [He, K., et al. (2015). Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
4. **Layer Normalization**: [Ba, J., et al. (2016). Layer Normalization](https://arxiv.org/abs/1607.06450)

### GPT Models

5. **GPT-1**: [Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised)
6. **GPT-2**: [Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models)
7. **GPT-3**: [Brown, T., et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
8. **Weight Tying**: [Press, O., & Wolf, L. (2017). Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)

### Tokenization

9. **BPE Tokenization**: [Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
10. **SentencePiece**: [Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)

### Embeddings & Positional Encoding

11. **Word2Vec**: [Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
12. **GloVe**: [Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
13. **Neural Language Models**: [Bengio, Y., et al. (2003). A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
14. **Relative Position**: [Shaw, P., et al. (2018). Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
15. **Rotary Embeddings**: [Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

### Activation Functions

16. **GELU Activation**: [Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
17. **GLU Variants**: [Shazeer, N. (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

### Training & Optimization

18. **Adam Optimizer**: [Kingma, D.P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
19. **AdamW Optimizer**: [Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
20. **Learning Rate Warmup**: [Goyal, P., et al. (2017). Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)
21. **Gradient Clipping**: [Pascanu, R., et al. (2013). On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)

### Text Generation & Sampling

22. **Top-K Sampling**: [Fan, A., et al. (2018). Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
23. **Nucleus (Top-P) Sampling**: [Holtzman, A., et al. (2019). The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
24. **Sampling Strategies**: [Ippolito, D., et al. (2019). Comparison of Diverse Decoding Methods](https://arxiv.org/abs/1909.00459)

### Fine-Tuning & Adaptation

25. **LoRA**: [Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
26. **QLoRA**: [Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
27. **Parameter-Efficient Transfer**: [He, J., et al. (2021). Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)

### Additional Resources

28. **The Illustrated Transformer**: [Jay Alammar's Visual Guide](https://jalammar.github.io/illustrated-transformer/)
29. **Attention Mechanism Origins**: [Bahdanau, D., et al. (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
