"""
Educational LLM Implementation from Scratch

This package provides a complete, working implementation of a transformer-based
language model using only NumPy. It is designed for educational purposes to help
understand how LLMs work at the fundamental level.

Modules:
    activations: Activation functions (softmax, GELU, etc.)
    layers: Neural network layers (Linear, LayerNorm, Embedding)
    attention: Multi-head self-attention mechanism
    transformer: Transformer blocks and encoder stacks
    model: Complete GPT-style language model
    tokenizer: BPE (Byte Pair Encoding) tokenizer
    optimizer: AdamW optimizer implementation
    lora: Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
    utils: Data loading, batching, and model saving/loading utilities

Reference:
    "Attention Is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762
"""

__version__ = "1.0.0"
__author__ = "Educational LLM Project"
