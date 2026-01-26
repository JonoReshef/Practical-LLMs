#!/usr/bin/env python3
"""
LLM Explainer Demo Script

This script provides a comprehensive demonstration of the entire LLM lifecycle:
1. Pre-training a language model from scratch
2. Full fine-tuning on a Q&A task
3. LoRA fine-tuning (parameter-efficient)
4. Interactive text generation

The goal is to show how each component works and allow experimentation.

Usage:
    python run_demo.py [mode] [options]

    Modes:
        help      - Show detailed help with examples and instructions
        full      - Run complete demo (pretrain + finetune + generate)
        pretrain  - Only run pre-training
        finetune  - Only run fine-tuning (requires pretrained model)
        generate  - Only run interactive generation (requires model)
        quick     - Quick demo with reduced training (for testing)
        mac-accel - GPU-accelerated training using Apple MLX (M1/M2/M3/M4)

    Options for mac-accel mode:
        --model-size  - Model size: tiny, small, medium, large (default: small)
        --data-size   - Data fraction: 0.0-1.0 or chars like '100k' (default: 0.1)

Examples:
    python run_demo.py help
    python run_demo.py quick
    python run_demo.py mac-accel --model-size medium --data-size 0.5
    python run_demo.py mac-accel --model-size large --data-size 500k
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lora import (
    apply_lora_to_model,
    count_lora_parameters,
    count_total_parameters,
    get_lora_gradients,
    get_lora_parameters,
)
from src.model import (
    GPTConfig,
    GPTModel,
    cross_entropy_loss,
    cross_entropy_loss_backward,
)
from src.optimizer import AdamW, clip_gradient_norm, get_learning_rate_with_warmup
from src.tokenizer import BPETokenizer

# MLX imports (only loaded when needed to avoid import errors on non-Mac systems)
MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as mlx_optim

    from src.model_mlx import (
        GPTConfigMLX,
        GPTModelMLX,
        count_parameters_mlx,
        create_model_mlx,
        cross_entropy_loss_mlx,
    )

    MLX_AVAILABLE = True
except ImportError:
    pass
from src.utils import (
    DataLoader,
    TextDataset,
    create_qa_pairs_from_shakespeare,
    download_shakespeare,
    load_checkpoint,
    save_checkpoint,
)


def print_header(text: str):
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()


def print_section(text: str):
    """Print a section divider."""
    print()
    print("-" * 40)
    print(text)
    print("-" * 40)


def show_help():
    """Display detailed help with examples and instructions."""
    help_text = """
================================================================================
                    LLM EXPLAINER - COMMAND LINE INTERFACE
================================================================================

An educational implementation of a GPT-style language model from scratch.
This CLI provides tools for training, fine-tuning, and generating text.

--------------------------------------------------------------------------------
                              AVAILABLE COMMANDS
--------------------------------------------------------------------------------

  help        Show this detailed help message
  quick       Quick demo with reduced training (2-3 minutes)
  full        Complete demo: pretrain + finetune + generate (15-30 minutes)
  pretrain    Pre-train the model on Shakespeare text
  finetune    Fine-tune a pre-trained model (requires running pretrain first)
  generate    Interactive text generation (requires a trained model)
  mac-accel   GPU-accelerated training using Apple MLX (Apple Silicon only)

--------------------------------------------------------------------------------
                              QUICK START EXAMPLES
--------------------------------------------------------------------------------

  1. First time? Start with a quick demo:
     $ python run_demo.py quick

  2. Run the full training pipeline:
     $ python run_demo.py full

  3. Train a model from scratch:
     $ python run_demo.py pretrain

  4. Generate text with a trained model:
     $ python run_demo.py generate

--------------------------------------------------------------------------------
                         MAC-ACCELERATED MODE (MLX)
--------------------------------------------------------------------------------

  For Apple Silicon Macs (M1/M2/M3/M4), use GPU acceleration for faster training.

  PREREQUISITES:
     $ pip install mlx

  BASIC USAGE:
     $ python run_demo.py mac-accel

  OPTIONS:
     --model-size    Model size preset (tiny, small, medium, large)
     --data-size     Amount of training data to use

  MODEL SIZE PRESETS:
     tiny      ~100K params   - Fast iteration, quick experiments
     small     ~500K params   - Default, balanced speed/quality
     medium    ~2M params     - Better quality, slower training
     large     ~8M params     - Best quality, requires more time/memory

  DATA SIZE FORMATS:
     Fraction:    0.1, 0.5, 1.0         (10%, 50%, 100% of dataset)
     Count:       50000                  (exact character count)
     Suffix:      100k, 500k, 1m        (shorthand for thousands/millions)

  EXAMPLES:
     # Quick test with tiny model
     $ python run_demo.py mac-accel --model-size tiny --data-size 50k

     # Balanced training
     $ python run_demo.py mac-accel --model-size small --data-size 0.2

     # Higher quality with more data
     $ python run_demo.py mac-accel --model-size medium --data-size 0.5

     # Best quality (full dataset, large model)
     $ python run_demo.py mac-accel --model-size large --data-size 1.0

--------------------------------------------------------------------------------
                              WORKFLOW GUIDE
--------------------------------------------------------------------------------

  TYPICAL LEARNING WORKFLOW:

     Step 1: Run quick demo to see how it works
             $ python run_demo.py quick

     Step 2: Explore individual modules (each has educational demos)
             $ python -m src.tokenizer
             $ python -m src.attention
             $ python -m src.transformer
             $ python -m src.model

     Step 3: Train your own model
             $ python run_demo.py pretrain

     Step 4: Experiment with generation
             $ python run_demo.py generate

  FOR APPLE SILICON USERS:

     Step 1: Install MLX
             $ pip install mlx

     Step 2: Compare NumPy vs MLX speed
             $ python run_demo.py quick          # NumPy (CPU)
             $ python run_demo.py mac-accel      # MLX (GPU)

     Step 3: Experiment with model/data sizes
             $ python run_demo.py mac-accel --model-size medium --data-size 0.5

--------------------------------------------------------------------------------
                              FILE STRUCTURE
--------------------------------------------------------------------------------

  After training, files are saved to:

     checkpoints/              NumPy model checkpoints
        tokenizer.json         Trained BPE tokenizer
        model_best.npz         Best model weights

     checkpoints_mlx/          MLX model checkpoints (mac-accel mode)
        tokenizer.json         Trained BPE tokenizer

     data/                     Training data
        shakespeare.txt        Downloaded Shakespeare corpus

--------------------------------------------------------------------------------
                              RUNNING TESTS
--------------------------------------------------------------------------------

     # Run all tests
     $ pytest tests/ -v

     # Run specific test file
     $ pytest tests/test_model.py -v

     # Run with coverage
     $ pytest tests/ --cov=src

--------------------------------------------------------------------------------
                              MORE INFORMATION
--------------------------------------------------------------------------------

  See README.md for:
     - Architecture overview
     - Module documentation
     - Mathematical explanations
     - References to research papers

  GitHub: https://github.com/your-repo/LLM-explainer

================================================================================
"""
    print(help_text)


def download_data(data_dir: str = "data") -> str:
    """Download Shakespeare dataset if not present."""
    print("Checking for Shakespeare dataset...")
    filepath = download_shakespeare(data_dir)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {filepath}")
    return filepath


def train_tokenizer(text: str, vocab_size: int, checkpoint_dir: str) -> BPETokenizer:
    """Train or load BPE tokenizer."""
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"Training new BPE tokenizer with vocab_size={vocab_size}...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocabulary_size=vocab_size)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    print(f"Vocabulary size: {tokenizer.vocabulary_size}")

    # Show tokenization example
    sample = "To be or not to be"
    tokens = tokenizer.encode(sample)
    decoded = tokenizer.decode(tokens)
    print(f"Example: '{sample}' -> {tokens} -> '{decoded}'")

    return tokenizer


def create_model(tokenizer: BPETokenizer, config_dict: dict) -> GPTModel:
    """Create a GPT model."""
    config = GPTConfig(
        vocab_size=tokenizer.vocabulary_size,
        embedding_dim=config_dict.get("embedding_dim", 128),
        num_heads=config_dict.get("num_heads", 4),
        num_layers=config_dict.get("num_layers", 4),
        ffn_hidden_dim=config_dict.get("ffn_hidden_dim", 512),
        max_sequence_length=config_dict.get("max_sequence_length", 128),
    )

    model = GPTModel(config)

    total_params = count_total_parameters(model)
    print(f"Created GPT model with {total_params:,} parameters")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Attention heads: {config.num_heads}")
    print(f"  Transformer layers: {config.num_layers}")
    print(f"  FFN hidden dim: {config.ffn_hidden_dim}")
    print(f"  Max sequence length: {config.max_sequence_length}")

    return model


def pretrain(
    model: GPTModel,
    tokenizer: BPETokenizer,
    text: str,
    config: dict,
    checkpoint_dir: str,
) -> GPTModel:
    """Pre-train the model on text data."""
    print_section("Pre-training")

    # Create dataset
    max_length = config.get("max_sequence_length", 128)
    dataset = TextDataset(text, tokenizer, sequence_length=max_length)

    batch_size = config.get("batch_size", 16)
    num_epochs = config.get("num_epochs", 5)
    learning_rate = config.get("learning_rate", 3e-4)
    warmup_steps = config.get("warmup_steps", 100)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = num_epochs * len(train_loader)

    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Total training steps: {total_steps}")

    # Initialize optimizer
    optimizer = AdamW(learning_rate=learning_rate)
    optimizer.initialize(model.get_parameters())

    # Training loop
    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            # Get learning rate with warmup
            lr = get_learning_rate_with_warmup(
                global_step, learning_rate, warmup_steps, total_steps
            )

            # Forward pass
            logits = model.forward(inputs)
            loss = cross_entropy_loss(logits, targets)

            # Backward pass
            grad_logits = cross_entropy_loss_backward(logits, targets)
            gradients = model.backward(grad_logits)

            # Update
            gradients = clip_gradient_norm(gradients, max_norm=1.0)
            optimizer.step(gradients, learning_rate=lr)

            epoch_loss += loss
            global_step += 1

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, os.path.join(checkpoint_dir, "model_best.npz"), global_step
            )

    print(f"Pre-training complete. Best loss: {best_loss:.4f}")
    return model


def finetune_full(
    model: GPTModel,
    tokenizer: BPETokenizer,
    qa_pairs: list,
    config: dict,
    checkpoint_dir: str,
) -> GPTModel:
    """Full fine-tuning on Q&A data."""
    print_section("Full Fine-tuning")

    # Create Q&A dataset
    max_length = config.get("max_sequence_length", 64)
    batch_size = config.get("batch_size", 16)
    num_epochs = config.get("num_epochs", 5)
    learning_rate = config.get("learning_rate", 1e-4)

    # Prepare sequences
    sequences = []
    for qa in qa_pairs:
        text = f"Q: {qa['question']} A: {qa['answer']}"
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        sequences.append(np.array(tokens, dtype=np.int64))

    print(f"Q&A pairs: {len(sequences)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")

    # Initialize optimizer
    optimizer = AdamW(learning_rate=learning_rate)
    optimizer.initialize(model.get_parameters())

    total_steps = num_epochs * (len(sequences) // batch_size + 1)
    warmup_steps = config.get("warmup_steps", 20)

    # Training loop
    global_step = 0

    for epoch in range(num_epochs):
        np.random.shuffle(sequences)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            if len(batch) < 2:
                continue

            batch = np.stack(batch)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            lr = get_learning_rate_with_warmup(
                global_step, learning_rate, warmup_steps, total_steps
            )

            logits = model.forward(inputs)
            loss = cross_entropy_loss(
                logits, targets, ignore_index=tokenizer.pad_token_id
            )

            grad_logits = cross_entropy_loss_backward(
                logits, targets, ignore_index=tokenizer.pad_token_id
            )
            gradients = model.backward(grad_logits)

            gradients = clip_gradient_norm(gradients, max_norm=1.0)
            optimizer.step(gradients, learning_rate=lr)

            epoch_loss += loss
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    save_checkpoint(
        model, os.path.join(checkpoint_dir, "model_finetune_full.npz"), global_step
    )
    print("Full fine-tuning complete.")
    return model


def finetune_lora(
    model: GPTModel,
    tokenizer: BPETokenizer,
    qa_pairs: list,
    config: dict,
    checkpoint_dir: str,
) -> GPTModel:
    """LoRA fine-tuning (parameter-efficient)."""
    print_section("LoRA Fine-tuning")

    # LoRA configuration
    lora_rank = config.get("lora_rank", 4)
    lora_alpha = config.get("lora_alpha", 8)
    target_modules = config.get("target_modules", ["query", "value"])

    print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
    print(f"Target modules: {target_modules}")

    total_params = count_total_parameters(model)
    model = apply_lora_to_model(
        model, rank=lora_rank, alpha=lora_alpha, target_modules=target_modules
    )
    lora_params = count_lora_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Parameter reduction: {100 * (1 - lora_params / total_params):.1f}%")

    # Prepare data
    max_length = config.get("max_sequence_length", 64)
    batch_size = config.get("batch_size", 16)
    num_epochs = config.get("num_epochs", 10)
    learning_rate = config.get("learning_rate", 3e-4)

    sequences = []
    for qa in qa_pairs:
        text = f"Q: {qa['question']} A: {qa['answer']}"
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        sequences.append(np.array(tokens, dtype=np.int64))

    # Initialize optimizer for LoRA parameters only
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.0)
    optimizer.initialize(get_lora_parameters(model))

    total_steps = num_epochs * (len(sequences) // batch_size + 1)
    warmup_steps = config.get("warmup_steps", 20)

    # Training loop
    global_step = 0

    for epoch in range(num_epochs):
        np.random.shuffle(sequences)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            if len(batch) < 2:
                continue

            batch = np.stack(batch)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            lr = get_learning_rate_with_warmup(
                global_step, learning_rate, warmup_steps, total_steps
            )

            logits = model.forward(inputs)
            loss = cross_entropy_loss(
                logits, targets, ignore_index=tokenizer.pad_token_id
            )

            grad_logits = cross_entropy_loss_backward(
                logits, targets, ignore_index=tokenizer.pad_token_id
            )
            model.backward(grad_logits)

            # Only get LoRA gradients
            lora_grads = get_lora_gradients(model)
            lora_grads = clip_gradient_norm(lora_grads, max_norm=1.0)
            optimizer.step(lora_grads, learning_rate=lr)

            epoch_loss += loss
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    print("LoRA fine-tuning complete.")
    return model


def generate_interactive(model: GPTModel, tokenizer: BPETokenizer):
    """Interactive text generation."""
    print_header("Interactive Text Generation")
    print("Enter a prompt and the model will continue it.")
    print("Type 'quit' to exit.")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not prompt:
            continue

        # Tokenize
        tokens = np.array([tokenizer.encode(prompt)], dtype=np.int64)

        # Generate
        generated = model.generate(tokens, max_new_tokens=50, temperature=0.8, top_k=50)

        # Decode
        output = tokenizer.decode(generated[0].tolist())
        print(f"Model: {output}")
        print()


def demo_generation(model: GPTModel, tokenizer: BPETokenizer):
    """Demonstrate text generation with various prompts."""
    print_section("Sample Generations")

    prompts = [
        "To be or not to be",
        "The king said",
        "In the forest",
        "Love is",
        "Q: Write about Shakespeare A:",
    ]

    for prompt in prompts:
        tokens = np.array([tokenizer.encode(prompt)], dtype=np.int64)
        generated = model.generate(tokens, max_new_tokens=30, temperature=0.8)
        output = tokenizer.decode(generated[0].tolist())
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print()


def run_quick_demo():
    """Run a quick demo with minimal training for testing."""
    print_header("Quick Demo Mode (Minimal Training)")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Download and prepare data
    data_path = download_data()
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Use smaller subset for quick demo
    text = text[:50000]  # First 50k characters

    # Train tokenizer
    tokenizer = train_tokenizer(text, vocab_size=500, checkpoint_dir=checkpoint_dir)

    # Create smaller model
    config = {
        "embedding_dim": 64,
        "num_heads": 2,
        "num_layers": 2,
        "ffn_hidden_dim": 128,
        "max_sequence_length": 64,
        "batch_size": 8,
        "num_epochs": 2,
        "learning_rate": 1e-3,
        "warmup_steps": 10,
    }

    print_section("Creating Model")
    model = create_model(tokenizer, config)

    # Quick pre-training
    config["num_epochs"] = 2
    model = pretrain(model, tokenizer, text, config, checkpoint_dir)

    # Quick demo generation
    demo_generation(model, tokenizer)

    # Create Q&A pairs and do quick LoRA demo
    qa_pairs = create_qa_pairs_from_shakespeare(text, num_pairs=20)

    config["num_epochs"] = 3
    config["lora_rank"] = 2
    config["lora_alpha"] = 4
    model = finetune_lora(model, tokenizer, qa_pairs, config, checkpoint_dir)

    demo_generation(model, tokenizer)

    print_header("Quick Demo Complete!")


def run_full_demo():
    """Run the full demonstration."""
    print_header("Full LLM Demo")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Download data
    data_path = download_data()
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Train tokenizer
    tokenizer = train_tokenizer(text, vocab_size=2000, checkpoint_dir=checkpoint_dir)

    # Model configuration
    config = {
        "embedding_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
        "ffn_hidden_dim": 512,
        "max_sequence_length": 128,
        "batch_size": 16,
        "num_epochs": 5,
        "learning_rate": 3e-4,
        "warmup_steps": 100,
    }

    print_section("Creating Model")
    model = create_model(tokenizer, config)

    # Pre-training
    model = pretrain(model, tokenizer, text, config, checkpoint_dir)

    print_section("Post Pre-training Generation")
    demo_generation(model, tokenizer)

    # Create Q&A data
    qa_pairs = create_qa_pairs_from_shakespeare(text, num_pairs=200)

    # Full fine-tuning
    finetune_config = {
        **config,
        "num_epochs": 5,
        "learning_rate": 1e-4,
        "max_sequence_length": 64,
    }
    model_ft = create_model(tokenizer, config)
    load_checkpoint(model_ft, os.path.join(checkpoint_dir, "model_best.npz"))
    model_ft = finetune_full(
        model_ft, tokenizer, qa_pairs, finetune_config, checkpoint_dir
    )

    print_section("Post Full Fine-tuning Generation")
    demo_generation(model_ft, tokenizer)

    # LoRA fine-tuning
    lora_config = {
        **config,
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "max_sequence_length": 64,
        "lora_rank": 4,
        "lora_alpha": 8,
        "target_modules": ["query", "value"],
    }
    model_lora = create_model(tokenizer, config)
    load_checkpoint(model_lora, os.path.join(checkpoint_dir, "model_best.npz"))
    model_lora = finetune_lora(
        model_lora, tokenizer, qa_pairs, lora_config, checkpoint_dir
    )

    print_section("Post LoRA Fine-tuning Generation")
    demo_generation(model_lora, tokenizer)

    # Interactive mode
    print("\nWould you like to enter interactive mode? (y/n)")
    try:
        response = input().strip().lower()
        if response == "y":
            generate_interactive(model_lora, tokenizer)
    except (EOFError, KeyboardInterrupt):
        pass

    print_header("Full Demo Complete!")


def parse_data_size(data_size_str: str, total_chars: int) -> int:
    """
    Parse data size specification to number of characters.

    Args:
        data_size_str: Either a fraction (0.0-1.0) or a size like '100k', '1m'
        total_chars: Total characters available in the dataset

    Returns:
        Number of characters to use
    """
    data_size_str = data_size_str.lower().strip()

    # Check for k/m suffix
    if data_size_str.endswith("k"):
        return int(float(data_size_str[:-1]) * 1000)
    elif data_size_str.endswith("m"):
        return int(float(data_size_str[:-1]) * 1000000)

    # Try as fraction
    try:
        fraction = float(data_size_str)
        if 0.0 < fraction <= 1.0:
            return int(total_chars * fraction)
        elif fraction > 1.0:
            # Treat as absolute character count
            return int(fraction)
    except ValueError:
        pass

    raise ValueError(
        f"Invalid data size: {data_size_str}. Use fraction (0.1), count (50000), or suffix (100k, 1m)"
    )


def run_mac_accel_demo(model_size: str = "small", data_size: str = "0.1"):
    """
    Run GPU-accelerated training using Apple MLX.

    This demonstrates how more data and larger models improve performance,
    leveraging the M1/M2/M3/M4 GPU for significant speedups.

    Args:
        model_size: One of "tiny", "small", "medium", "large"
        data_size: Data to use - fraction (0.1), count (50000), or suffix (100k)
    """
    if not MLX_AVAILABLE:
        print("=" * 60)
        print("ERROR: MLX is not available")
        print("=" * 60)
        print()
        print("MLX is required for mac-accel mode. To install:")
        print("  pip install mlx")
        print()
        print("Note: MLX only works on Apple Silicon Macs (M1/M2/M3/M4).")
        print("For Intel Macs or other systems, use 'quick' or 'full' mode.")
        return

    print_header("Mac Accelerated Training (MLX)")
    print(f"Model size: {model_size}")
    print(f"Data size: {data_size}")
    print()
    print("This mode uses Apple MLX for GPU-accelerated training.")
    print("Compare results with 'quick' mode to see quality vs speed tradeoffs.")
    print()

    checkpoint_dir = "checkpoints_mlx"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Download and prepare data
    data_path = download_data()
    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Parse and apply data size
    total_chars = len(full_text)
    use_chars = parse_data_size(data_size, total_chars)
    use_chars = min(use_chars, total_chars)  # Cap at available data

    text = full_text[:use_chars]
    print(
        f"Using {len(text):,} characters ({100 * len(text) / total_chars:.1f}% of dataset)"
    )
    print()

    # Train tokenizer
    vocab_size = 1000 if model_size == "tiny" else 2000
    tokenizer = train_tokenizer(
        text, vocab_size=vocab_size, checkpoint_dir=checkpoint_dir
    )

    # Create MLX model
    print_section("Creating MLX Model")
    model, config = create_model_mlx(tokenizer.vocabulary_size, size=model_size)
    num_params = count_parameters_mlx(model)

    print(f"Model configuration ({model_size}):")
    print(f"  Parameters: {num_params:,}")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Attention heads: {config.num_heads}")
    print(f"  Transformer layers: {config.num_layers}")
    print(f"  FFN hidden dim: {config.ffn_hidden_dim}")
    print(f"  Max sequence length: {config.max_sequence_length}")
    print()

    # Prepare dataset
    print_section("Pre-training with MLX")
    max_length = config.max_sequence_length

    # Tokenize all text and create training sequences
    all_tokens = tokenizer.encode(text)
    sequences = []
    for i in range(0, len(all_tokens) - max_length, max_length // 2):
        seq = all_tokens[i : i + max_length + 1]
        if len(seq) == max_length + 1:
            sequences.append(seq)

    print(f"Training sequences: {len(sequences)}")

    # Training configuration based on model size
    batch_sizes = {"tiny": 32, "small": 16, "medium": 8, "large": 4}
    epoch_counts = {"tiny": 10, "small": 8, "medium": 5, "large": 15}

    batch_size = batch_sizes.get(model_size, 16)
    num_epochs = epoch_counts.get(model_size, 5)
    learning_rate = 3e-4
    warmup_steps = min(100, len(sequences) // batch_size)

    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Initialize optimizer
    optimizer = mlx_optim.AdamW(learning_rate=learning_rate)

    # Create loss function for value_and_grad
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        return cross_entropy_loss_mlx(logits, targets)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    global_step = 0
    total_steps = num_epochs * (len(sequences) // batch_size)
    best_loss = float("inf")

    training_start = time.time()

    for epoch in range(num_epochs):
        # Shuffle sequences
        np.random.shuffle(sequences)

        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            if len(batch_seqs) < 2:
                continue

            # Convert to MLX arrays
            batch = mx.array(np.array(batch_seqs, dtype=np.int32))
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            # Forward and backward with automatic differentiation
            loss, grads = loss_and_grad_fn(model, inputs, targets)

            # Learning rate warmup
            if global_step < warmup_steps:
                lr = learning_rate * (global_step + 1) / warmup_steps
            else:
                # Cosine decay
                progress = (global_step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                lr = learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

            # Update with current learning rate
            optimizer.learning_rate = lr
            optimizer.update(model, grads)

            # Evaluate to trigger computation
            mx.eval(model.parameters(), optimizer.state, loss)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        tokens_per_sec = (num_batches * batch_size * max_length) / epoch_time

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s | Tokens/s: {tokens_per_sec:.0f}"
        )

        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss

    total_time = time.time() - training_start
    print()
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best loss: {best_loss:.4f}")
    print()

    # Demo generation with MLX model
    print_section("Sample Generations (MLX)")

    prompts = [
        "To be or not to be",
        "The king said",
        "In the forest",
        "Love is",
    ]

    for prompt in prompts:
        tokens = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)

        # Ensure we don't exceed max sequence length
        if tokens.shape[1] >= config.max_sequence_length:
            tokens = tokens[:, : config.max_sequence_length - 1]

        gen_start = time.time()
        generated = model.generate(tokens, max_new_tokens=30, temperature=0.8, top_k=50)
        gen_time = time.time() - gen_start

        output = tokenizer.decode(generated[0].tolist())
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print(f"  (Generated in {gen_time * 1000:.0f}ms)")
        print()

    # Performance summary
    print_header("Performance Summary")
    print(f"Model: {model_size} ({num_params:,} parameters)")
    print(f"Data: {len(text):,} characters ({len(sequences)} sequences)")
    print(f"Training time: {total_time:.1f}s")
    print(f"Final loss: {best_loss:.4f}")
    print()
    print("Experiment suggestions:")
    print("  - Try --model-size large --data-size 1.0 for best quality")
    print("  - Compare with 'python run_demo.py quick' to see speedup")
    print("  - Use Activity Monitor to see GPU utilization")
    print()

    print_header("Mac Accelerated Demo Complete!")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Explainer Demo - Educational Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
For detailed help and examples, run:
  python run_demo.py help
        """,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="quick",
        choices=[
            "help",
            "full",
            "quick",
            "pretrain",
            "finetune",
            "generate",
            "mac-accel",
        ],
        help="Demo mode to run (use 'help' for detailed instructions)",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Model size for mac-accel mode (default: small)",
    )
    parser.add_argument(
        "--data-size",
        default="0.1",
        help="Data to use: fraction (0.1), count (50000), or suffix (100k, 1m). Default: 0.1",
    )
    args = parser.parse_args()

    # Handle help command first (before seed and header)
    if args.mode == "help":
        show_help()
        return

    np.random.seed(42)

    print_header("LLM Explainer - Educational Language Model Demo")
    print("This demo shows how to train, fine-tune, and run a language model.")
    print(f"Mode: {args.mode}")

    if args.mode == "quick":
        run_quick_demo()
    elif args.mode == "full":
        run_full_demo()
    elif args.mode == "pretrain":
        # Just run pre-training
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        data_path = download_data()
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = train_tokenizer(
            text, vocab_size=2000, checkpoint_dir=checkpoint_dir
        )
        config = {
            "embedding_dim": 128,
            "num_heads": 4,
            "num_layers": 4,
            "ffn_hidden_dim": 512,
            "max_sequence_length": 128,
            "batch_size": 16,
            "num_epochs": 5,
            "learning_rate": 3e-4,
            "warmup_steps": 100,
        }
        model = create_model(tokenizer, config)
        pretrain(model, tokenizer, text, config, checkpoint_dir)
        demo_generation(model, tokenizer)
    elif args.mode == "generate":
        # Interactive generation with existing model
        checkpoint_dir = "checkpoints"
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
        model_path = os.path.join(checkpoint_dir, "model_best.npz")

        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            print("Error: No trained model found. Run 'pretrain' first.")
            return

        tokenizer = BPETokenizer.load(tokenizer_path)
        config = GPTConfig(
            vocab_size=tokenizer.vocabulary_size,
            embedding_dim=128,
            num_heads=4,
            num_layers=4,
            ffn_hidden_dim=512,
            max_sequence_length=128,
        )
        model = GPTModel(config)
        load_checkpoint(model, model_path)
        generate_interactive(model, tokenizer)
    elif args.mode == "mac-accel":
        run_mac_accel_demo(
            model_size=args.model_size,
            data_size=args.data_size,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
