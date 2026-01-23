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
    python run_demo.py [mode]

    Modes:
        full      - Run complete demo (pretrain + finetune + generate)
        pretrain  - Only run pre-training
        finetune  - Only run fine-tuning (requires pretrained model)
        generate  - Only run interactive generation (requires model)
        quick     - Quick demo with reduced training (for testing)

Example:
    python run_demo.py quick
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="LLM Explainer Demo")
    parser.add_argument(
        "mode",
        nargs="?",
        default="quick",
        choices=["full", "quick", "pretrain", "finetune", "generate"],
        help="Demo mode to run",
    )
    args = parser.parse_args()

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

    print("\nDone!")


if __name__ == "__main__":
    main()
