#!/usr/bin/env python3
"""
Pretrain Script for GPT Language Model

This script demonstrates how to train a GPT model from scratch using
the pure NumPy implementation. It downloads the Shakespeare dataset
and trains a small language model.

Usage:
    python train_pretrain.py

The script will:
1. Download the Shakespeare dataset
2. Train a BPE tokenizer on the data
3. Create a GPT model with the specified configuration
4. Train the model using language modeling objective
5. Save checkpoints periodically

Training Configuration:
    - Dataset: Shakespeare (Tiny Shakespeare ~1MB)
    - Tokenizer: BPE with ~2000 vocabulary
    - Model: 4 layers, 128 embedding dim, 4 heads
    - Optimizer: AdamW with warmup and cosine decay
"""

import os

# Add parent directory to path if running as script
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    GPTConfig,
    GPTModel,
    cross_entropy_loss,
    cross_entropy_loss_backward,
)
from src.optimizer import AdamW, clip_gradient_norm, get_learning_rate_with_warmup
from src.tokenizer import BPETokenizer
from src.utils import DataLoader, TextDataset, download_shakespeare, save_checkpoint


def train_step(
    model: GPTModel,
    optimizer: AdamW,
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Perform a single training step.

    Args:
        model: GPT model
        optimizer: AdamW optimizer
        inputs: Input token IDs, shape (batch_size, sequence_length)
        targets: Target token IDs, shape (batch_size, sequence_length)
        learning_rate: Current learning rate
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Loss value for this step
    """
    # Forward pass
    logits = model.forward(inputs)

    # Compute loss
    loss = cross_entropy_loss(logits, targets)

    # Backward pass
    grad_logits = cross_entropy_loss_backward(logits, targets)
    gradients = model.backward(grad_logits)

    # Clip gradients
    gradients = clip_gradient_norm(gradients, max_grad_norm)

    # Update parameters
    optimizer.step(gradients, learning_rate=learning_rate)

    return loss


def evaluate(model: GPTModel, data_loader: DataLoader, max_batches: int = 10) -> float:
    """
    Evaluate model on validation data.

    Args:
        model: GPT model
        data_loader: Validation data loader
        max_batches: Maximum number of batches to evaluate

    Returns:
        Average loss over evaluation batches
    """
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in data_loader:
        if num_batches >= max_batches:
            break

        logits = model.forward(inputs)
        loss = cross_entropy_loss(logits, targets)
        total_loss += loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def generate_sample(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
) -> str:
    """
    Generate a text sample from the model.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_array = np.array([prompt_tokens], dtype=np.int64)

    # Generate
    generated = model.generate(
        prompt_array, max_new_tokens=max_new_tokens, temperature=temperature
    )

    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())

    return generated_text


def main():
    """Main training function."""
    print("=" * 60)
    print("GPT Language Model Pretraining")
    print("=" * 60)
    print()

    # ==================== Configuration ====================
    # Training hyperparameters
    batch_size = 32
    sequence_length = 64
    num_epochs = 3
    learning_rate = 3e-4
    warmup_steps = 100
    max_grad_norm = 1.0

    # Model configuration
    model_config = GPTConfig(
        vocab_size=2000,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_hidden_dim=512,
        max_sequence_length=128,
    )

    # Paths
    data_dir = "data"
    checkpoint_dir = "checkpoints"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ==================== Data Loading ====================
    print("Loading data...")

    # Download Shakespeare dataset
    data_path = download_shakespeare(data_dir)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters of text")

    # Split into train/val
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")
    print()

    # ==================== Tokenizer ====================
    print("Training tokenizer...")

    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocabulary_size=model_config.vocab_size)

    print(f"Vocabulary size: {tokenizer.vocabulary_size}")

    # Save tokenizer
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))
    print(f"Saved tokenizer to {checkpoint_dir}/tokenizer.json")
    print()

    # ==================== Dataset and DataLoader ====================
    print("Creating datasets...")

    train_dataset = TextDataset(
        text=train_text, tokenizer=tokenizer, sequence_length=sequence_length
    )

    val_dataset = TextDataset(
        text=val_text, tokenizer=tokenizer, sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")
    print(f"Batches per epoch: {len(train_loader)}")
    print()

    # ==================== Model ====================
    print("Creating model...")

    # Update vocab size to match tokenizer
    model_config.vocab_size = tokenizer.vocabulary_size
    model = GPTModel(model_config)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"  - Embedding dim: {model_config.embedding_dim}")
    print(f"  - Layers: {model_config.num_layers}")
    print(f"  - Heads: {model_config.num_heads}")
    print(f"  - FFN hidden: {model_config.ffn_hidden_dim}")
    print()

    # ==================== Optimizer ====================
    print("Initializing optimizer...")

    optimizer = AdamW(
        learning_rate=learning_rate, beta1=0.9, beta2=0.999, weight_decay=0.01
    )
    optimizer.initialize(model.get_parameters())

    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print()

    # ==================== Training Loop ====================
    print("Starting training...")
    print("-" * 60)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Get learning rate with warmup and decay
            lr = get_learning_rate_with_warmup(
                global_step, learning_rate, warmup_steps, total_steps
            )

            # Training step
            loss = train_step(
                model,
                optimizer,
                inputs,
                targets,
                learning_rate=lr,
                max_grad_norm=max_grad_norm,
            )

            epoch_loss += loss
            global_step += 1

            # Log progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Step {batch_idx + 1}/{steps_per_epoch} | "
                    f"Loss: {loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / steps_per_epoch

        # Validation
        val_loss = evaluate(model, val_loader)

        print("-" * 60)
        print(f"Epoch {epoch + 1} complete in {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        # Generate sample
        sample_text = generate_sample(
            model, tokenizer, "ROMEO:", max_new_tokens=50, temperature=0.8
        )
        print(f"  Sample: {sample_text[:100]}...")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.npz")
        save_checkpoint(model, checkpoint_path, step=global_step)
        print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, "model_best.npz")
            save_checkpoint(model, best_path, step=global_step)
            print(f"  New best model! Saved to {best_path}")

        print("-" * 60)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoint: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
