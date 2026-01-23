#!/usr/bin/env python3
"""
Full Fine-tuning Script for GPT Language Model

This script demonstrates how to fine-tune all parameters of a pre-trained
GPT model on a specific task (Shakespeare Q&A style).

Full fine-tuning updates ALL model parameters, which:
- Gives maximum flexibility for adaptation
- Requires more memory/compute
- Risk of catastrophic forgetting

Usage:
    python train_finetune_full.py

Prerequisites:
    - A pre-trained model checkpoint (run train_pretrain.py first)
    - Trained tokenizer
"""

import os
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
from src.utils import (
    DataLoader,
    create_qa_pairs_from_shakespeare,
    load_checkpoint,
    save_checkpoint,
)


class QADataset:
    """
    Dataset for Q&A style fine-tuning.

    Formats Q&A pairs as: "<question> <answer>"
    Target is the same sequence shifted by 1 (language modeling).
    """

    def __init__(self, qa_pairs: list, tokenizer: BPETokenizer, max_length: int = 128):
        """
        Initialize Q&A dataset.

        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length (pad/truncate to this)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token_id

        # Prepare all sequences
        self.sequences = []

        for qa in qa_pairs:
            # Format: "Q: <question> A: <answer>"
            text = f"Q: {qa['question']} A: {qa['answer']}"
            tokens = tokenizer.encode(text)

            # Truncate or pad
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif len(tokens) < max_length:
                tokens = tokens + [self.pad_token] * (max_length - len(tokens))

            self.sequences.append(np.array(tokens, dtype=np.int64))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        return input_ids, target_ids


def train_step(
    model: GPTModel,
    optimizer: AdamW,
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    ignore_index: int = 0,  # PAD token
) -> float:
    """
    Perform a single fine-tuning step.
    """
    # Forward pass
    logits = model.forward(inputs)

    # Compute loss (ignore padding)
    loss = cross_entropy_loss(logits, targets, ignore_index=ignore_index)

    # Backward pass
    grad_logits = cross_entropy_loss_backward(
        logits, targets, ignore_index=ignore_index
    )
    gradients = model.backward(grad_logits)

    # Clip gradients
    gradients = clip_gradient_norm(gradients, max_grad_norm)

    # Update ALL parameters
    optimizer.step(gradients, learning_rate=learning_rate)

    return loss


def main():
    """Main fine-tuning function."""
    print("=" * 60)
    print("GPT Full Fine-tuning")
    print("=" * 60)
    print()

    # ==================== Configuration ====================
    # Fine-tuning hyperparameters (typically smaller than pretraining)
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-4  # Smaller than pretraining
    warmup_steps = 20
    max_grad_norm = 1.0
    max_length = 64

    # Paths
    checkpoint_dir = "checkpoints"
    pretrained_path = os.path.join(checkpoint_dir, "model_best.npz")
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

    # Check for pretrained model
    if not os.path.exists(pretrained_path):
        print(f"Error: Pretrained model not found at {pretrained_path}")
        print("Please run train_pretrain.py first.")
        return

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please run train_pretrain.py first.")
        return

    # ==================== Load Tokenizer ====================
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Vocabulary size: {tokenizer.vocabulary_size}")
    print()

    # ==================== Create Fine-tuning Data ====================
    print("Creating fine-tuning dataset...")

    # Load Shakespeare text for creating Q&A pairs
    data_path = "data/shakespeare.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create Q&A pairs
    qa_pairs = create_qa_pairs_from_shakespeare(text, num_pairs=200)

    # Split into train/val
    split_idx = int(len(qa_pairs) * 0.9)
    train_pairs = qa_pairs[:split_idx]
    val_pairs = qa_pairs[split_idx:]

    print(f"Train Q&A pairs: {len(train_pairs)}")
    print(f"Val Q&A pairs: {len(val_pairs)}")

    # Create datasets
    train_dataset = QADataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = QADataset(val_pairs, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Batches per epoch: {len(train_loader)}")
    print()

    # ==================== Load Pre-trained Model ====================
    print("Loading pre-trained model...")

    # Create model with same config
    model_config = GPTConfig(
        vocab_size=tokenizer.vocabulary_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_hidden_dim=512,
        max_sequence_length=128,
    )
    model = GPTModel(model_config)

    # Load pretrained weights
    step, _ = load_checkpoint(model, pretrained_path)
    print(f"Loaded checkpoint from step {step}")
    print(f"Model parameters: {model.count_parameters():,} (ALL trainable)")
    print()

    # ==================== Optimizer ====================
    print("Initializing optimizer...")

    optimizer = AdamW(
        learning_rate=learning_rate, beta1=0.9, beta2=0.999, weight_decay=0.01
    )
    optimizer.initialize(model.get_parameters())

    total_steps = num_epochs * len(train_loader)
    print(f"Total training steps: {total_steps}")
    print()

    # ==================== Fine-tuning Loop ====================
    print("Starting fine-tuning...")
    print("-" * 60)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Learning rate with warmup
            lr = get_learning_rate_with_warmup(
                global_step,
                learning_rate,
                warmup_steps,
                total_steps,
                min_learning_rate=learning_rate / 10,
            )

            # Training step
            loss = train_step(
                model,
                optimizer,
                inputs,
                targets,
                learning_rate=lr,
                max_grad_norm=max_grad_norm,
                ignore_index=tokenizer.pad_token_id,
            )

            epoch_loss += loss
            global_step += 1

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Validation
        val_loss = 0.0
        for inputs, targets in val_loader:
            logits = model.forward(inputs)
            val_loss += cross_entropy_loss(
                logits, targets, ignore_index=tokenizer.pad_token_id
            )
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {avg_epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, "model_finetuned_full.npz")
            save_checkpoint(model, save_path, step=global_step)
            print(f"  -> New best! Saved to {save_path}")

    print("-" * 60)
    print()

    # ==================== Test Generation ====================
    print("Testing fine-tuned model:")
    print("-" * 60)

    test_prompts = [
        "Q: Write about love in Shakespeare's style: A:",
        "Q: Continue this passage: To be or not A:",
        "Q: Recite a Shakespeare line: A:",
    ]

    for prompt in test_prompts:
        tokens = np.array([tokenizer.encode(prompt)], dtype=np.int64)
        generated = model.generate(tokens, max_new_tokens=30, temperature=0.8)
        output = tokenizer.decode(generated[0].tolist())
        print(f"Input: {prompt}")
        print(f"Output: {output}")
        print()

    print("=" * 60)
    print("Fine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
