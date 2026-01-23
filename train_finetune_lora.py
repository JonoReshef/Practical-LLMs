#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for GPT Language Model

This script demonstrates parameter-efficient fine-tuning using LoRA
(Low-Rank Adaptation). LoRA freezes the pre-trained model weights
and only trains small adapter matrices.

Advantages of LoRA:
    - Much fewer trainable parameters (e.g., 1% of full model)
    - Prevents catastrophic forgetting
    - Multiple adapters can be stored/swapped efficiently
    - Similar or better performance to full fine-tuning

Usage:
    python train_finetune_lora.py

Prerequisites:
    - A pre-trained model checkpoint (run train_pretrain.py first)
    - Trained tokenizer
"""

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
    set_lora_parameters,
)
from src.model import (
    GPTConfig,
    GPTModel,
    cross_entropy_loss,
    cross_entropy_loss_backward,
)
from src.optimizer import AdamW, clip_gradient_norm, get_learning_rate_with_warmup
from src.tokenizer import BPETokenizer
from src.utils import DataLoader, create_qa_pairs_from_shakespeare, load_checkpoint


class QADataset:
    """Dataset for Q&A style fine-tuning."""

    def __init__(self, qa_pairs: list, tokenizer: BPETokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token_id

        self.sequences = []

        for qa in qa_pairs:
            text = f"Q: {qa['question']} A: {qa['answer']}"
            tokens = tokenizer.encode(text)

            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif len(tokens) < max_length:
                tokens = tokens + [self.pad_token] * (max_length - len(tokens))

            self.sequences.append(np.array(tokens, dtype=np.int64))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return sequence[:-1], sequence[1:]


def train_step_lora(
    model: GPTModel,
    optimizer: AdamW,
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    ignore_index: int = 0,
) -> float:
    """
    Perform a single LoRA fine-tuning step.

    Key difference from full fine-tuning:
    - Only update LoRA parameters (A and B matrices)
    - Base model weights remain frozen
    """
    # Forward pass (LoRA layers automatically add their contribution)
    logits = model.forward(inputs)

    # Compute loss
    loss = cross_entropy_loss(logits, targets, ignore_index=ignore_index)

    # Backward pass
    grad_logits = cross_entropy_loss_backward(
        logits, targets, ignore_index=ignore_index
    )
    _ = model.backward(grad_logits)  # This computes gradients for LoRA layers

    # Get ONLY LoRA gradients
    lora_gradients = get_lora_gradients(model)

    # Clip gradients
    lora_gradients = clip_gradient_norm(lora_gradients, max_grad_norm)

    # Update ONLY LoRA parameters
    optimizer.step(lora_gradients, learning_rate=learning_rate)

    return loss


def save_lora_checkpoint(
    model: GPTModel, filepath: str, step: int = 0, config: dict = None
) -> None:
    """
    Save only LoRA parameters to a checkpoint.

    This creates a small checkpoint file containing just the LoRA
    adapter weights, not the full model.
    """
    lora_params = get_lora_parameters(model)

    save_dict = {}
    for name, param in lora_params.items():
        save_dict[f"lora_{name}"] = param

    save_dict["step"] = np.array([step])

    if config is not None:
        save_dict["config"] = np.array([config])

    np.savez(filepath, **save_dict)


def load_lora_checkpoint(model: GPTModel, filepath: str) -> int:
    """
    Load LoRA parameters from a checkpoint.
    """
    data = np.load(filepath, allow_pickle=True)

    lora_params = {}
    for key in data.files:
        if key.startswith("lora_"):
            param_name = key[5:]  # Remove 'lora_' prefix
            lora_params[param_name] = data[key]

    set_lora_parameters(model, lora_params)

    step = int(data["step"][0]) if "step" in data.files else 0
    return step


def main():
    """Main LoRA fine-tuning function."""
    print("=" * 60)
    print("GPT LoRA Fine-tuning (Parameter-Efficient)")
    print("=" * 60)
    print()

    # ==================== Configuration ====================
    # LoRA configuration
    lora_rank = 4  # Rank of LoRA matrices (lower = fewer params)
    lora_alpha = 8  # Scaling factor (usually 2x rank)
    target_modules = ["query", "value"]  # Which modules to add LoRA to

    # Training hyperparameters
    batch_size = 16
    num_epochs = 10  # Can train longer since fewer params
    learning_rate = 3e-4  # Can use higher LR with fewer params
    warmup_steps = 20
    max_grad_norm = 1.0
    max_length = 64

    # Paths
    checkpoint_dir = "checkpoints"
    pretrained_path = os.path.join(checkpoint_dir, "model_best.npz")
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

    # Check prerequisites
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

    data_path = "data/shakespeare.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    qa_pairs = create_qa_pairs_from_shakespeare(text, num_pairs=200)

    split_idx = int(len(qa_pairs) * 0.9)
    train_pairs = qa_pairs[:split_idx]
    val_pairs = qa_pairs[split_idx:]

    print(f"Train Q&A pairs: {len(train_pairs)}")
    print(f"Val Q&A pairs: {len(val_pairs)}")

    train_dataset = QADataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = QADataset(val_pairs, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print()

    # ==================== Load Pre-trained Model ====================
    print("Loading pre-trained model...")

    model_config = GPTConfig(
        vocab_size=tokenizer.vocabulary_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_hidden_dim=512,
        max_sequence_length=128,
    )
    model = GPTModel(model_config)

    step, _ = load_checkpoint(model, pretrained_path)
    print(f"Loaded checkpoint from step {step}")

    total_params = count_total_parameters(model)
    print(f"Total model parameters: {total_params:,}")
    print()

    # ==================== Apply LoRA ====================
    print("Applying LoRA adapters...")
    print(f"  Rank: {lora_rank}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Target modules: {target_modules}")

    model = apply_lora_to_model(
        model, rank=lora_rank, alpha=lora_alpha, target_modules=target_modules
    )

    lora_params = count_lora_parameters(model)
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Parameter reduction: {100 * (1 - lora_params / total_params):.1f}%")
    print()

    # ==================== Optimizer (for LoRA params only) ====================
    print("Initializing optimizer for LoRA parameters...")

    optimizer = AdamW(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,  # Often no weight decay for LoRA
    )

    # Initialize optimizer with LoRA parameters
    lora_param_dict = get_lora_parameters(model)
    optimizer.initialize(lora_param_dict)

    total_steps = num_epochs * len(train_loader)
    print(f"Total training steps: {total_steps}")
    print()

    # ==================== LoRA Fine-tuning Loop ====================
    print("Starting LoRA fine-tuning...")
    print("-" * 60)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            lr = get_learning_rate_with_warmup(
                global_step,
                learning_rate,
                warmup_steps,
                total_steps,
                min_learning_rate=learning_rate / 10,
            )

            loss = train_step_lora(
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

        # Save best LoRA checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, "lora_adapter.npz")
            save_lora_checkpoint(
                model,
                save_path,
                step=global_step,
                config={
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                    "target_modules": target_modules,
                },
            )
            print(f"  -> New best! Saved LoRA adapter to {save_path}")

    print("-" * 60)
    print()

    # ==================== Test Generation ====================
    print("Testing LoRA fine-tuned model:")
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

    # ==================== Summary ====================
    print("=" * 60)
    print("LoRA Fine-tuning Summary:")
    print(f"  Total model parameters: {total_params:,}")
    print(f"  LoRA trainable parameters: {lora_params:,}")
    print(f"  Parameter reduction: {100 * (1 - lora_params / total_params):.1f}%")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print()
    print("LoRA adapter saved to: checkpoints/lora_adapter.npz")
    print("This small file can be loaded with any base model checkpoint!")
    print("=" * 60)


if __name__ == "__main__":
    main()
