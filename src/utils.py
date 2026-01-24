"""
Utility Functions for Training and Inference

This module provides utility functions for:
- Data loading and batching
- Model checkpointing (save/load)
- Training helpers

Classes:
    TextDataset: Dataset for text data with tokenization
    DataLoader: Batched data loading with shuffling

Functions:
    create_batches: Split data into batches
    get_batch: Get a training batch from token sequence
    save_checkpoint: Save model and training state
    load_checkpoint: Load model and training state
"""

from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.model import GPTModel
from src.tokenizer import BPETokenizer


class TextDataset:
    """
    Dataset for language modeling.

    Prepares text data for training by tokenizing and creating input-target pairs.
    For language modeling, the target is the input shifted by one position:
        input:  [token_0, token_1, ..., token_{n-1}]
        target: [token_1, token_2, ..., token_n]

    Attributes:
        token_ids: Full tokenized sequence
        sequence_length: Length of each training sequence
        stride: Step size between sequences (for overlap)
    """

    def __init__(
        self,
        text: str,
        tokenizer: BPETokenizer,
        sequence_length: int = 128,
        stride: Optional[int] = None,
    ):
        """
        Initialize dataset from text.

        Args:
            text: Raw text to tokenize
            tokenizer: BPE tokenizer for encoding text
            sequence_length: Length of each training sequence
            stride: Step between sequences. If None, equals sequence_length (no overlap)
        """
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length

        # Tokenize the entire text
        self.token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

        # Calculate number of sequences
        if len(self.token_ids) <= sequence_length:
            self._num_sequences = 0
        else:
            # Need at least sequence_length + 1 tokens for input-target pairs
            self._num_sequences = max(
                0, (len(self.token_ids) - sequence_length - 1) // self.stride + 1
            )

    def __len__(self) -> int:
        """Return number of sequences in the dataset."""
        return self._num_sequences

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single training example.

        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (input_ids, target_ids), each of shape (sequence_length,)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        start = idx * self.stride
        end = start + self.sequence_length

        # Input: tokens at positions [start, end)
        input_ids = self.token_ids[start:end]

        # Target: tokens at positions [start+1, end+1)
        target_ids = self.token_ids[start + 1 : end + 1]

        return input_ids, target_ids


class DataLoader:
    """
    DataLoader for batched training.

    Iterates over a dataset in batches, with optional shuffling.

    Usage:
        dataset = TextDataset(text, tokenizer)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            for inputs, targets in loader:
                # Training step
                pass
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize DataLoader.

        Args:
            dataset: Dataset to load from (must support __len__ and __getitem__)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices each epoch
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._indices = np.arange(len(dataset))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Create an iterator over batches."""
        indices = self._indices.copy()

        if self.shuffle:
            np.random.shuffle(indices)

        # Generate batches
        for start in range(0, len(indices), self.batch_size):
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]

            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Gather batch data
            batch_inputs = []
            batch_targets = []

            for idx in batch_indices:
                input_ids, target_ids = self.dataset[idx]
                batch_inputs.append(input_ids)
                batch_targets.append(target_ids)

            yield np.stack(batch_inputs), np.stack(batch_targets)

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_batches(
    data: np.ndarray, batch_size: int, drop_last: bool = False
) -> List[np.ndarray]:
    """
    Split data into batches.

    Args:
        data: Array to split, shape (num_samples, ...)
        batch_size: Size of each batch
        drop_last: Whether to drop incomplete final batch

    Returns:
        List of batch arrays
    """
    num_samples = data.shape[0]
    batches = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        if drop_last and (end - start) < batch_size:
            continue

        batches.append(data[start:end])

    return batches


def get_batch(
    token_ids: np.ndarray,
    batch_size: int,
    sequence_length: int,
    random_offset: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a training batch from a token sequence.

    This function creates a batch of input-target pairs for language modeling
    by randomly sampling starting positions from the token sequence.

    Args:
        token_ids: 1D array of token IDs
        batch_size: Number of sequences in the batch
        sequence_length: Length of each sequence
        random_offset: Whether to use random starting positions

    Returns:
        Tuple of (inputs, targets), each of shape (batch_size, sequence_length)
    """
    # Make sure we have enough tokens
    max_start = len(token_ids) - sequence_length - 1

    if max_start <= 0:
        raise ValueError(
            f"Token sequence too short. Need at least {sequence_length + 1} tokens, "
            f"got {len(token_ids)}"
        )

    if random_offset:
        # Random starting positions
        start_positions = np.random.randint(0, max_start, size=batch_size)
    else:
        # Sequential starting positions
        start_positions = np.arange(batch_size) * (max_start // batch_size)

    # Gather sequences
    inputs = np.zeros((batch_size, sequence_length), dtype=np.int64)
    targets = np.zeros((batch_size, sequence_length), dtype=np.int64)

    for i, start in enumerate(start_positions):
        inputs[i] = token_ids[start : start + sequence_length]
        targets[i] = token_ids[start + 1 : start + sequence_length + 1]

    return inputs, targets


def save_checkpoint(
    model: GPTModel,
    filepath: str,
    step: int = 0,
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint.

    Saves model parameters, configuration, training step, and optionally
    optimizer state to a .npz file.

    Args:
        model: GPT model to save
        filepath: Path to save checkpoint (should end in .npz)
        step: Current training step
        optimizer_state: Optional optimizer state dictionary
        extra_data: Optional additional data to save
    """
    # Collect all data to save
    save_dict = {}

    # Save model parameters
    for name, param in model.get_parameters().items():
        save_dict[f"param_{name}"] = param

    # Save config as dict
    save_dict["config"] = np.array([asdict(model.config)])  # Wrap in array for npz

    # Save training step
    save_dict["step"] = np.array([step])

    # Save optimizer state if provided
    if optimizer_state is not None:
        save_dict["optimizer_state"] = np.array([optimizer_state])

    # Save extra data if provided
    if extra_data is not None:
        for key, value in extra_data.items():
            save_dict[f"extra_{key}"] = np.array([value])

    # Save to npz file
    np.savez(filepath, **save_dict)


def load_checkpoint(
    model: GPTModel, filepath: str
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Load a model checkpoint.

    Loads model parameters from a checkpoint file and restores them
    to the given model.

    Args:
        model: GPT model to load parameters into
        filepath: Path to checkpoint file

    Returns:
        Tuple of (step, optimizer_state)
    """
    # Load checkpoint
    data = np.load(filepath, allow_pickle=True)

    # Load model parameters
    params = {}
    for key in data.files:
        if key.startswith("param_"):
            param_name = key[6:]  # Remove 'param_' prefix
            params[param_name] = data[key]

    # Set parameters on model
    model.set_parameters(params)

    # Get training step
    step = int(data["step"][0]) if "step" in data.files else 0

    # Get optimizer state
    optimizer_state = None
    if "optimizer_state" in data.files:
        optimizer_state = data["optimizer_state"][0]

    return step, optimizer_state


def download_shakespeare(data_dir: str = "data") -> str:
    """
    Download the Shakespeare dataset.

    Downloads the complete works of Shakespeare from a public URL
    and saves it to the specified directory.

    Args:
        data_dir: Directory to save the data

    Returns:
        Path to the downloaded file
    """
    import os
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, "shakespeare.txt")

    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
    else:
        print(f"Shakespeare dataset already exists at {filepath}")

    return filepath


def create_qa_pairs_from_shakespeare(
    text: str, num_pairs: int = 100
) -> List[Dict[str, str]]:
    """
    Create Q&A pairs from Shakespeare text.

    Creates simple Q&A pairs by selecting passages and forming
    completion tasks.

    Args:
        text: Shakespeare text
        num_pairs: Number of Q&A pairs to create

    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    # Split into lines
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20]

    qa_pairs = []

    for i in range(min(num_pairs, len(lines) - 1)):
        line = lines[i]

        # Create different types of Q&A
        if i % 3 == 0:
            # Completion task: "Continue this passage: ..."
            words = line.split()
            if len(words) > 6:
                prompt = " ".join(words[: len(words) // 2])
                completion = " ".join(words[len(words) // 2 :])
                qa_pairs.append(
                    {
                        "question": f"Continue this passage: {prompt}",
                        "answer": completion,
                    }
                )
        elif i % 3 == 1:
            # Style task: "Write in Shakespeare's style about..."
            if "love" in line.lower():
                qa_pairs.append(
                    {
                        "question": "Write about love in Shakespeare's style:",
                        "answer": line,
                    }
                )
            elif "death" in line.lower():
                qa_pairs.append(
                    {
                        "question": "Write about mortality in Shakespeare's style:",
                        "answer": line,
                    }
                )
            else:
                qa_pairs.append(
                    {"question": "Write a line in Shakespeare's style:", "answer": line}
                )
        else:
            # Direct line learning
            qa_pairs.append({"question": "Recite a Shakespeare line:", "answer": line})

    return qa_pairs[:num_pairs]


# =============================================================================
# EDUCATIONAL DEMO
# Run with: python -m src.utils
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("UTILITIES DEMO - Data Loading and Checkpointing")
    print("=" * 70)
    print()
    print("This module provides the 'plumbing' for training LLMs:")
    print("  - TextDataset: Prepares text as input-target pairs")
    print("  - DataLoader: Batches data for efficient training")
    print("  - Checkpointing: Save/load model state")
    print()
    print("Dependencies:")
    print("  - src.tokenizer (for encoding text)")
    print("  - src.model (for model serialization)")
    print()

    import os
    import tempfile

    from src.model import GPTConfig, GPTModel
    from src.tokenizer import BPETokenizer

    # -------------------------------------------------------------------------
    # TEXT DATASET
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("1. TEXT DATASET - Preparing Data for Language Modeling")
    print("-" * 70)
    print()
    print("Language models learn to predict the next token. We create")
    print("input-target pairs where the target is the input shifted by 1:")
    print()
    print('  Text:    "The cat sat on the mat"')
    print("  Tokens:  [The] [cat] [sat] [on] [the] [mat]")
    print("  Input:   [The] [cat] [sat] [on] [the]")
    print("  Target:  [cat] [sat] [on] [the] [mat]")
    print()
    print("The model learns: given [The], predict [cat]; given [cat], predict [sat]...")
    print()

    # Create a simple tokenizer trained on some text
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die, to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to, 'tis a consummation
    Devoutly to be wished. To die, to sleep;
    To sleep, perchance to dream: ay, there's the rub.
    """

    print("Training tokenizer on sample text...")
    tokenizer = BPETokenizer()
    tokenizer.train(
        sample_text * 10, vocabulary_size=200
    )  # Repeat for more training data
    print(f"Vocabulary size: {tokenizer.vocabulary_size}")
    print()

    # Create dataset
    sequence_length = 16
    dataset = TextDataset(
        text=sample_text,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=8,  # Overlapping sequences
    )

    print("Dataset created:")
    print(f"  Total tokens: {len(dataset.token_ids)}")
    print(f"  Sequence length: {dataset.sequence_length}")
    print(f"  Stride (overlap): {dataset.stride}")
    print(f"  Number of sequences: {len(dataset)}")
    print()

    # Show an example
    if len(dataset) > 0:
        input_ids, target_ids = dataset[0]
        print("Example sequence (index 0):")
        print(f"  Input IDs:  {input_ids[:10]}... (len={len(input_ids)})")
        print(f"  Target IDs: {target_ids[:10]}... (len={len(target_ids)})")
        print()
        print("Notice: target[i] = input[i+1] (shifted by one position)")
        print(f"  input[0]={input_ids[0]} -> target[0]={target_ids[0]}")
        print(f"  input[1]={input_ids[1]} -> target[1]={target_ids[1]}")
        print()

    # -------------------------------------------------------------------------
    # DATA LOADER
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("2. DATA LOADER - Batching for Efficient Training")
    print("-" * 70)
    print()
    print("Training works best with batches of data:")
    print("  - GPU parallelism: Process multiple sequences at once")
    print("  - Gradient stability: Average over multiple examples")
    print("  - Shuffling: Different order each epoch prevents overfitting")
    print()

    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print("DataLoader created:")
    print(f"  Batch size: {batch_size}")
    print("  Shuffle: True (different order each epoch)")
    print("  Drop last: False (include partial final batch)")
    print(f"  Number of batches: {len(loader)}")
    print()

    # Show one batch
    print("Iterating through one epoch:")
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"  Batch {batch_idx}: inputs={inputs.shape}, targets={targets.shape}")
        if batch_idx >= 2:
            print("  ...")
            break
    print()

    # -------------------------------------------------------------------------
    # GET_BATCH FUNCTION
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("3. GET_BATCH - Quick Batch Sampling")
    print("-" * 70)
    print()
    print("For simpler training loops, get_batch() samples random sequences:")
    print()

    token_ids = np.array(tokenizer.encode(sample_text), dtype=np.int64)

    inputs, targets = get_batch(
        token_ids=token_ids, batch_size=8, sequence_length=16, random_offset=True
    )

    print("get_batch() result:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    print()
    print("Each row is a randomly sampled sequence from the text.")
    print("Good for simple training; DataLoader better for full coverage.")
    print()

    # -------------------------------------------------------------------------
    # CHECKPOINTING
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("4. CHECKPOINTING - Saving and Loading Models")
    print("-" * 70)
    print()
    print("Training large models takes time. Checkpoints let you:")
    print("  - Resume training after interruption")
    print("  - Save best model during training")
    print("  - Share trained models with others")
    print()

    # Create a small model
    config = GPTConfig(
        vocab_size=tokenizer.vocabulary_size,
        embedding_dim=64,
        num_heads=2,
        num_layers=2,
        ffn_hidden_dim=128,
        max_sequence_length=32,
    )
    model = GPTModel(config)

    print(
        f"Created model with {sum(p.size for p in model.get_parameters().values()):,} parameters"
    )
    print()

    # Get initial weights for comparison
    initial_params = {k: v.copy() for k, v in model.get_parameters().items()}

    # Modify some weights (simulating training)
    params = model.get_parameters()
    for name, param in params.items():
        params[name] = param + np.random.randn(*param.shape) * 0.01
    model.set_parameters(params)

    print("Modified weights (simulating training)...")
    print()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        checkpoint_path = f.name

    save_checkpoint(
        model=model,
        filepath=checkpoint_path,
        step=1000,
        optimizer_state={"learning_rate": 0.001, "beta1": 0.9},
        extra_data={"loss": 2.5, "epoch": 5},
    )

    file_size = os.path.getsize(checkpoint_path)
    print("Saved checkpoint:")
    print(f"  Path: {checkpoint_path}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    print()

    # Create fresh model and load checkpoint
    model2 = GPTModel(config)

    # Verify weights are different before loading
    params_before = model2.get_parameters()
    diff_before = np.abs(
        params_before["token_embedding.weight"] - params["token_embedding.weight"]
    ).mean()

    step, opt_state = load_checkpoint(model2, checkpoint_path)

    # Verify weights match after loading
    params_after = model2.get_parameters()
    diff_after = np.abs(
        params_after["token_embedding.weight"] - params["token_embedding.weight"]
    ).mean()

    print("Loaded checkpoint:")
    print(f"  Step: {step}")
    print(f"  Optimizer state: {opt_state}")
    print()
    print("Weight verification:")
    print(f"  Difference before load: {diff_before:.6f}")
    print(f"  Difference after load:  {diff_after:.6f}")
    print(f"  Weights restored correctly: {diff_after < 1e-10}")
    print()

    # Cleanup
    os.unlink(checkpoint_path)

    # -------------------------------------------------------------------------
    # SHAKESPEARE DOWNLOAD
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("5. DATA DOWNLOAD - Getting Training Data")
    print("-" * 70)
    print()
    print("The download_shakespeare() function fetches the Tiny Shakespeare")
    print("dataset (~1MB of text) commonly used for LLM experiments:")
    print()
    print("  filepath = download_shakespeare('data')")
    print("  # Downloads to data/shakespeare.txt")
    print()
    print("This provides enough text to train a small model and see")
    print("it generate Shakespeare-like output!")
    print()

    # -------------------------------------------------------------------------
    # TRAINING LOOP PATTERN
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("6. TRAINING LOOP PATTERN")
    print("-" * 70)
    print()
    print("Here's how these utilities fit together in training:")
    print()
    print("```python")
    print("# Setup")
    print("tokenizer = BPETokenizer()")
    print("tokenizer.load('tokenizer.json')")
    print("dataset = TextDataset(text, tokenizer, sequence_length=128)")
    print("loader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print()
    print("model = GPTModel(config)")
    print("optimizer = AdamW(learning_rate=3e-4)")
    print()
    print("# Training loop")
    print("for epoch in range(num_epochs):")
    print("    for batch_idx, (inputs, targets) in enumerate(loader):")
    print("        # Forward pass")
    print("        logits = model.forward(inputs)")
    print("        loss = cross_entropy(logits, targets)")
    print()
    print("        # Backward pass")
    print("        model.zero_grad()")
    print("        model.backward(targets)")
    print()
    print("        # Optimizer step")
    print("        optimizer.step(model.get_parameters(), model.get_gradients())")
    print()
    print("        # Checkpoint periodically")
    print("        if batch_idx % 1000 == 0:")
    print("            save_checkpoint(model, f'checkpoint_{batch_idx}.npz')")
    print("```")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("- TextDataset: Converts text to (input, target) pairs for LM training")
    print("- DataLoader: Batches data with shuffling for efficient training")
    print("- get_batch(): Quick random batch sampling")
    print("- save/load_checkpoint(): Persist and restore model state")
    print("- download_shakespeare(): Get training data easily")
    print()
    print("These utilities handle the 'plumbing' so you can focus on the model!")
    print()
    print("CONGRATULATIONS! You've explored all the core modules.")
    print("Next: Try running the training scripts:")
    print("  - python train_pretrain.py (train from scratch)")
    print("  - python train_finetune_lora.py (efficient fine-tuning)")
    print("  - python run_demo.py (interactive generation)")
