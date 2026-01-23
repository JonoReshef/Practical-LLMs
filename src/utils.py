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
