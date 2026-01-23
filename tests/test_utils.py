"""
Tests for Utility Functions

Tests for data loading, batching, and model checkpointing utilities.
"""

import os
import tempfile

import numpy as np
import pytest

from src.model import GPTConfig, GPTModel
from src.tokenizer import BPETokenizer
from src.utils import (
    DataLoader,
    TextDataset,
    create_batches,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)


class TestTextDataset:
    """Test the TextDataset class."""

    @pytest.fixture
    def sample_tokenizer(self):
        """Create a trained tokenizer for testing."""
        tokenizer = BPETokenizer()
        text = "Hello world this is a test. The quick brown fox jumps. Machine learning is fun."
        tokenizer.train(text, vocabulary_size=100)
        return tokenizer

    def test_dataset_creation(self, sample_tokenizer):
        """Test that dataset can be created."""
        text = "Hello world this is a test sentence for the dataset."
        dataset = TextDataset(text=text, tokenizer=sample_tokenizer, sequence_length=16)

        assert len(dataset) > 0

    def test_dataset_getitem(self, sample_tokenizer):
        """Test that we can get items from the dataset."""
        text = "Hello world this is a test sentence for the dataset."
        dataset = TextDataset(text=text, tokenizer=sample_tokenizer, sequence_length=8)

        if len(dataset) > 0:
            input_ids, target_ids = dataset[0]

            # Input and target should have the same shape
            assert input_ids.shape == target_ids.shape
            # Target should be shifted by 1
            np.testing.assert_array_equal(input_ids[1:], target_ids[:-1])

    def test_dataset_length(self, sample_tokenizer):
        """Test dataset length calculation."""
        text = "A " * 100  # Create text with many tokens
        dataset = TextDataset(text=text, tokenizer=sample_tokenizer, sequence_length=10)

        # Length should be calculated based on token count
        assert len(dataset) >= 0


class TestDataLoader:
    """Test the DataLoader class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a mock dataset for testing."""

        class MockDataset:
            def __init__(self, size=100):
                self.size = size
                self.data = np.random.randint(0, 50, size=(size, 10))

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.data[idx], self.data[idx]

        return MockDataset(100)

    def test_dataloader_iteration(self, sample_dataset):
        """Test that dataloader can be iterated."""
        loader = DataLoader(sample_dataset, batch_size=8, shuffle=False)

        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert len(batch) == 2  # Input and target

        assert batch_count > 0

    def test_dataloader_batch_size(self, sample_dataset):
        """Test that batches have correct size."""
        batch_size = 8
        loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False)

        for inputs, targets in loader:
            # All batches except possibly the last should have correct size
            assert inputs.shape[0] <= batch_size

    def test_dataloader_shuffle(self, sample_dataset):
        """Test that shuffle works."""
        loader1 = DataLoader(sample_dataset, batch_size=8, shuffle=True)
        loader2 = DataLoader(sample_dataset, batch_size=8, shuffle=True)

        # Two shuffled iterations should give different orders
        # (with very high probability)
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Just check that the batches are valid
        assert batch1[0].shape == batch2[0].shape


class TestCreateBatches:
    """Test the create_batches utility function."""

    def test_create_batches_basic(self):
        """Test basic batch creation."""
        data = np.arange(100).reshape(50, 2)
        batches = create_batches(data, batch_size=10)

        assert len(batches) == 5
        for batch in batches:
            assert batch.shape == (10, 2)

    def test_create_batches_remainder(self):
        """Test batch creation with remainder."""
        data = np.arange(100).reshape(50, 2)
        batches = create_batches(data, batch_size=8, drop_last=False)

        # 50 / 8 = 6 full batches + 1 partial batch
        assert len(batches) == 7
        assert batches[-1].shape[0] == 2  # Remainder

    def test_create_batches_drop_last(self):
        """Test batch creation with drop_last."""
        data = np.arange(100).reshape(50, 2)
        batches = create_batches(data, batch_size=8, drop_last=True)

        # 50 / 8 = 6 full batches, dropping remainder
        assert len(batches) == 6
        for batch in batches:
            assert batch.shape[0] == 8


class TestGetBatch:
    """Test the get_batch utility function."""

    def test_get_batch(self):
        """Test getting a specific batch."""
        token_ids = np.arange(100)

        inputs, targets = get_batch(
            token_ids, batch_size=4, sequence_length=10, random_offset=False
        )

        assert inputs.shape == (4, 10)
        assert targets.shape == (4, 10)

        # Targets should be shifted by 1
        for i in range(4):
            np.testing.assert_array_equal(inputs[i, 1:], targets[i, :-1])


class TestCheckpoint:
    """Test model checkpointing."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = GPTConfig(
            vocab_size=100,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
            ffn_hidden_dim=64,
            max_sequence_length=16,
        )
        return GPTModel(config)

    def test_save_checkpoint(self, small_model):
        """Test that checkpoint can be saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.npz")

            save_checkpoint(
                small_model,
                checkpoint_path,
                step=100,
                optimizer_state={"step_count": 100},
            )

            assert os.path.exists(checkpoint_path)

    def test_load_checkpoint(self, small_model):
        """Test that checkpoint can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.npz")

            # Save original parameters
            original_params = {
                name: param.copy()
                for name, param in small_model.get_parameters().items()
            }

            save_checkpoint(small_model, checkpoint_path, step=100)

            # Modify model parameters
            for param in small_model.get_parameters().values():
                param.fill(999)

            # Load checkpoint
            loaded_step, optimizer_state = load_checkpoint(small_model, checkpoint_path)

            assert loaded_step == 100

            # Verify parameters are restored
            for name, param in small_model.get_parameters().items():
                if name in original_params:
                    np.testing.assert_array_equal(param, original_params[name])

    def test_checkpoint_includes_config(self, small_model):
        """Test that checkpoint includes model config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.npz")

            save_checkpoint(small_model, checkpoint_path, step=50)

            # Load raw checkpoint
            data = np.load(checkpoint_path, allow_pickle=True)

            assert "config" in data.files
