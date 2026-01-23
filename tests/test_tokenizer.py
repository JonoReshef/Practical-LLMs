"""
Tests for BPE (Byte Pair Encoding) tokenizer module.

Tests cover:
- Training BPE on text corpus
- Encoding text to token IDs
- Decoding token IDs back to text
- Special tokens handling
- Vocabulary building

Following TDD: these tests are written BEFORE the implementation.
"""


class TestBPETokenizerTraining:
    """
    Test suite for BPE tokenizer training.

    BPE (Byte Pair Encoding) is a subword tokenization algorithm that:
    1. Starts with individual characters as tokens
    2. Iteratively merges the most frequent pair of tokens
    3. Continues until desired vocabulary size is reached

    Reference: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
    """

    def test_tokenizer_train_creates_vocabulary(self):
        """Training should create a vocabulary of the specified size."""
        from src.tokenizer import BPETokenizer

        training_text = "hello hello hello world world"
        target_vocabulary_size = 20

        tokenizer = BPETokenizer()
        tokenizer.train(training_text, vocabulary_size=target_vocabulary_size)

        assert len(tokenizer.vocabulary) <= target_vocabulary_size, (
            f"Vocabulary size should be <= {target_vocabulary_size}"
        )
        assert len(tokenizer.vocabulary) > 0, "Vocabulary should not be empty"

    def test_tokenizer_train_includes_characters(self):
        """Vocabulary should include all unique characters from training text."""
        from src.tokenizer import BPETokenizer

        training_text = "abcdef"

        tokenizer = BPETokenizer()
        tokenizer.train(training_text, vocabulary_size=50)

        # All characters should be in vocabulary
        for char in "abcdef":
            assert char in tokenizer.vocabulary.values(), (
                f"Character '{char}' should be in vocabulary"
            )

    def test_tokenizer_train_creates_merges(self):
        """Training should create merge rules for frequent pairs."""
        from src.tokenizer import BPETokenizer

        # Repeated pattern should create merges
        training_text = "ab ab ab ab ab"

        tokenizer = BPETokenizer()
        tokenizer.train(training_text, vocabulary_size=50)

        # Should have some merge rules
        assert len(tokenizer.merges) > 0, (
            "Should have merge rules after training on repeated patterns"
        )

    def test_tokenizer_special_tokens(self):
        """Tokenizer should handle special tokens."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        # Check special tokens exist
        assert tokenizer.pad_token_id is not None, "Should have PAD token"
        assert tokenizer.unk_token_id is not None, "Should have UNK token"
        assert tokenizer.bos_token_id is not None, "Should have BOS token"
        assert tokenizer.eos_token_id is not None, "Should have EOS token"


class TestBPETokenizerEncoding:
    """
    Test suite for BPE tokenizer encoding (text -> token IDs).
    """

    def test_encode_returns_list_of_integers(self):
        """Encoding should return a list of integer token IDs."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        token_ids = tokenizer.encode("hello")

        assert isinstance(token_ids, list), "Encoded output should be a list"
        assert all(isinstance(id, int) for id in token_ids), (
            "All token IDs should be integers"
        )

    def test_encode_non_empty_output(self):
        """Encoding non-empty text should produce non-empty output."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        token_ids = tokenizer.encode("hello")

        assert len(token_ids) > 0, "Encoding should produce at least one token"

    def test_encode_empty_string(self):
        """Encoding empty string should return empty list."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        token_ids = tokenizer.encode("")

        assert len(token_ids) == 0, "Encoding empty string should return empty list"

    def test_encode_unknown_characters(self):
        """Unknown characters should be handled (mapped to UNK or character tokens)."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        # Try encoding text with characters not in training data
        token_ids = tokenizer.encode("xyz123")

        # Should not raise exception and should produce output
        assert isinstance(token_ids, list), "Should handle unknown characters"

    def test_encode_with_special_tokens(self):
        """Encoding with add_special_tokens should add BOS/EOS."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        token_ids_without = tokenizer.encode("hello", add_special_tokens=False)
        token_ids_with = tokenizer.encode("hello", add_special_tokens=True)

        # With special tokens should be longer (BOS + content + EOS)
        assert len(token_ids_with) == len(token_ids_without) + 2, (
            "Adding special tokens should add BOS and EOS"
        )
        assert token_ids_with[0] == tokenizer.bos_token_id, "First token should be BOS"
        assert token_ids_with[-1] == tokenizer.eos_token_id, "Last token should be EOS"


class TestBPETokenizerDecoding:
    """
    Test suite for BPE tokenizer decoding (token IDs -> text).
    """

    def test_decode_returns_string(self):
        """Decoding should return a string."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        token_ids = tokenizer.encode("hello")
        decoded_text = tokenizer.decode(token_ids)

        assert isinstance(decoded_text, str), "Decoded output should be a string"

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should recover original text."""
        from src.tokenizer import BPETokenizer

        original_text = "hello world"

        tokenizer = BPETokenizer()
        tokenizer.train(original_text, vocabulary_size=50)

        token_ids = tokenizer.encode(original_text, add_special_tokens=False)
        decoded_text = tokenizer.decode(token_ids)

        assert decoded_text == original_text, (
            f"Roundtrip failed: '{original_text}' -> {token_ids} -> '{decoded_text}'"
        )

    def test_decode_skip_special_tokens(self):
        """Decoding with skip_special_tokens should omit BOS/EOS."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        token_ids = tokenizer.encode("hello", add_special_tokens=True)
        decoded_with_special = tokenizer.decode(token_ids, skip_special_tokens=False)
        decoded_without_special = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Without special tokens should be shorter
        assert len(decoded_without_special) <= len(decoded_with_special), (
            "Skipping special tokens should produce shorter or equal output"
        )

    def test_decode_empty_list(self):
        """Decoding empty list should return empty string."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        decoded_text = tokenizer.decode([])

        assert decoded_text == "", "Decoding empty list should return empty string"


class TestBPETokenizerVocabulary:
    """
    Test suite for vocabulary management.
    """

    def test_vocabulary_is_dict(self):
        """Vocabulary should be a dictionary mapping IDs to tokens."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        assert isinstance(tokenizer.vocabulary, dict), (
            "Vocabulary should be a dictionary"
        )

    def test_inverse_vocabulary(self):
        """Should have inverse vocabulary mapping tokens to IDs."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        # Check that we can look up token IDs by token string
        assert hasattr(tokenizer, "token_to_id"), "Should have token_to_id mapping"

        # Verify consistency between vocab and inverse vocab
        for token_id, token in tokenizer.vocabulary.items():
            assert tokenizer.token_to_id.get(token) == token_id, (
                f"Inconsistent mapping for token '{token}'"
            )

    def test_vocabulary_size_property(self):
        """Should have a vocabulary_size property."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocabulary_size=50)

        assert tokenizer.vocabulary_size == len(tokenizer.vocabulary), (
            "vocabulary_size should match length of vocabulary"
        )


class TestBPETokenizerBatchProcessing:
    """
    Test suite for batch encoding/decoding.
    """

    def test_batch_encode(self):
        """Should be able to encode multiple texts at once."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world goodbye", vocabulary_size=50)

        texts = ["hello", "world", "hello world"]
        batch_encoded = tokenizer.batch_encode(texts)

        assert len(batch_encoded) == len(texts), (
            "Batch encoding should return same number of sequences"
        )

        # Each encoding should be a list of integers
        for encoded in batch_encoded:
            assert isinstance(encoded, list), "Each encoding should be a list"
            assert all(isinstance(id, int) for id in encoded), (
                "All token IDs should be integers"
            )

    def test_batch_encode_with_padding(self):
        """Batch encoding with padding should produce equal-length sequences."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world goodbye friend", vocabulary_size=50)

        texts = ["hi", "hello world"]  # Different lengths
        batch_encoded = tokenizer.batch_encode(texts, padding=True)

        # All sequences should have same length after padding
        lengths = [len(seq) for seq in batch_encoded]
        assert len(set(lengths)) == 1, (
            "All padded sequences should have the same length"
        )

    def test_batch_decode(self):
        """Should be able to decode multiple sequences at once."""
        from src.tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train("hello world goodbye", vocabulary_size=50)

        texts = ["hello", "world"]
        batch_encoded = tokenizer.batch_encode(texts, add_special_tokens=False)
        batch_decoded = tokenizer.batch_decode(batch_encoded)

        assert len(batch_decoded) == len(texts), (
            "Batch decoding should return same number of texts"
        )

        for original, decoded in zip(texts, batch_decoded):
            assert original == decoded, f"Roundtrip failed: '{original}' != '{decoded}'"


class TestBPETokenizerSaveLoad:
    """
    Test suite for saving and loading tokenizer.
    """

    def test_save_and_load(self, tmp_path):
        """Should be able to save and load tokenizer."""
        from src.tokenizer import BPETokenizer

        # Train tokenizer
        tokenizer = BPETokenizer()
        tokenizer.train("hello world hello", vocabulary_size=50)

        # Save
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(save_path))

        # Load into new tokenizer
        loaded_tokenizer = BPETokenizer.load(str(save_path))

        # Verify same encoding
        original_encoding = tokenizer.encode("hello world")
        loaded_encoding = loaded_tokenizer.encode("hello world")

        assert original_encoding == loaded_encoding, (
            "Loaded tokenizer should produce same encodings"
        )
