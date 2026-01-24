"""
Byte Pair Encoding (BPE) Tokenizer

This module implements a BPE tokenizer from scratch, which is the tokenization
algorithm used in GPT-2, GPT-3, GPT-4, and many other modern LLMs.

BPE works by:
1. Starting with individual characters as the initial vocabulary
2. Iteratively merging the most frequent pair of adjacent tokens
3. Continuing until the desired vocabulary size is reached

This results in a vocabulary that can represent common words as single tokens
while still being able to handle rare words by breaking them into subwords.

Reference:
    "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
    https://arxiv.org/abs/1508.07909

Classes:
    BPETokenizer: Main tokenizer class with train, encode, decode methods
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer.

    This tokenizer learns a vocabulary from a text corpus using the BPE algorithm,
    then can encode text into token IDs and decode token IDs back to text.

    Attributes:
        vocabulary: Dict mapping token IDs to token strings
        token_to_id: Dict mapping token strings to token IDs
        merges: List of merge rules (pairs of tokens to merge)
        vocabulary_size: Total number of tokens in vocabulary

    Special Tokens:
        <PAD>: Padding token for batch processing
        <UNK>: Unknown token for out-of-vocabulary characters
        <BOS>: Beginning of sequence token
        <EOS>: End of sequence token

    Example:
        >>> tokenizer = BPETokenizer()
        >>> tokenizer.train("hello world hello", vocabulary_size=50)
        >>> token_ids = tokenizer.encode("hello")
        >>> text = tokenizer.decode(token_ids)
    """

    # Special token strings
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        """Initialize empty tokenizer."""
        # Vocabulary: ID -> token string
        self.vocabulary: Dict[int, str] = {}
        # Inverse vocabulary: token string -> ID
        self.token_to_id: Dict[str, int] = {}
        # Merge rules: list of (token1, token2) pairs in order they were learned
        self.merges: List[Tuple[str, str]] = []

        # Special token IDs (set during training)
        self.pad_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

    def train(self, text: str, vocabulary_size: int, min_frequency: int = 2) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        Algorithm:
        1. Initialize vocabulary with all unique characters
        2. Add special tokens
        3. Repeat until vocabulary_size is reached:
           a. Count all adjacent token pairs in the corpus
           b. Find the most frequent pair
           c. Merge that pair into a new token
           d. Update the corpus with the merged token

        Args:
            text: Training text corpus
            vocabulary_size: Target vocabulary size (including special tokens)
            min_frequency: Minimum frequency for a pair to be merged

        Note:
            The actual vocabulary size may be less than the target if there
            are no more pairs to merge.
        """
        # Step 1: Initialize vocabulary with special tokens
        self._initialize_special_tokens()
        next_token_id = len(self.vocabulary)

        # Step 2: Split text into characters and add to vocabulary
        # We represent words with a special end-of-word marker for better tokenization
        words = self._preprocess_text(text)

        # Add all unique characters to vocabulary
        all_chars = set()
        for word in words:
            all_chars.update(word)

        for char in sorted(all_chars):
            if char not in self.token_to_id:
                self.vocabulary[next_token_id] = char
                self.token_to_id[char] = next_token_id
                next_token_id += 1

        # Step 3: Convert words to lists of token IDs for merging
        # Each word is a list of tokens that we'll merge
        word_tokens: List[List[str]] = [list(word) for word in words]

        # Step 4: Iteratively merge most frequent pairs
        while len(self.vocabulary) < vocabulary_size:
            # Count pair frequencies
            pair_frequencies = self._count_pair_frequencies(word_tokens)

            if not pair_frequencies:
                break  # No more pairs to merge

            # Find most frequent pair
            most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
            frequency = pair_frequencies[most_frequent_pair]

            if frequency < min_frequency:
                break  # No pairs meet minimum frequency

            # Create new merged token
            token1, token2 = most_frequent_pair
            new_token = token1 + token2

            # Add to vocabulary
            self.vocabulary[next_token_id] = new_token
            self.token_to_id[new_token] = next_token_id
            next_token_id += 1

            # Record merge rule
            self.merges.append(most_frequent_pair)

            # Apply merge to all words
            word_tokens = self._apply_merge(word_tokens, most_frequent_pair, new_token)

    def _initialize_special_tokens(self) -> None:
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        for i, token in enumerate(special_tokens):
            self.vocabulary[i] = token
            self.token_to_id[token] = i

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text into words for BPE training.

        We preserve spaces by treating them as separate tokens.
        This is a simplified version - production tokenizers use more
        sophisticated preprocessing.

        Args:
            text: Raw input text

        Returns:
            List of preprocessed words (including space as separate word)
        """
        # Split but preserve spaces as tokens
        words = []
        current_word = ""

        for char in text:
            if char == " ":
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(" ")  # Space as its own token
            else:
                current_word += char

        if current_word:
            words.append(current_word)

        return words

    def _count_pair_frequencies(
        self, word_tokens: List[List[str]]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count frequencies of all adjacent token pairs.

        Args:
            word_tokens: List of words, each word is a list of tokens

        Returns:
            Dictionary mapping (token1, token2) pairs to their frequencies
        """
        pair_frequencies: Dict[Tuple[str, str], int] = defaultdict(int)

        for word in word_tokens:
            if len(word) < 2:
                continue

            # Count all adjacent pairs in this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_frequencies[pair] += 1

        return pair_frequencies

    def _apply_merge(
        self, word_tokens: List[List[str]], pair: Tuple[str, str], new_token: str
    ) -> List[List[str]]:
        """
        Apply a merge rule to all words.

        Replaces all occurrences of the pair with the new merged token.

        Args:
            word_tokens: List of words (each word is a list of tokens)
            pair: The (token1, token2) pair to merge
            new_token: The new token to replace the pair with

        Returns:
            Updated word_tokens with merges applied
        """
        token1, token2 = pair
        new_word_tokens = []

        for word in word_tokens:
            new_word = []
            i = 0

            while i < len(word):
                # Check if we have the pair at current position
                if i < len(word) - 1 and word[i] == token1 and word[i + 1] == token2:
                    new_word.append(new_token)
                    i += 2  # Skip both tokens
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_tokens.append(new_word)

        return new_word_tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text into a list of token IDs.

        Process:
        1. Split text into words (preserving spaces)
        2. For each word, start with characters
        3. Apply merge rules in order they were learned
        4. Convert final tokens to IDs

        Args:
            text: Input text to encode
            add_special_tokens: If True, add BOS at start and EOS at end

        Returns:
            List of integer token IDs
        """
        if not text:
            return []

        token_ids = []

        # Process text preserving spaces
        words = self._preprocess_text(text)

        for word in words:
            # Start with characters
            tokens = list(word)

            # Apply merge rules in order
            for merge_pair in self.merges:
                tokens = self._apply_single_merge(tokens, merge_pair)

            # Convert to IDs
            for token in tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # Unknown token - try character by character
                    for char in token:
                        if char in self.token_to_id:
                            token_ids.append(self.token_to_id[char])
                        else:
                            token_ids.append(self.unk_token_id)

        # Add special tokens if requested
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def _apply_single_merge(
        self, tokens: List[str], merge_pair: Tuple[str, str]
    ) -> List[str]:
        """Apply a single merge rule to a list of tokens."""
        token1, token2 = merge_pair
        new_token = token1 + token2

        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == token1 and tokens[i + 1] == token2:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            token_ids: List of integer token IDs
            skip_special_tokens: If True, omit special tokens from output

        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""

        special_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            if token_id in self.vocabulary:
                tokens.append(self.vocabulary[token_id])
            else:
                tokens.append(self.UNK_TOKEN if not skip_special_tokens else "")

        # Join tokens - we need to be smart about spacing
        # For simplicity, we just concatenate and handle word boundaries
        text = "".join(tokens)

        return text

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        padding: bool = False,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Encode multiple texts at once.

        Args:
            texts: List of texts to encode
            add_special_tokens: Add BOS/EOS to each sequence
            padding: Pad all sequences to the same length
            max_length: Maximum length (truncate if longer, optional)

        Returns:
            List of token ID lists
        """
        encoded = [self.encode(text, add_special_tokens) for text in texts]

        if max_length:
            encoded = [ids[:max_length] for ids in encoded]

        if padding:
            max_len = max(len(ids) for ids in encoded) if encoded else 0
            encoded = [
                ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded
            ]

        return encoded

    def batch_decode(
        self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode multiple sequences at once.

        Args:
            batch_token_ids: List of token ID lists
            skip_special_tokens: Omit special tokens from output

        Returns:
            List of decoded text strings
        """
        return [self.decode(ids, skip_special_tokens) for ids in batch_token_ids]

    def save(self, path: str) -> None:
        """
        Save tokenizer to a JSON file.

        Args:
            path: File path to save to
        """
        data = {
            "vocabulary": {str(k): v for k, v in self.vocabulary.items()},
            "merges": self.merges,
            "special_tokens": {
                "pad": self.pad_token_id,
                "unk": self.unk_token_id,
                "bos": self.bos_token_id,
                "eos": self.eos_token_id,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load tokenizer from a JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded BPETokenizer instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()

        # Restore vocabulary
        tokenizer.vocabulary = {int(k): v for k, v in data["vocabulary"].items()}
        tokenizer.token_to_id = {v: k for k, v in tokenizer.vocabulary.items()}

        # Restore merges
        tokenizer.merges = [tuple(m) for m in data["merges"]]

        # Restore special token IDs
        special = data["special_tokens"]
        tokenizer.pad_token_id = special["pad"]
        tokenizer.unk_token_id = special["unk"]
        tokenizer.bos_token_id = special["bos"]
        tokenizer.eos_token_id = special["eos"]

        return tokenizer


# =============================================================================
# EDUCATIONAL DEMO
# Run with: python -m src.tokenizer
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOKENIZER DEMO - Converting Text to Numbers")
    print("=" * 70)
    print()
    print("Before a neural network can process text, we must convert it to numbers.")
    print("This module implements Byte Pair Encoding (BPE), the tokenization method")
    print("used by GPT, LLaMA, and most modern language models.")
    print()
    print("Dependencies: None (this is a foundational module)")
    print()

    import numpy as np  # Only for demo array conversion

    # -------------------------------------------------------------------------
    # WHY TOKENIZATION?
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("1. WHY TOKENIZATION? The problem we're solving")
    print("-" * 70)
    print()
    print("Neural networks need numbers, not text. We could:")
    print("  - Use characters: 'hello' = [h, e, l, l, o] = 5 tokens")
    print("  - Use whole words: 'hello' = [hello] = 1 token")
    print()
    print("Problem with characters: Too many tokens, hard to learn meaning")
    print("Problem with words: Can't handle new words, huge vocabulary")
    print()
    print("Solution: BPE finds a middle ground - common subwords become tokens.")
    print("  - 'hello' might be ['hel', 'lo'] or just ['hello']")
    print("  - 'unhappiness' might be ['un', 'happy', 'ness']")
    print()

    # -------------------------------------------------------------------------
    # TRAINING THE TOKENIZER
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("2. TRAINING THE TOKENIZER - Learning the vocabulary")
    print("-" * 70)
    print()

    # Sample text to train on
    training_text = """
    To be or not to be, that is the question.
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles.
    """

    tokenizer = BPETokenizer()
    print("Training BPE tokenizer on Shakespeare excerpt...")
    print(f"Text length: {len(training_text)} characters")
    print()

    # Train with small vocabulary for demonstration
    tokenizer.train(training_text, vocabulary_size=60)

    print(f"Vocabulary size: {tokenizer.vocabulary_size}")
    print()
    print("Special tokens (always present):")
    print(f"  <PAD> = {tokenizer.pad_token_id} (for padding sequences)")
    print(f"  <UNK> = {tokenizer.unk_token_id} (for unknown tokens)")
    print(f"  <BOS> = {tokenizer.bos_token_id} (beginning of sequence)")
    print(f"  <EOS> = {tokenizer.eos_token_id} (end of sequence)")
    print()

    # Show some learned merges
    print("BPE learns to merge common character pairs:")
    for i, merge in enumerate(tokenizer.merges[:5]):
        print(f"  Merge {i + 1}: '{merge[0]}' + '{merge[1]}' -> '{merge[0]}{merge[1]}'")
    print(f"  ... ({len(tokenizer.merges)} merges total)")
    print()

    # -------------------------------------------------------------------------
    # ENCODING: Text to token IDs
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("3. ENCODING - Converting text to token IDs")
    print("-" * 70)
    print()

    test_texts = [
        "to be",
        "question",
        "the mind",
        "hello world",  # Contains 'hello' which wasn't in training text
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"'{text}'")
        print(f"  -> Token IDs: {tokens}")

        # Show what each token represents
        token_strs = [tokenizer.vocabulary.get(t, "<UNK>") for t in tokens]
        print(f"  -> Tokens: {token_strs}")
        print()

    # -------------------------------------------------------------------------
    # DECODING: Token IDs back to text
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("4. DECODING - Converting token IDs back to text")
    print("-" * 70)
    print()

    original = "To be or not to be"
    tokens = tokenizer.encode(original)
    decoded = tokenizer.decode(tokens)

    print(f"Original: '{original}'")
    print(f"Encoded:  {tokens}")
    print(f"Decoded:  '{decoded}'")
    print()
    print(
        "Roundtrip successful!"
        if original == decoded
        else "Note: Some normalization may occur"
    )
    print()

    # -------------------------------------------------------------------------
    # HANDLING UNKNOWN TEXT
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("5. HANDLING UNKNOWN TEXT")
    print("-" * 70)
    print()
    print("What happens with text the tokenizer hasn't seen?")
    print()

    unknown_text = "xyz123"  # Characters likely not in training
    tokens = tokenizer.encode(unknown_text)
    print(f"Unknown text: '{unknown_text}'")
    print(f"Token IDs: {tokens}")

    # Check for UNK tokens
    unk_count = sum(1 for t in tokens if t == tokenizer.unk_token_id)
    print(f"UNK tokens: {unk_count}")
    print()
    print("BPE handles this gracefully by falling back to character-level,")
    print("only using <UNK> for truly unknown characters.")
    print()

    # -------------------------------------------------------------------------
    # BATCH PROCESSING
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("6. BATCH PROCESSING - Multiple texts at once")
    print("-" * 70)
    print()
    print("For training, we process multiple texts in batches.")
    print("Texts have different lengths, so we pad shorter ones.")
    print()

    texts = ["to be", "or not to be", "that is the question"]
    batch = tokenizer.batch_encode(texts, padding=True, max_length=15)
    batch_array = np.array(batch)  # Convert to numpy for shape

    print("Input texts:")
    for t in texts:
        print(f"  '{t}'")
    print()
    print(f"Batch shape: {batch_array.shape} (3 texts, max tokens)")
    print()
    print("Padded batch (0 = PAD token):")
    for i, (text, row) in enumerate(zip(texts, batch)):
        print(f"  {i}: {row}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("- Tokenization converts text to numbers that neural networks can process")
    print("- BPE learns a vocabulary of subwords from training data")
    print("- encode() converts text to token IDs")
    print("- decode() converts token IDs back to text")
    print("- batch_encode() handles multiple texts with padding")
    print()
    print("The tokenizer is the FIRST step in any LLM pipeline.")
    print()
    print("Next step: Run 'python -m src.attention' to see how the model")
    print("           processes these token embeddings with attention.")
