# Tokenization: Converting Text to Numbers

Tokenization is the first and fundamental step in processing text for language models. Before a neural network can understand "The cat sat on the mat," we must convert these words into numerical representations.

---

## Table of Contents

1. [Why Tokenization?](#why-tokenization)
2. [Tokenization Approaches](#tokenization-approaches)
3. [Byte Pair Encoding (BPE) Algorithm](#byte-pair-encoding-bpe-algorithm)
4. [Step-by-Step Numeric Example](#step-by-step-numeric-example)
5. [Special Tokens](#special-tokens)
6. [Code Implementation](#code-implementation)
7. [Visualization](#visualization)
8. [Trade-offs and Considerations](#trade-offs-and-considerations)
9. [References](#references)

---

## Why Tokenization?

Neural networks operate on numbers, not text. We need a systematic way to:

1. **Convert text to numbers**: Map each unit of text to a unique integer ID
2. **Maintain meaning**: Similar concepts should have related representations
3. **Handle any text**: Including words never seen during training
4. **Balance efficiency**: Not too many tokens (slow) or too few (loss of meaning)

### The Granularity Problem

Consider three approaches:

| Approach            | Example: "unhappiness"                          | Vocabulary Size | Sequence Length |
| ------------------- | ----------------------------------------------- | --------------- | --------------- |
| **Character-level** | `['u','n','h','a','p','p','i','n','e','s','s']` | ~100            | Very long       |
| **Word-level**      | `['unhappiness']`                               | ~500,000+       | Short           |
| **Subword (BPE)**   | `['un', 'happiness']`                           | ~30,000-50,000  | Balanced        |

**Character-level** creates very long sequences the model must process, making it computationally expensive and harder to capture long-range dependencies.

**Word-level** requires an enormous vocabulary to cover all words, including rare ones, technical terms, and misspellings. Out-of-vocabulary words become a major problem.

**Subword tokenization (BPE)** strikes the perfect balance - common words become single tokens while rare words are broken into meaningful subunits.

---

## Tokenization Approaches

### 1. Character-Level Tokenization

```
"Hello World" → ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd']
Token IDs:      [7,   4,   11,  11,  14,  0,   22,  14,  17,  11,  3]
```

**Pros**: Small vocabulary (~100 characters), handles any text
**Cons**: Very long sequences, hard to capture word-level meaning

### 2. Word-Level Tokenization

```
"Hello World" → ['Hello', 'World']
Token IDs:      [4521,    8932]
```

**Pros**: Short sequences, preserves word meaning
**Cons**: Huge vocabulary, can't handle unknown words

### 3. Subword Tokenization (BPE, WordPiece, SentencePiece)

```
"unhappiness" → ['un', 'happiness']  or  ['un', 'happ', 'iness']
Token IDs:      [892,  3421]              [892,  1205,  2341]
```

**Pros**: Balanced vocabulary size, handles rare words, preserves morphology
**Cons**: Slightly longer sequences than word-level

---

## Byte Pair Encoding (BPE) Algorithm

BPE was originally a data compression algorithm, adapted for NLP by [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909). It's used in GPT-2, GPT-3, GPT-4, and many other LLMs.

### The Core Idea

1. Start with individual characters as the initial vocabulary
2. Count all adjacent token pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Add the new token to the vocabulary
5. Repeat steps 2-4 until reaching the desired vocabulary size

### Why It Works

BPE learns the statistical structure of the language:

- Common words (like "the", "and") become single tokens
- Common prefixes/suffixes ("un-", "-ing", "-tion") become tokens
- Rare words decompose into known subwords

---

## Step-by-Step Numeric Example

Let's train a BPE tokenizer on a small corpus:

```
Training corpus: "low lower lowest"
Target vocabulary size: 15
```

### Initial State

**Step 0: Character Vocabulary**

| Token ID | Token       |
| -------- | ----------- |
| 0        | `<PAD>`     |
| 1        | `<UNK>`     |
| 2        | `<BOS>`     |
| 3        | `<EOS>`     |
| 4        | ` ` (space) |
| 5        | `e`         |
| 6        | `l`         |
| 7        | `o`         |
| 8        | `r`         |
| 9        | `s`         |
| 10       | `t`         |
| 11       | `w`         |

**Current tokenization:**

```
"low lower lowest" → [6, 7, 11, 4, 6, 7, 11, 5, 8, 4, 6, 7, 11, 5, 9, 10]
                      l  o  w   _  l  o  w   e  r  _  l  o  w   e  s  t
```

(where `_` represents space)

### Iteration 1: Find Most Frequent Pair

Count all adjacent pairs:

| Pair    | Count |
| ------- | ----- |
| (l, o)  | 3     |
| (o, w)  | 3     |
| (w, \_) | 2     |
| (\_, l) | 2     |
| (w, e)  | 1     |
| (e, r)  | 1     |
| (r, \_) | 1     |
| (e, s)  | 1     |
| (s, t)  | 1     |

**Most frequent pairs (tied):** `(l, o)` and `(o, w)` - both appear 3 times.

We select `(l, o)` (alphabetically first in this case):

**New merge rule**: `l` + `o` → `lo` (Token ID 12)

**Updated vocabulary:**

| Token ID | Token |
| -------- | ----- |
| ...      | ...   |
| 12       | `lo`  |

**Updated tokenization:**

```
"low lower lowest" → [12, 11, 4, 12, 11, 5, 8, 4, 12, 11, 5, 9, 10]
                      lo  w   _  lo  w   e  r  _  lo  w   e  s  t
```

Sequence length: 16 → 13 (reduced by 3)

### Iteration 2

New pair counts:

| Pair     | Count |
| -------- | ----- |
| (lo, w)  | 3     |
| (w, \_)  | 2     |
| (\_, lo) | 2     |
| (w, e)   | 1     |
| (e, r)   | 1     |
| (r, \_)  | 1     |
| (e, s)   | 1     |
| (s, t)   | 1     |

**Most frequent:** `(lo, w)` appears 3 times.

**New merge rule**: `lo` + `w` → `low` (Token ID 13)

**Updated tokenization:**

```
"low lower lowest" → [13, 4, 13, 5, 8, 4, 13, 5, 9, 10]
                      low _  low e  r  _  low e  s  t
```

Sequence length: 13 → 10 (reduced by 3)

### Iteration 3

New pair counts:

| Pair      | Count |
| --------- | ----- |
| (low, \_) | 2     |
| (\_, low) | 2     |
| (low, e)  | 2     |
| (e, r)    | 1     |
| (r, \_)   | 1     |
| (e, s)    | 1     |
| (s, t)    | 1     |

**Most frequent (tied):** `(low, _)`, `(_, low)`, and `(low, e)` all appear 2 times.

Select `(low, e)`:

**New merge rule**: `low` + `e` → `lowe` (Token ID 14)

**Updated tokenization:**

```
"low lower lowest" → [13, 4, 14, 8, 4, 14, 9, 10]
                      low _  lowe r  _  lowe s  t
```

### Final Merged Rules (Vocabulary = 15)

| Merge # | Rule           | New Token |
| ------- | -------------- | --------- |
| 1       | l + o → lo     | ID 12     |
| 2       | lo + w → low   | ID 13     |
| 3       | low + e → lowe | ID 14     |

### Encoding New Text

Now let's encode text using the trained tokenizer:

**Text: "low"**

1. Split into characters: `['l', 'o', 'w']`
2. Apply merge rule 1: `['lo', 'w']`
3. Apply merge rule 2: `['low']`
4. Result: `[13]`

**Text: "flower"** (not in training data!)

1. Split into characters: `['f', 'l', 'o', 'w', 'e', 'r']`
2. 'f' not in vocabulary → `['<UNK>', 'l', 'o', 'w', 'e', 'r']`
3. Apply merge rule 1: `['<UNK>', 'lo', 'w', 'e', 'r']`
4. Apply merge rule 2: `['<UNK>', 'low', 'e', 'r']`
5. Apply merge rule 3: `['<UNK>', 'lowe', 'r']`
6. Result: `[1, 14, 8]`

---

## Special Tokens

Special tokens serve important functions in language model architectures:

| Token   | Description           | Usage                                        |
| ------- | --------------------- | -------------------------------------------- |
| `<PAD>` | Padding               | Fill shorter sequences to match batch length |
| `<UNK>` | Unknown               | Represent out-of-vocabulary characters       |
| `<BOS>` | Beginning of Sequence | Signal the start of a text                   |
| `<EOS>` | End of Sequence       | Signal the end of generated text             |

### Example with Special Tokens

```
Original:  "Hello"
With BOS:  "<BOS> Hello"
With EOS:  "<BOS> Hello <EOS>"
Token IDs: [2, 234, 567, 3]
```

For batching sequences of different lengths:

```
Sequence 1: "Hello"     → [2, 234, 567, 3, 0, 0]  (padded)
Sequence 2: "Hi there"  → [2, 89, 456, 789, 3, 0]  (padded)
Batch shape: (2, 6)
```

---

## Code Implementation

From [src/tokenizer.py](src/tokenizer.py), here's the core BPE implementation:

### Training the Tokenizer

```python
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
    """
    # Step 1: Initialize vocabulary with special tokens
    self._initialize_special_tokens()
    next_token_id = len(self.vocabulary)

    # Step 2: Split text into characters and add to vocabulary
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

    # Step 3: Convert words to lists of tokens for merging
    word_tokens: List[List[str]] = [list(word) for word in words]

    # Step 4: Iteratively merge most frequent pairs
    while len(self.vocabulary) < vocabulary_size:
        # Count pair frequencies
        pair_frequencies = self._count_pair_frequencies(word_tokens)

        if not pair_frequencies:
            break

        # Find most frequent pair
        most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
        frequency = pair_frequencies[most_frequent_pair]

        if frequency < min_frequency:
            break

        # Create new merged token
        token1, token2 = most_frequent_pair
        new_token = token1 + token2

        # Add to vocabulary and record merge rule
        self.vocabulary[next_token_id] = new_token
        self.token_to_id[new_token] = next_token_id
        next_token_id += 1
        self.merges.append(most_frequent_pair)

        # Apply merge to all words
        word_tokens = self._apply_merge(word_tokens, most_frequent_pair, new_token)
```

### Encoding Text

```python
def encode(self, text: str) -> List[int]:
    """Convert text to token IDs."""
    words = self._preprocess_text(text)
    all_token_ids = []

    for word in words:
        # Start with characters
        tokens = list(word)

        # Apply merge rules in order
        for merge_pair in self.merges:
            tokens = self._apply_merge_to_tokens(tokens, merge_pair)

        # Convert to IDs
        for token in tokens:
            if token in self.token_to_id:
                all_token_ids.append(self.token_to_id[token])
            else:
                all_token_ids.append(self.unk_token_id)

    return all_token_ids
```

### Decoding Token IDs

```python
def decode(self, token_ids: List[int]) -> str:
    """Convert token IDs back to text."""
    tokens = []
    for token_id in token_ids:
        if token_id in self.vocabulary:
            token = self.vocabulary[token_id]
            # Skip special tokens in output
            if token not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                if token == self.UNK_TOKEN:
                    tokens.append('?')  # Placeholder for unknown
                else:
                    tokens.append(token)
    return ''.join(tokens)
```

---

## Visualization

### BPE Training Progress

```
Iteration 0 (Characters):
┌───────────────────────────────────────────────────────┐
│ l │ o │ w │   │ l │ o │ w │ e │ r │   │ l │ o │ w │...│
└───────────────────────────────────────────────────────┘
16 tokens

Iteration 1 (Merge: l+o → lo):
┌─────────────────────────────────────────────────┐
│ lo  │ w │   │ lo  │ w │ e │ r │   │ lo  │ w │...│
└─────────────────────────────────────────────────┘
13 tokens

Iteration 2 (Merge: lo+w → low):
┌───────────────────────────────────────┐
│ low │   │ low │ e │ r │   │ low │ e │...│
└───────────────────────────────────────┘
10 tokens

Iteration 3 (Merge: low+e → lowe):
┌─────────────────────────────────┐
│ low │   │ lowe │ r │   │ lowe │...│
└─────────────────────────────────┘
8 tokens
```

### Vocabulary Growth vs. Sequence Compression

```
Vocab Size:  11  →  12  →  13  →  14  →  15
             ▲      ▲      ▲      ▲      ▲
Merges:      0      1      2      3      4

Sequence:    16  →  13  →  10  →  8   →  ...
Length       ▼      ▼      ▼      ▼
             -19%   -23%   -20%
```

### Token Distribution (Real-World Example)

In GPT-2's tokenizer with ~50,000 tokens:

```
Common words (single tokens):
┌─────────────────┬────────────┐
│ Token           │ Frequency  │
├─────────────────┼────────────┤
│ "the"           │ Very high  │
│ "and"           │ Very high  │
│ "is"            │ High       │
│ "in"            │ High       │
└─────────────────┴────────────┘

Subword decomposition (rare words):
"tokenization" → ["token", "ization"]
"unprecedented" → ["un", "pre", "ced", "ented"]
```

---

## Trade-offs and Considerations

### Vocabulary Size

| Size            | Pros                       | Cons                                  |
| --------------- | -------------------------- | ------------------------------------- |
| Small (<10K)    | Fast training, small model | Long sequences, OOV issues            |
| Medium (30-50K) | Good balance               | Standard choice for most LLMs         |
| Large (>100K)   | Short sequences            | Slower training, embedding table size |

### Computational Complexity

- **Training**: O(n \* vocab_size) where n = corpus size
- **Encoding**: O(m \* k) where m = text length, k = number of merges
- **Decoding**: O(1) per token (simple lookup)

### Real-World Tokenizer Statistics

| Model   | Tokenizer      | Vocab Size | Avg chars/token |
| ------- | -------------- | ---------- | --------------- |
| GPT-2   | BPE            | 50,257     | ~4              |
| GPT-3/4 | BPE (tiktoken) | ~100,000   | ~4              |
| BERT    | WordPiece      | 30,522     | ~4              |
| LLaMA   | SentencePiece  | 32,000     | ~4              |

---

## Try It Yourself

Run the tokenizer demo to see BPE in action:

```bash
python -m src.tokenizer
```

Or experiment in Python:

```python
from src.tokenizer import BPETokenizer

# Create and train a tokenizer
tokenizer = BPETokenizer()
tokenizer.train("To be or not to be, that is the question.", vocabulary_size=50)

# Encode text
tokens = tokenizer.encode("To be or not")
print(f"Tokens: {tokens}")

# Decode back
text = tokenizer.decode(tokens)
print(f"Decoded: {text}")

# Check the vocabulary
print(f"Vocabulary size: {tokenizer.vocabulary_size}")
print(f"Merges learned: {len(tokenizer.merges)}")
```

---

## References

1. **Original BPE Paper**: [Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

2. **Hugging Face NLP Course**: [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)

3. **Karpathy's GPT Tokenizer Lecture**: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

4. **Sebastian Raschka's Tutorial**: [Implementing a BPE Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

5. **This Repository**: See [src/tokenizer.py](src/tokenizer.py) for the complete implementation with educational comments.

---

**Next Step**: Once text is tokenized into IDs, we need to convert these IDs into meaningful numerical representations. Continue to [02 - Embeddings.md](02%20-%20Embeddings.md) to learn how.
