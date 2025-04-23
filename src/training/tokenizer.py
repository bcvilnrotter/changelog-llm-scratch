"""
A simple ByteLevelBPETokenizer implementation.
"""

from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path
from collections import Counter
import regex as re
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase

class SimpleTokenizer(PreTrainedTokenizerBase):
    """A basic tokenizer implementing byte-level BPE."""
    
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None  # No fast version available
    can_save_slow_tokenizer = True
    is_fast = False

    def __init__(  # type: ignore
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]"
    ):
        kwargs = {
            "model_input_names": self.model_input_names,
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
        }
        super().__init__(**kwargs)
        
        # Load vocabulary if provided
        if vocab_file and Path(vocab_file).exists():
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        else:
            # Initialize with special tokens
            self.vocab = {
                self.pad_token: 0,
                self.bos_token: 1,
                self.eos_token: 2,
                self.unk_token: 3
            }
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Load merges if provided
        if merges_file and Path(merges_file).exists():
            with open(merges_file, 'r', encoding='utf-8') as f:
                # Load merges and validate them
                self.merges = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.merges.append(tuple(parts))
                    else:
                        # Skip invalid merges
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Skipping invalid merge during loading: {line.strip()}")
        else:
            self.merges = []
            
        # Enhanced tokenization pattern with more specific handling of common patterns
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|n't|\s+\p{L}+|\s+\p{N}+|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    @property
    def vocab_size(self) -> int:
        """Get the size of vocabulary."""
        return len(self.vocab)
        
    def save_vocabulary(self, save_directory: str) -> Tuple[str, str]:
        """Save the tokenizer vocabulary and merges to files."""
        vocab_file = Path(save_directory) / "vocab.json"
        merges_file = Path(save_directory) / "merges.txt"
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)
            
        with open(merges_file, 'w', encoding='utf-8') as f:
            for merge in self.merges:
                # Validate that merge is a tuple with at least 2 elements
                if isinstance(merge, tuple) and len(merge) >= 2:
                    f.write(f"{merge[0]} {merge[1]}\n")
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Skipping invalid merge: {merge}")
                
        return str(vocab_file), str(merges_file)
        
    def _tokenize(self, text: str, dropout_prob: float = 0.0) -> List[str]:
        """
        Tokenize text into subwords with optional BPE-dropout.
        
        Args:
            text: The text to tokenize
            dropout_prob: Probability of dropping a merge (0.0 = no dropout, 0.1 = 10% dropout)
        """
        import random  # Import at function level to avoid circular imports
        
        tokens = []
        
        for token in re.findall(self.pat, text):
            token = token.strip()
            if not token:
                continue
                
            # Apply BPE with dropout
            word = tuple(token)
            while len(word) > 1:
                pairs = list(zip(word[:-1], word[1:]))
                if not pairs:
                    break
                
                # Apply dropout to valid pairs - randomly skip some merges during training
                valid_pairs = []
                for pair in pairs:
                    if pair in self.merges and (dropout_prob == 0.0 or random.random() > dropout_prob):
                        valid_pairs.append(pair)
                
                if not valid_pairs:
                    break
                
                # Use the first valid merge in the list (earliest learned merge)
                bigram = valid_pairs[0]
                i = pairs.index(bigram)
                word = word[:i] + (bigram[0] + bigram[1],) + word[i+2:]
            
            tokens.extend(word)
            
        return tokens
        
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID in the vocabulary."""
        return self.vocab.get(token, self.vocab[self.unk_token])
        
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token in the vocabulary."""
        return self.ids_to_tokens.get(index, self.unk_token)
        
    def _convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        return " ".join(tokens)
        
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is special and [0] else."""
        if already_has_special_tokens:
            return [1 if token in [self.pad_token_id, self.eos_token_id, self.bos_token_id] else 0
                    for token in token_ids_0]
        
        mask = [0] * len(token_ids_0) + [1]  # For eos_token
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]  # For eos_token
        return mask

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create a mask from the two sequences passed."""
        eos = [self.eos_token_id]
        
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + len(eos))
        return [0] * (len(token_ids_0) + len(eos)) + [1] * (len(token_ids_1) + len(eos))

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> str:
        """Prepare text for tokenization."""
        return text

    def get_config(self) -> Dict:
        """Return the tokenizer configuration."""
        return {
            "model_input_names": self.model_input_names,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
            "do_lower_case": self.do_lower_case,
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save the tokenizer configuration and vocabulary."""
        config_file = Path(save_directory) / "tokenizer_config.json"
        vocab_files = self.save_vocabulary(save_directory)
        
        # Save config
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_config(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a tokenizer from a pretrained model."""
        config_file = Path(pretrained_model_name_or_path) / "tokenizer_config.json"
        vocab_file = Path(pretrained_model_name_or_path) / "vocab.json"
        merges_file = Path(pretrained_model_name_or_path) / "merges.txt"
        
        if not config_file.exists() or not vocab_file.exists():
            raise ValueError(f"No tokenizer files found in {pretrained_model_name_or_path}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Only pass the relevant arguments to __init__
        init_kwargs = {
            "vocab_file": str(vocab_file),
            "merges_file": str(merges_file) if merges_file.exists() else None,
            "pad_token": config.get("pad_token", "[PAD]"),
            "bos_token": config.get("bos_token", "[BOS]"),
            "eos_token": config.get("eos_token", "[EOS]"),
            "unk_token": config.get("unk_token", "[UNK]")
        }
        
        return cls(**init_kwargs)


    def get_added_vocab(self) -> Dict[str, int]:
        """Get dictionary of added tokens."""
        base_vocab = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        return {k: v for k, v in self.vocab.items() if k not in base_vocab}

    @property
    def do_lower_case(self) -> bool:
        """Whether this tokenizer does lower case."""
        return False

    def __len__(self) -> int:
        """Size of vocabulary."""
        return len(self.vocab)

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Main entry point for tokenization."""
        if isinstance(text, str):
            return self.encode(
                text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors
            )
        else:
            # Handle batch input
            batch_outputs = [
                self.encode(
                    t,
                    text_pair=text_pair[i] if text_pair else None,
                    add_special_tokens=add_special_tokens,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors=None
                )
                for i, t in enumerate(text)
            ]
            
            # Combine outputs
            combined = {
                k: [d[k] for d in batch_outputs]
                for k in batch_outputs[0].keys()
            }
            
            # Handle padding for batches
            if padding:
                max_len = max(len(ids) for ids in combined["input_ids"])
                for key in combined:
                    combined[key] = [
                        x + [self.pad_token_id if key == "input_ids" else 0] * (max_len - len(x))
                        for x in combined[key]
                    ]
            
            # Convert to tensors if requested
            if return_tensors == "pt":
                combined = {k: torch.tensor(v) for k, v in combined.items()}
                
            return combined

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to their IDs."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def encode(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        dropout_prob: float = 0.0  # Add dropout_prob parameter
    ) -> torch.Tensor:
        """
        Tokenize text and convert to tensor.
        
        Args:
            text: The text to tokenize
            truncation: Whether to truncate to max_length
            max_length: Maximum sequence length
            return_tensors: Type of tensors to return ("pt" for PyTorch)
            dropout_prob: Probability of BPE dropout (0.0 = no dropout)
        """
        # Tokenize text with optional dropout (use dropout during training, not inference)
        tokens = self._tokenize(text, dropout_prob=dropout_prob)
        if truncation and max_length:
            tokens = tokens[:max_length-1]  # Leave room for EOS token
            
        # Convert to IDs
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        token_ids.append(self._convert_token_to_id(self.eos_token))
        
        # Convert to tensor
        if return_tensors == "pt":
            return torch.tensor([token_ids])
        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Convert token ids back to string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [
                self._convert_token_to_id(self.pad_token),
                self._convert_token_to_id(self.eos_token),
                self._convert_token_to_id(self.bos_token)
            ]:
                continue
            tokens.append(self._convert_id_to_token(token_id))
            
        text = " ".join(tokens)
        if clean_up_tokenization_spaces:
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            
        return text
        
    def train(
        self,
        files: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        vocab_size: int = 10000,  # Increased from 5000 to 10000
        min_frequency: int = 2
    ) -> None:
        """Train the tokenizer on texts or files."""
        # Count word frequencies
        counter = Counter()
        
        # Process files if provided
        if files:
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    counter.update(re.findall(self.pat, text))
        
        # Process texts if provided
        if texts:
            for text in texts:
                counter.update(re.findall(self.pat, text))
                
        # Filter by frequency and initialize vocabulary
        vocab = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        
        # Determine how many merges to allow
        reserved_merges = 5000
        target_token_limit = vocab_size - reserved_merges
        
        # Add most common tokens
        for token, count in counter.most_common():
            if count < min_frequency:
                break
            if len(vocab) >= target_token_limit:
                break
            if token not in vocab:
                vocab[token] = len(vocab)
                
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        
        # Learn BPE merges
        word_freqs = Counter()
        for token, count in counter.items():
            if count < min_frequency:
                continue
            word_freqs[tuple(token)] = count
            
        num_merges = min(reserved_merges, vocab_size - len(vocab))
        for _ in range(num_merges):
            pairs = Counter()
            for word, freq in word_freqs.items():
                if len(word) == 1:
                    continue
                for i in range(len(word) - 1):
                    pairs[word[i], word[i + 1]] += freq
                    
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # Update word frequencies
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                if len(word) == 1:
                    new_word_freqs[word] = freq
                    continue
                    
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
                
            word_freqs = new_word_freqs
        
        for merge in self.merges:
            merged_token = merge[0] + merge[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[merged_token]] = merged_token
