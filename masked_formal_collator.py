# masked_formal_collator.py
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer


class MaskedFormalCollator:
    """
    A data collator that tokenizes formal language traces and creates a
    boolean mask to protect the initial formula from being masked during training.

    The collator identifies the '#' symbol in each trace and generates a
    `do_not_mask` tensor, which is True for all tokens up to and including '#'
    and False for all subsequent tokens.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get the token ID for the '#' separator.
        # It's crucial that '#' is a distinct token in your vocabulary.
        # If it's not, you may need to add it as a special token.
        self.hash_token_id = self.tokenizer.convert_tokens_to_ids("#")
        if self.hash_token_id == self.tokenizer.unk_token_id:
            print("Warning: '#' is not in the tokenizer vocabulary and maps to UNK.")
            print("Conditional masking may not work as intended.")

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Processes a list of examples to create a batch tensor."""
        texts = [item["text"] for item in batch]

        tokenized_output = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]

        # --- START: Robust 'do_not_mask' Creation ---

        # 1. Create a boolean mask for all '#' tokens
        is_hash = input_ids == self.hash_token_id

        # 2. Find the index of the first '#' in each row.
        # To robustly handle rows that have NO '#', we can find the index
        # of the first non-padding token instead as a fallback.
        is_not_padding = attention_mask.bool()

        # We want the cutoff to be the first '#' or, if none, the end of the real sequence.
        # Let's find the length of each sequence (where attention_mask is 1)
        seq_lengths = is_not_padding.sum(dim=1)

        # Create a default cutoff index pointing to the end of each sequence.
        # We use seq_lengths - 1 because indices are 0-based.
        cutoff_indices = seq_lengths - 1

        # Find the index of the first hash mark, if it exists
        hash_indices = torch.argmax(is_hash.int(), dim=1)

        # A boolean mask where True means a hash mark was found in the row.
        # (argmax returns 0 if nothing is found, so we check if the token at that index is actually a hash)
        hash_found = is_hash[torch.arange(is_hash.shape[0]), hash_indices]

        # Where a hash was found, update the cutoff_indices to that position.
        # Otherwise, it remains the end of the sequence.
        cutoff_indices[hash_found] = hash_indices[hash_found]

        # 3. Create the 'do_not_mask' using the calculated cutoff indices.
        col_indices = torch.arange(input_ids.shape[1], device=input_ids.device)
        do_not_mask = col_indices <= cutoff_indices.unsqueeze(1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_not_mask": do_not_mask,
        }
