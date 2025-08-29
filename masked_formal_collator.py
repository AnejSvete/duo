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

        # --- Logic for 'do_not_mask' ---

        # 1. Create a boolean mask for all '#' tokens
        is_hash = input_ids == self.hash_token_id

        # 2. Use cumsum to find the *first* '#' in each row
        hash_counts = is_hash.cumsum(dim=1)
        is_first_hash = is_hash & (hash_counts == 1)

        # 3. Create the 'do_not_mask' tensor
        cutoff_indices = torch.argmax(is_first_hash.int(), dim=1)
        col_indices = torch.arange(input_ids.shape[1], device=input_ids.device)
        do_not_mask = col_indices <= cutoff_indices.unsqueeze(1)

        # 4. Handle the important edge case where a row has no '#' tokens at all
        has_no_hash = ~is_hash.any(dim=1)
        do_not_mask[has_no_hash] = True

        # --- Debugging Print Statements ---
        print(f"Text: {texts[:3]}")
        print(f"Input IDs: {input_ids[:3]}")
        print(f"Is hash: {is_hash[:3]}")
        print(f"Hash counts: {hash_counts[:3]}")
        print(f"Attention Mask: {attention_mask[:3]}")
        print(f"Do Not Mask: {do_not_mask[:3]}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_not_mask": do_not_mask,
        }
